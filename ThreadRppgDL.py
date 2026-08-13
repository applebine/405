"""
DL rPPG 生理信号监测线程

使用 open-rppg 库内置的 FacePhys 深度学习模型，从摄像头实时提取
心率(HR)、呼吸率(BR)、心率变异性(HRV) 等生理指标，
通过 Qt 信号发送给 feedback2.0 界面显示。

硬件要求：USB 摄像头（索引 0）
运行依赖：pip install open-rppg
          （numpy, opencv-python, PySide6 为基础依赖）

适配自 DL_rPPG_complete_version/main.py 的 CameraManager + ModelManager。
"""

import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None

from PySide6.QtCore import QThread, Signal

# 尝试导入 open-rppg（未安装时设为 None）
try:
    import rppg
except ImportError:
    rppg = None


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """视频帧数据"""
    frame: Optional[object] = None
    face_box: Optional[Tuple[float, float, float, float]] = None
    timestamp: float = 0.0


@dataclass
class VitalMetrics:
    """生理指标"""
    heart_rate: Optional[float] = None
    breath_rate: Optional[float] = None
    hrv: Optional[float] = None
    spo2: Optional[float] = None


# ---------------------------------------------------------------------------
# 摄像头管理器（简化自 DL_rPPG_complete_version）
# ---------------------------------------------------------------------------

class CameraManager:
    """
    摄像头管理器

    使用 open-rppg 内置的人脸检测 + BVP 提取管线。
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.rppg_model = None
        self.is_running = False
        self.frame_generator = None
        self.video_context = None
        self._initialize_model()
        self._debug_printed = False

    def _initialize_model(self):
        """初始化 open-rppg 模型"""
        if rppg is None:
            print("[rPPG-DL] 错误: open-rppg 库未安装")
            print("[rPPG-DL] 请运行: pip install open-rppg")
            return

        try:
            self.rppg_model = rppg.Model()
            print("[rPPG-DL] open-rppg 模型初始化成功")
        except Exception as e:
            print(f"[rPPG-DL] 错误: 模型初始化失败 - {e}")
            self.rppg_model = None

    def start(self) -> bool:
        """启动摄像头"""
        if self.rppg_model is None:
            print("[rPPG-DL] 错误: rPPG 模型未初始化")
            return False

        try:
            self.video_context = self.rppg_model.video_capture(self.camera_id)
            self.video_context.__enter__()
            self.frame_generator = self.rppg_model.preview
            self.is_running = True
            print(f"[rPPG-DL] 摄像头 {self.camera_id} 已启动")
            return True
        except Exception as e:
            print(f"[rPPG-DL] 错误: 无法打开摄像头 - {e}")
            self.is_running = False
            return False

    def get_frame(self) -> Optional[FrameData]:
        """获取一帧"""
        if not self.is_running or self.frame_generator is None:
            return None

        try:
            frame, box = next(self.frame_generator)
            if not self._debug_printed:
                self._debug_printed = True
            return FrameData(frame=frame, face_box=box, timestamp=time.time())
        except StopIteration:
            return None
        except Exception as e:
            print(f"[rPPG-DL] 帧获取异常: {e}")
            return None

    def stop(self):
        """停止摄像头"""
        if self.is_running:
            try:
                if self.video_context is not None:
                    self.video_context.__exit__(None, None, None)
                self.is_running = False
                print("[rPPG-DL] 摄像头已停止")
            except Exception as e:
                print(f"[rPPG-DL] 停止时异常: {e}")


# ---------------------------------------------------------------------------
# 模型推理管理器（简化自 DL_rPPG_complete_version）
# ---------------------------------------------------------------------------

class ModelManager:
    """
    rPPG 模型推理管理器

    使用 open-rppg 内置的 model.hr() 方法直接获取 HR/HRV/呼吸率。
    """

    def __init__(self, rppg_model: Optional[object] = None, window_size: int = 10):
        self.model = rppg_model
        self.window_size = window_size
        self.last_inference_time = 0
        self.inference_interval = 1.0   # 每秒推理一次

    def infer(self) -> Optional[VitalMetrics]:
        """执行推理，返回生理指标"""
        if self.model is None:
            return None

        now = time.time()
        if now - self.last_inference_time < self.inference_interval:
            return None
        self.last_inference_time = now

        try:
            result = self.model.hr(start=-self.window_size)
            if result is None:
                return None

            metrics = VitalMetrics()

            # 心率
            if result.get('hr'):
                metrics.heart_rate = float(result['hr'])

            # HRV + 呼吸率
            hrv_data = result.get('hrv', {}) or {}
            if hrv_data.get('rmssd'):
                metrics.hrv = float(hrv_data['rmssd'])
            if hrv_data.get('breathingrate'):
                metrics.breath_rate = float(hrv_data['breathingrate']) * 60

            # 血氧（基于心率的估算，非真实测量）
            if metrics.heart_rate:
                hr = metrics.heart_rate
                if hr < 60:
                    metrics.spo2 = 98.5
                elif hr < 100:
                    metrics.spo2 = 97.5 + (100 - hr) * 0.05
                else:
                    metrics.spo2 = 96.0 + (120 - hr) * 0.02

            return metrics

        except Exception as e:
            print(f"[rPPG-DL] 推理异常: {e}")
            return None


# ---------------------------------------------------------------------------
# DL rPPG Qt 线程
# ---------------------------------------------------------------------------

class RppgDLThread(QThread):
    """
    DL rPPG 生理信号采集线程

    整合 CameraManager + ModelManager，
    通过 Qt 信号发射心率、血氧、呼吸率、HRV。
    """

    metrics_ready = Signal(dict)
    face_status = Signal(bool)
    error_occurred = Signal(str)
    initialized = Signal(bool)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.is_running = False
        self.camera_manager: Optional[CameraManager] = None
        self.model_manager: Optional[ModelManager] = None

    def run(self):
        """主循环"""

        if cv2 is None or np is None:
            self.error_occurred.emit("缺少基础依赖。请运行: pip install opencv-python numpy")
            self.initialized.emit(False)
            return

        if rppg is None:
            self.error_occurred.emit("open-rppg 库未安装。请运行: pip install open-rppg")
            self.initialized.emit(False)
            return

        self.camera_manager = CameraManager(self.camera_id)
        if self.camera_manager.rppg_model is None:
            self.error_occurred.emit("rPPG 模型初始化失败")
            self.initialized.emit(False)
            return

        self.model_manager = ModelManager(self.camera_manager.rppg_model)

        if not self.camera_manager.start():
            self.error_occurred.emit("摄像头启动失败")
            self.initialized.emit(False)
            return

        self.is_running = True
        self.initialized.emit(True)
        print("[rPPG-DL] 线程已启动，开始采集...")

        last_face_status_time = 0.0

        while self.is_running:
            # 持续消费摄像头帧（保持管线运行）
            frame_data = self.camera_manager.get_frame()
            if frame_data is None:
                self.msleep(5)
                continue

            # 依据 face_box 是否有效判断人脸（与新版 main.py 一致）
            has_face = False
            if frame_data.face_box is not None:
                box = frame_data.face_box
                try:
                    if isinstance(box, np.ndarray) and box.shape == (2, 2):
                        x1, y1, x2, y2 = float(box[0, 0]), float(box[0, 1]), float(box[1, 0]), float(box[1, 1])
                        has_face = min(x1, x2) >= 0 and min(y1, y2) >= 0 and abs(x2 - x1) > 0 and abs(y2 - y1) > 0
                    elif isinstance(box, (list, tuple)) and len(box) == 2:
                        x1, y1, x2, y2 = float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1])
                        has_face = min(x1, x2) >= 0 and min(y1, y2) >= 0 and abs(x2 - x1) > 0 and abs(y2 - y1) > 0
                except (TypeError, ValueError, IndexError):
                    has_face = False

            # 节流：人脸状态变化最多每秒上报一次
            now = time.time()
            if now - last_face_status_time >= 1.0:
                last_face_status_time = now
                self.face_status.emit(has_face)

            # 每秒推理一次（model.hr 内部有滑窗缓冲）
            metrics = self.model_manager.infer()
            if metrics is not None and metrics.heart_rate is not None:
                self.metrics_ready.emit({
                    'heart_rate': round(metrics.heart_rate, 1),
                    'spo2': round(metrics.spo2, 1) if metrics.spo2 else None,
                    'breath_rate': (
                        round(metrics.breath_rate, 1)
                        if metrics.breath_rate else None
                    ),
                    'hrv': round(metrics.hrv, 1) if metrics.hrv else None,
                })

            self.msleep(5)

        self.camera_manager.stop()
        print("[rPPG-DL] 线程已退出")

    def stop(self):
        """安全停止：先中断阻塞的帧采集，再退出循环"""
        self.is_running = False
        if self.camera_manager is not None:
            self.camera_manager.stop()   # 中断 get_frame() 的 next() 阻塞
