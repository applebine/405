"""
手写 POS 算法的 rPPG 生理信号线程（不依赖 open-rppg）

流程：
    摄像头采集 → Haar 自动人脸检测（首次/丢失后）→ KLT 光流跟踪
    → 肤色提取 RGB 均值 → POS 算法 → 心率/血氧/呼吸率/HRV
    → Qt 信号发射给 feedback2.0 界面

依赖：numpy, scipy, opencv-python（Python 3.8 全兼容，无额外依赖）
人脸检测：opencv 自带 Haar Cascade（haarcascade_frontalface_default.xml）
"""

import os
import shutil
import tempfile

import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal

from rppg.pos import POS
from rppg.skin_klt_tracker import SkinKLTTracker, compute_region_rgb_means

# 人脸检测模型路径：优先用项目内自带文件（不依赖 cv2 打包情况），
# 找不到时 fallback 到 cv2.data.haarcascades
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HAAR_CASCADE = os.path.join(_PROJECT_DIR, 'rppg', 'haarcascade_frontalface_default.xml')
if not os.path.exists(_HAAR_CASCADE):
    _HAAR_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def _load_cascade(path):
    """加载 Haar 级联分类器。

    OpenCV 在 Windows 上无法读取含中文/非 ASCII 字符的路径，
    因此：路径为纯 ASCII 时直接加载；否则先复制到系统临时目录（纯 ASCII）再加载。
    """
    def _is_ascii(s):
        try:
            s.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False

    if _is_ascii(path):
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            return cascade

    # 非 ASCII 路径（或直接加载失败），复制到临时目录再加载
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), 'haarcascade_frontalface_default.xml')
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) != os.path.getsize(path):
            shutil.copyfile(path, tmp_path)
        cascade2 = cv2.CascadeClassifier(tmp_path)
        if not cascade2.empty():
            return cascade2
    except Exception as e:
        print(f"[rPPG] 复制模型到临时目录失败: {e}")

    return cv2.CascadeClassifier(path)


class RppgThread(QThread):
    """rPPG 生理信号采集线程（手写 POS + 自动人脸检测）"""

    # 生理指标信号：{'heart_rate', 'spo2', 'breath_rate', 'hrv'}
    metrics_ready = Signal(dict)
    # 人脸检测状态（True=检测到人脸，False=丢失）
    face_status = Signal(bool)
    # 错误信号
    error_occurred = Signal(str)
    # 初始化完成信号
    initialized = Signal(bool)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.is_running = False
        self.cap = None

        # 人脸检测器（项目自带模型，兼容中文路径）
        self.face_cascade = _load_cascade(_HAAR_CASCADE)

        # KLT 跟踪器 + POS 算法
        self.klt_tracker = SkinKLTTracker()
        self.pos = POS()
        self.pos.set_callback(self._on_rppg_result)

        # 状态
        self.tracking = False          # 是否正在跟踪人脸
        self.detect_frame_count = 0    # 未跟踪时的帧计数（每 N 帧跑一次 Haar）
        self._last_face_status = None  # 用于状态变化时才发信号

    def run(self):
        """主循环"""
        if self.face_cascade.empty():
            self.error_occurred.emit("Haar 人脸检测模型加载失败")
            self.initialized.emit(False)
            return

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"无法打开摄像头 {self.camera_id}")
            self.initialized.emit(False)
            return

        self.is_running = True
        self.initialized.emit(True)
        print(f"[rPPG] 摄像头 {self.camera_id} 已启动，开始自动人脸检测...")

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.msleep(10)
                continue

            # ---- 未跟踪：每隔几帧用 Haar 检测人脸 ----
            if not self.tracking:
                self.detect_frame_count += 1
                if self.detect_frame_count % 5 == 0:
                    face = self._detect_face(frame)
                    if face is not None:
                        x, y, w, h = face
                        self.klt_tracker.select_roi(frame, (x, y, w, h))
                        self.tracking = True
                        print(f"[rPPG] 检测到人脸 ({x},{y},{w}x{h})，开始跟踪")
                        self._emit_face_status(True)

            # ---- 正在跟踪：KLT 光流 + 肤色 + POS ----
            if self.tracking:
                smooth_box, points = self.klt_tracker.track(frame)

                if smooth_box is None:
                    # 跟踪丢失（特征点不足），回到 Haar 检测
                    self.tracking = False
                    self.klt_tracker.initialized = False
                    print("[rPPG] 人脸跟踪丢失，重新检测...")
                    self._emit_face_status(False)
                else:
                    # 皮肤掩膜 → RGB 均值 → POS
                    full_mask = self.klt_tracker.get_roi_and_mask(frame, smooth_box)
                    if np.any(full_mask == 255):
                        skin_mean = compute_region_rgb_means(frame, full_mask)
                        self.pos.add_rgb(skin_mean)

            # 帧率控制（约 30fps）
            self.msleep(30)

        # 清理
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.pos.stop()
        print("[rPPG] 线程已退出")

    def _detect_face(self, frame):
        """Haar 级联检测人脸，返回最大的脸 (x, y, w, h) 或 None"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) > 0:
            return max(faces, key=lambda f: f[2] * f[3])  # 面积最大的脸
        return None

    def _emit_face_status(self, has_face: bool):
        """仅在状态变化时发射信号"""
        if has_face != self._last_face_status:
            self._last_face_status = has_face
            self.face_status.emit(has_face)

    def _on_rppg_result(self, rppg_signal, rppg_filtered, bpm,
                        respiratory_rate, spo2, hrv, fps):
        """POS 算法回调：收到新的生理指标"""
        metrics = {}
        if bpm is not None:
            metrics['heart_rate'] = round(float(bpm), 1)
        if spo2 is not None:
            metrics['spo2'] = round(float(spo2), 1)
        if respiratory_rate is not None:
            metrics['breath_rate'] = round(float(respiratory_rate), 1)
        if hrv is not None:
            metrics['hrv'] = hrv

        if metrics:
            self.metrics_ready.emit(metrics)

    def stop(self):
        """安全停止"""
        self.is_running = False
