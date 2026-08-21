"""
手写 POS 算法的 rPPG 生理信号线程（不依赖 open-rppg）

流程：
    摄像头采集 → Haar 自动人脸检测（首次/丢失后）→ KLT 光流跟踪
    → 肤色提取 RGB 均值 → POS 算法 → 心率/血氧/呼吸率/HRV
    → Qt 信号发射给 feedback2.0 界面

依赖：numpy, scipy, opencv-python（Python 3.8 全兼容，无额外依赖）
人脸检测：opencv 自带 Haar Cascade（haarcascade_frontalface_default.xml）

重要：cv2 在 run() 内部延迟导入，避免 OpenCV 在 Qt/GLX 初始化前加载，
     导致 Jetson 上 QtWebEngine 报 "Could not initialize GLX" 崩溃。
"""

import os
import shutil
import tempfile

from PySide6.QtCore import QThread, Signal

from rppg.pos import POS   # 只依赖 numpy/scipy，不 import cv2，可安全在顶部导入

# 延迟导入：在 run() 里赋值（见下方 global 声明）
_cv2 = None
_np = None
_SkinKLTTracker = None
_compute_region_rgb_means = None

# 人脸检测模型路径
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HAAR_CASCADE = os.path.join(_PROJECT_DIR, 'rppg', 'haarcascade_frontalface_default.xml')


def _load_cascade(path):
    """加载 Haar 级联分类器（兼容非 ASCII 路径）。依赖全局 _cv2。"""
    def _is_ascii(s):
        try:
            s.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False

    if _is_ascii(path):
        cascade = _cv2.CascadeClassifier(path)
        if not cascade.empty():
            return cascade

    # 非 ASCII 路径（或直接加载失败），复制到临时目录再加载
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), 'haarcascade_frontalface_default.xml')
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) != os.path.getsize(path):
            shutil.copyfile(path, tmp_path)
        cascade2 = _cv2.CascadeClassifier(tmp_path)
        if not cascade2.empty():
            return cascade2
    except Exception as e:
        print(f"[rPPG] 复制模型到临时目录失败: {e}")

    return _cv2.CascadeClassifier(path)


class RppgThread(QThread):
    """rPPG 生理信号采集线程（手写 POS + 自动人脸检测）"""

    metrics_ready = Signal(dict)
    face_status = Signal(bool)
    error_occurred = Signal(str)
    initialized = Signal(bool)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.is_running = False
        self.cap = None

        # POS 算法（不依赖 cv2，可在此初始化）
        self.pos = POS()
        self.pos.set_callback(self._on_rppg_result)

        # 依赖 cv2 的对象，延迟到 run() 里初始化
        self.klt_tracker = None
        self.face_cascade = None

        self.tracking = False
        self.detect_frame_count = 0
        self._last_face_status = None

    def run(self):
        """主循环"""
        global _cv2, _np, _SkinKLTTracker, _compute_region_rgb_means

        # 延迟导入 cv2 及其依赖，避免与 Qt/GLX 初始化冲突（Jetson 上 OpenCV 可能带 OpenGL/Qt）
        import cv2 as _cv2
        import numpy as _np
        from rppg.skin_klt_tracker import (
            SkinKLTTracker as _SkinKLTTracker,
            compute_region_rgb_means as _compute_region_rgb_means,
        )

        # 人脸检测器
        cascade_path = _HAAR_CASCADE
        if not os.path.exists(cascade_path):
            cascade_path = _cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = _load_cascade(cascade_path)
        if self.face_cascade is None or self.face_cascade.empty():
            self.error_occurred.emit("Haar 人脸检测模型加载失败")
            self.initialized.emit(False)
            return

        # KLT 跟踪器
        self.klt_tracker = _SkinKLTTracker()

        # 打开摄像头
        self.cap = _cv2.VideoCapture(self.camera_id)
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
                    self.tracking = False
                    self.klt_tracker.initialized = False
                    print("[rPPG] 人脸跟踪丢失，重新检测...")
                    self._emit_face_status(False)
                else:
                    full_mask = self.klt_tracker.get_roi_and_mask(frame, smooth_box)
                    if _np.any(full_mask == 255):
                        skin_mean = _compute_region_rgb_means(frame, full_mask)
                        self.pos.add_rgb(skin_mean)

            self.msleep(30)

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.pos.stop()
        print("[rPPG] 线程已退出")

    def _detect_face(self, frame):
        """Haar 级联检测人脸，返回最大的脸 (x, y, w, h) 或 None"""
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
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
