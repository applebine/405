"""
rPPG 独立进程服务器

把摄像头采集 + 人脸检测 + POS 算法放在独立进程里运行，
通过本地 TCP socket 把生理指标发给 feedback2.0 主进程。

为什么独立进程：Jetson 自带的 OpenCV 4.2 编译时链接了 Qt5，
与 feedback2.0 用的 PySide6(Qt6) 在同一进程里会 ABI 冲突，
导致 QtWebEngine 渲染黑屏。独立进程后 cv2 与 Qt 互不干扰。

用法（由 main.py 自动启动）：
    python rppg_server.py [port] [camera_id]
默认端口 8005，摄像头 0。
"""

import os
import sys
import json
import socket
import shutil
import tempfile
import time

# 预加载 libgomp（防 TLS 错误），先于 cv2
import ctypes
try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import numpy as np
import cv2

from rppg.pos import POS
from rppg.skin_klt_tracker import SkinKLTTracker, compute_region_rgb_means

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_HAAR_CASCADE = os.path.join(_PROJECT_DIR, 'rppg', 'haarcascade_frontalface_default.xml')


def _load_cascade(path):
    """加载 Haar 级联分类器（兼容中文/非 ASCII 路径）。"""
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


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8005
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # ---- TCP 服务端 ----
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen(1)
    print(f"[rPPG] 服务器监听 127.0.0.1:{port}")

    # ---- 摄像头 + 算法 ----
    face_cascade = _load_cascade(_HAAR_CASCADE)
    if face_cascade.empty():
        print("[rPPG] 错误: Haar 人脸模型加载失败")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[rPPG] 错误: 无法打开摄像头 {camera_id}")
        return

    klt = SkinKLTTracker()
    pos = POS()

    conn_holder = {'conn': None}

    def _send(obj):
        if conn_holder['conn'] is None:
            return
        try:
            conn_holder['conn'].sendall((json.dumps(obj) + '\n').encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def on_rppg_result(rppg_signal, rppg_filtered, bpm, rr, spo2, hrv, fps):
        metrics = {}
        if bpm is not None:
            metrics['heart_rate'] = round(float(bpm), 1)
        if spo2 is not None:
            metrics['spo2'] = round(float(spo2), 1)
        if rr is not None:
            metrics['breath_rate'] = round(float(rr), 1)
        if hrv is not None:
            metrics['hrv'] = hrv
        if metrics:
            _send(metrics)

    pos.set_callback(on_rppg_result)

    # ---- 等主进程连接 ----
    print(f"[rPPG] 摄像头 {camera_id} 已启动，等待主进程连接...")
    conn, addr = server.accept()
    conn_holder['conn'] = conn
    print(f"[rPPG] 主进程已连接: {addr}")

    tracking = False
    detect_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if not tracking:
                detect_count += 1
                if detect_count % 5 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        klt.select_roi(frame, (x, y, w, h))
                        tracking = True
                        print(f"[rPPG] 检测到人脸 ({x},{y},{w}x{h})，开始跟踪")
                        _send({'face': True})

            if tracking:
                smooth_box, _ = klt.track(frame)
                if smooth_box is None:
                    tracking = False
                    klt.initialized = False
                    print("[rPPG] 人脸跟踪丢失，重新检测...")
                    _send({'face': False})
                else:
                    mask = klt.get_roi_and_mask(frame, smooth_box)
                    if np.any(mask == 255):
                        skin_mean = compute_region_rgb_means(frame, mask)
                        pos.add_rgb(skin_mean)

            time.sleep(0.03)  # ~30fps

    except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        cap.release()
        pos.stop()
        try:
            conn.close()
        except Exception:
            pass
        server.close()
        print("[rPPG] 进程已退出")


if __name__ == '__main__':
    main()
