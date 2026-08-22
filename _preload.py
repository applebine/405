"""
预加载 libgomp.so.1

解决 Jetson (aarch64) 上 import cv2 时的错误：
    ImportError: /lib/aarch64-linux-gnu/libgomp.so.1:
    cannot allocate memory in static TLS block

原理：libgomp 需要静态 TLS 内存块，必须在 Qt 等大量库加载之前
预加载它，否则进程静态 TLS 空间被占满后 import cv2 会失败。

必须在「任何 Qt/PySide6 import 之前」import 本模块。
Windows 上 libgomp.so.1 不存在，会被 try-except 安全忽略。
"""

import ctypes

try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    # 非 Linux 或找不到该库时忽略（Windows/macOS 不受影响）
    pass
