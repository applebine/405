"""
rPPG 控制台测试程序（无需康复机器人、无需完整界面）

直接测试：摄像头 → Haar 自动人脸检测 → KLT 跟踪 → POS 算法 → 心率/血氧

用法：
    cd feedback2.0
    python test_rppg_console.py

将脸正对摄像头，保持静止，约 10-15 秒后开始打印心率/血氧数值。
程序 60 秒后自动退出。
"""
import sys

from PySide6.QtCore import QCoreApplication, QTimer

from ThreadAll.ThreadRppg import RppgThread


def main():
    app = QCoreApplication(sys.argv)

    thread = RppgThread(camera_id=0)

    def on_metrics(m):
        hr = m.get('heart_rate')
        spo2 = m.get('spo2')
        br = m.get('breath_rate')
        hrv = m.get('hrv')
        parts = []
        if hr is not None:
            parts.append(f"心率 {hr} BPM")
        if spo2 is not None:
            parts.append(f"血氧 {spo2}%")
        if br is not None:
            parts.append(f"呼吸率 {br} 次/分")
        if hrv is not None:
            parts.append(f"HRV {hrv}")
        if parts:
            print("[rPPG] " + " | ".join(parts))

    def on_face(has_face):
        print("[rPPG] " + ("检测到人脸，开始跟踪" if has_face else "人脸丢失，重新检测"))

    def on_error(msg):
        print("[rPPG] 错误: " + msg)

    def on_init(ok):
        if ok:
            print("[rPPG] 初始化成功！请将脸正对摄像头，保持静止...")
        else:
            print("[rPPG] 初始化失败，请检查摄像头")

    thread.metrics_ready.connect(on_metrics)
    thread.face_status.connect(on_face)
    thread.error_occurred.connect(on_error)
    thread.initialized.connect(on_init)

    thread.start()

    # 60 秒后自动退出
    QTimer.singleShot(60000, app.quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
