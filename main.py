import _preload  # 必须先于 Qt 导入：预加载 libgomp，解决 Jetson 上 cv2 的 TLS 错误
import sys
import socket
import threading
import subprocess
import json
import os
import pandas as pd
import serial.tools.list_ports
from PySide6.QtWidgets import QApplication, QWidget, QFrame, QLabel, QSpacerItem, QSizePolicy, QVBoxLayout
from PySide6.QtCore import Slot, QTimer, QTime, QThread, Qt, Signal

from Ui.showUi import Ui_Form  # 这里的类名取决于你在 Qt Designer 中的顶层对象类型
from ThreadAll.ThreadCommand import CommandWorker
from ThreadAll.ThreadGetMotorData import GetMotorDataThread
from ThreadAll.ThreadGetTorqueData import GetTorqueThread
from ThreadAll.ThreadWebgl import WebglThread

# 在 import 阶段就固定项目绝对路径（WebglThread 运行后会 os.chdir 改变工作目录，
# 后续不能再依赖 os.path.abspath(__file__) 计算路径）
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class RppgClientThread(QThread):
    """rPPG 客户端线程：连接独立的 rppg_server 进程，接收生理指标。

    主进程不 import cv2，避免与 Qt6 的 ABI 冲突（Jetson 黑屏问题）。
    """
    metrics_received = Signal(dict)
    face_received = Signal(bool)
    error_received = Signal(str)

    def __init__(self, port=8005):
        super().__init__()
        self.port = port
        self.running = True

    def run(self):
        s = None
        # 重试连接（等 rppg_server 子进程启动）
        while self.running:
            try:
                s = socket.create_connection(('127.0.0.1', self.port), timeout=3)
                break
            except OSError:
                self.msleep(500)
        if s is None:
            return

        buf = b''
        while self.running:
            try:
                data = s.recv(4096)
                if not data:
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line.decode('utf-8'))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if 'face' in obj:
                        self.face_received.emit(bool(obj['face']))
                    else:
                        self.metrics_received.emit(obj)
            except (OSError, ConnectionResetError):
                break
        try:
            if s:
                s.close()
        except Exception:
            pass

    def stop(self):
        self.running = False


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        """-----------------------------------------参数信息-----------------------------------------------------------"""
        self.uploadDataMode = 0                     # 0表示读取位置角度和速度信息，1表示仅获取角度信息，2表示获取扭矩信息
        self.motorId = 0                            # 电机关节id
        self.passiveOrActiveIntentionMode = -1      # 被动模式和主动意图模式
        self.kneeHipAngle = 0                       # 膝关节运动时，髋关节固定角度，用于webgl展示
        self.ankleHipAngle = 0                      # 踝关节运动时，髋关节固定角度，用于webgl展示
        self.exerciseFlag = False                   # 是否在运动的标志，也可以表示是否开始痉挛保护的标志
        self.spasmTorqueThresholds = [0.01, 0.16]   # 痉挛保护和主动意图识别的膝关节的阈值，依次为下限和上限
        self.startClassifyFlag = False              # 主动意图识别开始分类标志
        self.count0 = 0                             # 主动意图启动初始点计数

        # 肌肉激活度数据，用于webgl展示
        self.muscleActivationDegreeAnkleDatas = pd.read_csv("./muscleActivationDegree/ankle.csv").to_numpy()
        self.muscleActivationDegreeKneeDatas = pd.read_csv("./muscleActivationDegree/knee.csv").to_numpy()
        self.muscleActivationDegreeHipDatas = pd.read_csv("./muscleActivationDegree/hip.csv").to_numpy()

        # 运动软件限位，依次为髋关节角度下限，髋关节角度上限，膝关节角度下限，膝关节角度上限，踝关节角度下限和踝关节角度上限
        self.angleThresholds = [-10, 60, -130, 10, -100, 40]

        """-----------------------------------------QT控件设置-----------------------------------------------------------"""
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.stackedWidget.setCurrentWidget(self.ui.startPage)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_lcd_func)
        self.timeLeft = QTime(0, 0, 0)

        """-----------------------------------------多线程处理-----------------------------------------------------------"""
        # # 打开串口，进行通信
        ports = serial.tools.list_ports.comports()      # 获取设备连接的串口，一般都仅有一个，那就是ports[0][0]
        self.downSer = serial.Serial(port=ports[0][0], baudrate=115200, timeout=1)   # Jetson <--> STM32
        self.upSer = serial.Serial(port="/dev/ttyTHS0", baudrate=115200, timeout=1)  # PC <--> Jetson
        #
        # # 接收 PC 端命令的线程
        self.commandWorker = CommandWorker(self.upSer, self.downSer)
        self.commandThread = QThread()
        self.commandWorker.moveToThread(self.commandThread)                     # 将worker移到线程中
        self.commandWorker.commandData.connect(self.get_com_thread_func)        # 连接信号和槽
        self.commandThread.started.connect(self.commandWorker.run)              # 当线程启动时，调用worker的run方法
        self.commandThread.start()
        #
        # # 获取电机的角度和速度信息的线程
        self.getMotorDataWorker = GetMotorDataThread(self.downSer)
        self.getMotorDataWorker.motorData.connect(self.get_motor_data_func)     # 连接信号和槽
        self.getMotorDataThread = None
        #
        # # 获取扭矩传感器的扭矩信息的线程
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    # 创建一个使用IPv4的UDP协议的套接字
        self.getTorqueWorker = GetTorqueThread(self.clientSocket)
        self.getTorqueWorker.torqueData.connect(self.get_torque_func)           # 连接信号和槽
        self.getTorqueThread = None

        # 展示 WebGl 的线程
        self.webglWorker = WebglThread()
        self.webglThread = threading.Thread(target=self.webglWorker.run)
        self.webglThread.start()
        self.ui.webEngineView.load('http://localhost:8004/index.html')          # 在控件上加载 WebGL

        """----------------------------------------- rPPG DL 生理信号监测 -----------------------------------------------------------"""
        self._init_rppg_ui()
        self._init_rppg_process()

    # ============== rPPG 生理信号监测 ==============

    def _init_rppg_ui(self):
        """在 showPage 信息栏添加心率/血氧显示卡片"""
        card_style = (
            "border-radius: 15px;"
            "background-color: rgb(255, 240, 240);"
        )
        label_style = "font: 18pt \"微软雅黑\";"

        def _make_card(title, obj_name):
            frame = QFrame(self.ui.showPage)
            frame.setObjectName(obj_name)
            frame.setStyleSheet(card_style)
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setFrameShadow(QFrame.Raised)
            vbox = QVBoxLayout(frame)
            vbox.setObjectName(f"{obj_name}_vbox")
            title_lbl = QLabel(frame)
            title_lbl.setText(title)
            vbox.addWidget(title_lbl)
            value_lbl = QLabel(frame)
            value_lbl.setObjectName(f"{obj_name}_value")
            value_lbl.setStyleSheet(label_style)
            value_lbl.setAlignment(Qt.AlignCenter)
            value_lbl.setText("--")
            return frame, value_lbl

        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self._hr_frame, self._hr_label = _make_card("心率 (BPM)", "rppg_hr_frame")
        self.ui.horizontalLayout_2.addWidget(self._hr_frame)
        self.ui.horizontalLayout_2.addItem(spacer)

        self._spo2_frame, self._spo2_label = _make_card("血氧 (%)", "rppg_spo2_frame")
        self.ui.horizontalLayout_2.addWidget(self._spo2_frame)
        self.ui.horizontalLayout_2.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        print("[rPPG] UI 卡片已添加: 心率 + 血氧")

    def _init_rppg_process(self):
        """启动独立的 rPPG 进程（避免 cv2 与 Qt6 冲突），并用 socket 接收数据"""
        self.rppg_port = 8005
        self.rppg_camera_id = 0

        # 启动 rppg_server.py 子进程
        script = os.path.join(_PROJECT_DIR, 'rppg_server.py')
        log_path = os.path.join(_PROJECT_DIR, 'rppg_server.log')
        log_file = open(log_path, 'w')
        self.rppg_proc = subprocess.Popen(
            [sys.executable, script, str(self.rppg_port), str(self.rppg_camera_id)],
            stdout=log_file, stderr=subprocess.STDOUT
        )

        # socket 客户端线程
        self.rppg_client = RppgClientThread(self.rppg_port)
        self.rppg_client.metrics_received.connect(self._on_rppg_metrics)
        self.rppg_client.face_received.connect(self._on_rppg_face_status)
        self.rppg_client.error_received.connect(self._on_rppg_error)
        self.rppg_client.start()
        print("[rPPG] 独立进程已启动，等待生理数据...")

    # ---- rPPG 信号槽 ----

    def _on_rppg_metrics(self, metrics: dict):
        """更新心率/血氧标签"""
        hr = metrics.get('heart_rate')
        spo2 = metrics.get('spo2')

        if hr is not None:
            self._hr_label.setText(str(int(round(hr))))
            self._hr_label.setStyleSheet("font: 18pt '微软雅黑'; color: #d32f2f;")

        if spo2 is not None:
            self._spo2_label.setText(str(int(round(spo2))))

    def _on_rppg_face_status(self, has_face: bool):
        """人脸检测状态变化"""
        if not has_face:
            self._hr_label.setText("无脸")
            self._spo2_label.setText("无脸")

    def _on_rppg_error(self, msg: str):
        """rPPG 错误处理"""
        print(f"[rPPG] 错误: {msg}")
        self._hr_label.setText("错误")
        self._spo2_label.setText("错误")


    # 关闭定时器和线程。将webgl界面回到原位
    def end_threads_and_reset(self):
        self.timer.stop()
        self.getMotorDataWorker.stop()
        self.getTorqueWorker.stop()
        self.getTorqueWorker.wait()  # 确保线程完全退出

        # 处理初始状态，保持 WebGL 同步
        js_code = """set_hip_angle(0, 3);"""
        self.ui.webEngineView.page().runJavaScript(js_code)
        js_code = f"""set_knee_angle(0, 3);"""
        self.ui.webEngineView.page().runJavaScript(js_code)
        js_code = f"""set_ankle_angle(0, 3);"""
        self.ui.webEngineView.page().runJavaScript(js_code)

    # 处理上位机信号
    @Slot(str)
    def get_com_thread_func(self, com):
        if com[0] == "0":                           # 接收上位机电刺激的信号，在反馈界面做出显示
            js_code = f"""set_Bulb_VR({com[1]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_Bulb_VL({com[2]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_Bulb_BM({com[3]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_Bulb_ST({com[4]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_Bulb_AT({com[5]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_Bulb_PL({com[6]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_Bulb_GM({com[7]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_Bulb_SM({com[8]});"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            return                                      # 电刺激仅在反馈界面做出反应，不下发至电机控制板
        elif com[0] == "1":                         # 设置角度命令
            self.kneeHipAngle = int(com[37:40])         # 获取膝关节运动时髋关节的固定角度，用于 WebGL 同步
            self.ankleHipAngle = int(com[40:43])        # 获取踝关节运动时髋关节的固定角度，用于 WebGL 同步
        elif com[0] == "2":                         # 设置运动状态命令，并且启动运动
            # if com[1] == '1':                           # 处理运动模式的展示和记录，用于 WebGL 使用
            #     self.motorId = 1
            #     self.ui.motorIdLbl.setText("髋关节")
            # elif com[1] == '2':
            #     self.motorId = 2
            #     self.ui.motorIdLbl.setText("膝关节")
            # elif com[1] == '3':
            #     self.motorId = 3
            #     self.ui.motorIdLbl.setText("踝关节")

            # 处理运动模式的展示和记录，用于 WebGL 使用
            motor_flag = ["髋关节", "膝关节", "踝关节"]

            self.motorId = int(com[1])
            self.ui.motorIdLbl.setText(motor_flag[self.motorId - 1])

            # 处理初始状态，保持 WebGL 同步
            js_code = """set_hip_angle(0, 3);"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_knee_angle(0, 3);"""
            self.ui.webEngineView.page().runJavaScript(js_code)
            js_code = f"""set_ankle_angle(0, 3);"""
            self.ui.webEngineView.page().runJavaScript(js_code)

            if self.motorId == 1:                       # 一定要所有情况都包括，否则碰到未包括的情况，会不继续运行。
                js_code = """set_hip_angle(0, 1);"""
            elif self.motorId == 2:
                js_code = f"""set_hip_angle({self.kneeHipAngle}, 0.3);"""
            elif self.motorId == 3:
                js_code = f"""set_hip_angle({self.ankleHipAngle}, 0.3);"""
            self.ui.webEngineView.page().runJavaScript(js_code)

            if self.uploadDataMode == 1:                # 正常模式只能上传电机信息或扭矩信息，但绝不能仅上传角度信息
                self.uploadDataMode = 0

            self.ui.stackedWidget.setCurrentWidget(self.ui.showPage)    # 切换为反馈界面

            # if com[2] == '0':                           # 处理左右腿的展示
            #     self.ui.legLbl.setText("右腿")
            # elif com[2] == '1':
            #     self.ui.legLbl.setText("左腿")

            # 处理左右腿的展示
            leg_flag = ["右腿", "左腿"]
            self.ui.legLbl.setText(leg_flag[int(com[2])])

            # 处理主动运动和被动运动
            time_num = int(com[6:8])                    # 获取被动运动的时间
            if com[3] == "0":                           # 被动运动
                self.passiveOrActiveIntentionMode = 0
                self.ui.modeLbl.setText("被动运动")
                self.timeLeft = QTime(0, time_num, 0)
                time_str = self.timeLeft.toString("mm:ss")
                self.ui.timeLcd.display(time_str)
                self.timer.start(1000)                      # 开启定时器，用于更新展示的时间，更新时间单位为 1s
                self.exerciseFlag = True                    # 被动运动直接开始痉挛保护
            elif com[3] == "1":                         # 主动运动
                self.passiveOrActiveIntentionMode = 1
                self.ui.modeLbl.setText("主动意图")
                self.timeLeft = QTime(0, 59, 59)
                time_str = self.timeLeft.toString("mm:ss")
                self.ui.timeLcd.display(time_str)
                self.startClassifyFlag = True
                self.exerciseFlag = False                   # 主动运动需要在运动的时候才开始痉挛保护

            # -----------------启动获取电机位置和速度的线程-------------------------------
            if not self.getMotorDataThread or not self.getMotorDataThread.is_alive():
                self.getMotorDataWorker.reset()     # 重置停止标志
                # 创建并启动第一个线程
                self.getMotorDataThread = threading.Thread(target=self.getMotorDataWorker.run)
                self.getMotorDataThread.start()
            # -----------------启动获取扭矩信息的线程-------------------------------
            if not self.getTorqueThread or not self.getTorqueThread.is_alive():
                self.getTorqueWorker.reset()      # 重置停止标志
                # 创建并启动第一个线程
                self.getTorqueThread = threading.Thread(target=self.getTorqueWorker.run)
                self.getTorqueThread.start()
        elif com[0] == "4":                         # 暂停命令
            self.exerciseFlag = False
            if self.passiveOrActiveIntentionMode == 0:
                self.timer.stop()                       # 暂停定时器
        elif com[0] == "5":                         # 重新启动命令
            self.exerciseFlag = True
            if self.passiveOrActiveIntentionMode == 0:
                self.timer.start(1000)                  # 注意，暂停再重启回有最大1s的误差
        elif com[0] == "6":                         # 结束命令
            self.exerciseFlag = False
            self.ui.stackedWidget.setCurrentWidget(self.ui.endPage)     # 切换为结束界面
            self.ui.label_4.setText("训练已结束，可以找医生查看训练结果。\n也可以进行再一次的训练。")
            self.end_threads_and_reset()
        elif com[0] == "8":                         # 获取角度信息命令
            self.uploadDataMode = 1                     # 0表示读取位置角度和速度信息，1表示仅获取角度信息
            # -----------------启动获取电机位置和速度的线程-------------------------------
            if not self.getMotorDataThread or not self.getMotorDataThread.is_alive():
                self.getMotorDataWorker.reset()         # 重置停止标志
                # 创建并启动第一个线程
                self.getMotorDataThread = threading.Thread(target=self.getMotorDataWorker.run)
                self.getMotorDataThread.start()
        # elif com[0] == "9":                         # 该命令修改上传给上位机的是角度信息还是扭矩信息
        #     if not self.uploadDataMode == 2:
        #         self.uploadDataMode = 2                 # 上传给上位机扭矩信息，用于上位机记录扭矩信息进行数据分析
        #     else:
        #         self.uploadDataMode = 0                 # 上传给上位机电机信息，用于正常系统运行的电刺激处理
        #     return                                      # 不发给电机控制板

        self.downSer.write(com.encode('utf-8'))     # 发送指令

    def update_lcd_func(self):
        time_str = self.timeLeft.toString("mm:ss")
        self.ui.timeLcd.display(time_str)

        if self.timeLeft == QTime(0, 0, 0):
            # 将结束信息（361，-1，4）上传给上位机
            self.upSer.write((str(361) + " " + str(-1) + " " + str(4) + "\r\n").encode("utf-8"))
            self.ui.stackedWidget.setCurrentWidget(self.ui.endPage)
            self.ui.label_4.setText("训练已结束，可以找医生查看训练结果。\n也可以进行再一次的训练。")
            self.downSer.write("6\r\n".encode('utf-8'))   # 电机结束
            self.end_threads_and_reset()
        else:
            self.timeLeft = self.timeLeft.addSecs(-1)  # 每次减1秒

    @Slot(list)
    def get_motor_data_func(self, value):
        if self.uploadDataMode == 0:         # 0表示读取位置角度和速度信息，1表示仅获取角度信息
            # 软件限位
            if value[0] < self.angleThresholds[2 * (self.motorId - 1)] or \
               value[0] > self.angleThresholds[2 * (self.motorId - 1) + 1]:     # 超出限位
                self.ui.stackedWidget.setCurrentWidget(self.ui.endPage)
                self.ui.label_4.setText("检测触碰到限位，设备正在缓慢回归原位。\n请等设备停稳，再做调整！")
                self.downSer.write("6\r\n".encode('utf-8'))  # 电机结束  发送停止命令
                self.end_threads_and_reset()

            # 展示角度和速度
            if self.motorId == 3:
                self.ui.angleLbl.setText(str(value[0] + 90))
            else:
                self.ui.angleLbl.setText(str(value[0]))
            value[1] = abs(value[1])
            self.ui.speedLbl.setText(str(value[1]))

            # 主动意图识别的控制
            if self.startClassifyFlag is False and (value[0]) == 0 and int(value[1]) == 0:
                self.count0 += 1
                if self.count0 == 10:
                    self.exerciseFlag = False
                    self.startClassifyFlag = True
                    self.count0 = 0

            # 同步 WebGL 中角度的变化
            js_code = None
            muscle_activation_degree = None
            if self.motorId == 1:
                js_code = f"""set_hip_angle({value[0]}, {value[1]});"""
                muscle_activation_degree = self.muscleActivationDegreeHipDatas
            elif self.motorId == 2:
                js_code = f"""set_knee_angle({abs(value[0])}, {value[1]});"""
                muscle_activation_degree = self.muscleActivationDegreeKneeDatas
            elif self.motorId == 3:
                js_code = f"""set_ankle_angle({value[0]}, {value[1]});"""
                muscle_activation_degree = self.muscleActivationDegreeAnkleDatas

            self.ui.webEngineView.page().runJavaScript(js_code)

            # 肌肉闪烁处理
            if not int(value[0]) == 0:
                js_code = f"""set_muscle({muscle_activation_degree[int(abs(value[0]))][0]},
                                         {muscle_activation_degree[int(abs(value[0]))][1]},
                                         {muscle_activation_degree[int(abs(value[0]))][2]},
                                         {muscle_activation_degree[int(abs(value[0]))][3]},
                                         {muscle_activation_degree[int(abs(value[0]))][4]},
                                         {muscle_activation_degree[int(abs(value[0]))][5]},
                                         {muscle_activation_degree[int(abs(value[0]))][6]},
                                         {muscle_activation_degree[int(abs(value[0]))][7]}, 1);"""
                self.ui.webEngineView.page().runJavaScript(js_code)

                js_code = f"""set_slider({muscle_activation_degree[int(abs(value[0]))][0]},
                                         {muscle_activation_degree[int(abs(value[0]))][1]},
                                         {muscle_activation_degree[int(abs(value[0]))][2]},
                                         {muscle_activation_degree[int(abs(value[0]))][3]},
                                         {muscle_activation_degree[int(abs(value[0]))][4]},
                                         {muscle_activation_degree[int(abs(value[0]))][5]},
                                         {muscle_activation_degree[int(abs(value[0]))][6]},
                                         {muscle_activation_degree[int(abs(value[0]))][7]});"""
                self.ui.webEngineView.page().runJavaScript(js_code)
            else:
                if self.passiveOrActiveIntentionMode == 0:
                    js_code = """set_muscle(0, 0, 0, 0, 0, 0 ,0, 0, 0);"""
                    self.ui.webEngineView.page().runJavaScript(js_code)

                    js_code = """set_slider(0, 0, 0, 0, 0, 0 ,0, 0);"""
                    self.ui.webEngineView.page().runJavaScript(js_code)
                else:
                    js_code = """set_muscle(1, 1, 1, 1, 1, 1, 1, 1, 0);"""
                    self.ui.webEngineView.page().runJavaScript(js_code)

                    js_code = """set_slider(0, 0, 0, 0, 0, 0 ,0, 0);"""
                    self.ui.webEngineView.page().runJavaScript(js_code)

            # 上传角度信息、速度信息和阶段信息
            self.upSer.write((str(value[0]) + " " + str(value[1]) + " " + str(value[2]) + "\r\n").encode("utf-8"))
        elif self.uploadDataMode == 1:       # 0表示读取位置角度和速度信息，1表示仅获取角度信息
            self.upSer.write(str(value[0]).encode("utf-8"))
        else:
            pass

    @Slot(list)
    def get_torque_func(self, value):
        self.ui.torqueLbl.setText(str(value[1]*1000))  # 仅显示当前运动的关节扭矩信息

        # 痉挛保护未开启和主动意图识别未开启，并且扭矩信息处于正常范围，表示已经开始运动，可以开始进行痉挛保护
        if self.exerciseFlag is False and self.startClassifyFlag is False:
            if self.spasmTorqueThresholds[0] < value[1] < self.spasmTorqueThresholds[1]:
                self.exerciseFlag = True

        # 痉挛保护处理
        if self.exerciseFlag:
            # 如果判断发生痉挛，暂停运动，回到原位
            if value[1] < self.spasmTorqueThresholds[0] or value[1] > self.spasmTorqueThresholds[1]:
                self.ui.stackedWidget.setCurrentWidget(self.ui.endPage)
                self.ui.label_4.setText("检测到痉挛，设备正在缓慢回归原位。\n请等设备停稳，再做调整！")
                self.downSer.write("6\r\n".encode('utf-8'))  # 电机结束
                self.end_threads_and_reset()

        # 做主动意图判别
        if self.startClassifyFlag and self.passiveOrActiveIntentionMode == 1:
            if value[1] < self.spasmTorqueThresholds[0] or value[1] > self.spasmTorqueThresholds[1]:
                self.downSer.write("3\r\n".encode('utf-8'))  # 主动意图运动信号发送，开始运动
                self.startClassifyFlag = False

        # if self.uploadDataMode == 2:
        #     for i in range(0, 3):
        #         value[i] = 100 * value[i]
        #     # 在测量扭矩信息时，上传扭矩信息用于展示
        #     self.upSer.write(("$" + str(value[0]) + " " + str(value[1]) + " " + str(value[2]) + ";").encode())

    def closeEvent(self, event):
        # 以下串口/网络对象仅在启用硬件通信（取消 __init__ 中注释）时才存在
        if hasattr(self, 'getMotorDataWorker'):
            self.getMotorDataWorker.stop()

        if hasattr(self, 'getTorqueWorker'):
            self.getTorqueWorker.stop()

        if hasattr(self, 'clientSocket') and self.clientSocket:
            try:
                self.clientSocket.close()       # 关闭 socket 连接
            except OSError as e:
                print(f"Error closing socket: {e}")

        if hasattr(self, 'commandWorker') and self.commandWorker:
            self.commandWorker.stop()           # 停止worker的工作
            self.commandThread.quit()           # 请求线程结束
            self.commandThread.wait()           # 等待线程完全退出
            self.commandThread = None           # 重置线程和worker
            self.commandWorker = None

        self.webglWorker.stop()
        self.webglWorker.wait()

        # 停止 rPPG 独立进程和客户端线程
        if hasattr(self, 'rppg_client'):
            self.rppg_client.stop()
            self.rppg_client.wait()
        if hasattr(self, 'rppg_proc'):
            self.rppg_proc.terminate()
            try:
                self.rppg_proc.wait(timeout=3)
            except Exception:
                pass
            print("[rPPG] 独立进程已停止")

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


