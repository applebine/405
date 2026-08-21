import numpy as np
import time
import logging
from scipy.stats import kurtosis
from rppg.respiratory import *
from rppg.HRV import *
from rppg.SPO2 import *
import threading


class POS:
    def __init__(self):
        # print("POS __init__ called")

        self.bpm_history = []  # 心率历史记录
        self.max_history = 5   # 保存最近5次心率值
        self.red_trace = []
        self.green_trace = []
        self.blue_trace = []
        self.time_stamp = []
        self.lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._stop_event = threading.Event()
        self._callback = None
        self._frame_counter = 0
        self._pending = False
        self._worker_thread.start()
        self._signal_quality = 1.0
        self.state_bpm = False
        self.bpm = 0
        self.oldbpm = 0

    def set_callback(self, callback):
        """设置结果回调函数，参数为(rppg_signal, bpm, fps, ...)"""
        self._callback = callback

    def stop(self):
        self._stop_event.set()
        self._worker_thread.join()

    def add_rgb(self, rgb):
        #在此增加关于rgb亮度的归一化处理
        with self.lock:
            brightness = np.sum(rgb)
            if brightness > 1e-6:
                normalized_rgb = rgb / brightness
            else:
                normalized_rgb = rgb
            self.red_trace.append(normalized_rgb[0])
            self.green_trace.append(normalized_rgb[1])
            self.blue_trace.append(normalized_rgb[2])
            self.time_stamp.append(time.time())
            if len(self.red_trace) > 120:
                self.red_trace.pop(0)
                self.green_trace.pop(0)
                self.blue_trace.pop(0)
                self.time_stamp.pop(0)
            self._frame_counter += 1
            # 满120帧后，每新来30帧触发一次
            if len(self.red_trace) == 120 and self._frame_counter >= 30:
                self._pending = True
                self._frame_counter = 0

    def _evaluate_signal_quality(self, signal):
        """评估信号质量(0-1)"""
        # 峰度评估(理想脉搏信号峰度在3左右)
        kur = kurtosis(signal)
        kur_score = 1 - min(abs(kur - 3) / 3, 1)

        # 频谱分析
        fft = np.abs(np.fft.rfft(signal - np.mean(signal))[1:])
        if len(fft) > 0:
            snr = np.max(fft) / (np.mean(fft) + 1e-6)
            snr_score = min(snr / 10, 1)
        else:
            snr_score = 0

        # 综合评分
        return 0.6 * kur_score + 0.4 * snr_score

    def _worker(self):
    
        while not self._stop_event.is_set():
            if self._pending:
                with self.lock:
                    trace = np.array([
                        self.red_trace,
                        self.green_trace,
                        self.blue_trace
                    ])
                    time_stamps = self.time_stamp.copy()
                    self._pending = False
                # 调用你的POS算法，得到rPPG信号、bpm等
                rppg_signal, rppg_filtered, bpm, fps = self._calc_rppg(trace, time_stamps)
                # 各种平滑策略
                # self._smooth_heart_rate_with_signal_quality(bpm)
                # self._smooth_heart_rate_without_signal_quality(bpm)
                # self._smooth_heart_rate_dual_filter(bpm)
                self._smooth_heart_rate_with_history_without_signal_quality(bpm)
                # self._smooth_heart_rate_with_history_and_signal_quality(bpm)


                # print("Smoothed heart rate: bpm =", self.bpm, "fps =", fps)#此处打印为平滑后的心率

                # 这里调用呼吸率并回调回主线程
                respiratory_rate = caculate_the_respiratory_rate(trace, fps)

                # 这里调用血氧并回调回主线程
                spo2 = int(caculate_spo2(trace))
                if spo2 >= 99:
                    spo2 = 99  # 避免满值干扰观察

                # 这里调用心率变异性
                hrv = caculate_HRV(rppg_signal[-30:], fps, time_stamps[-30:])
                # 注意：
                # bpm:原始的未平滑的心率值
                # self.bpm:平滑后的心率值
                if self._callback:
                    self._callback(rppg_signal, rppg_filtered, self.bpm, respiratory_rate, spo2, hrv, fps)
            else:
                time.sleep(0.01)  # 避免空转

    def _calc_rppg(self, trace, time_stamps):
        # POS算法，返回rPPG信号、bpm、fps等
        print("Calculating rPPG...")
        fps = 120 / (time_stamps[-1] - time_stamps[0]) if len(time_stamps) == 120 else 0
        
        rppg_signal = self._get_pos_signal(trace)
        rppg_filtered = self.apply_filter(rppg_signal, fps, hr_min=90, hr_max=180)
        
        self._signal_quality = self._evaluate_signal_quality(rppg_signal)
        bpm = self._estimate_bpm(rppg_filtered, fps)
    
        return rppg_signal,rppg_filtered, bpm, fps

    def _get_pos_signal(self, trace):
        # POS算法实现
        miu = trace.mean(1).reshape(3, 1)
        ntrace = trace / (miu + 1e-6)
        S = np.dot(np.array([[0, 1, -1], [-2, 1, 1]]), ntrace)
        return S[0, :] + S[1, :] * (np.std(S[0, :]) / (np.std(S[1, :]) + 1e-6))
    
    def apply_filter(self, signal, fps, hr_min, hr_max):
        
        # 检查信号长度
        if len(signal) < 50:
            print(f"警告: 信号太短 ({len(signal)})，跳过滤波")
            return signal
        
        try:
            nyquist = fps / 2
            low_freq = hr_min / 60 / nyquist
            high_freq = hr_max / 60 / nyquist
            
            low_freq = max(0.01, min(low_freq, 0.99))
            high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
            
            # 使用较低的滤波器阶数
            b, a = butter(2, [low_freq, high_freq], btype='band')
            
            # 限制padlen
            padlen = min(len(signal) // 3, 15)
            filtered = filtfilt(b, a, signal, padlen=padlen)
            
            return filtered
        
        except Exception as e:
            print(f"滤波失败: {e}，返回原始信号")
            return signal

    def _estimate_bpm(self, sig, fps):
        """
        改进的心率估计算法
        主要改进：
        1. 扩展心率范围到0.7-3.0 Hz (42-180 BPM)
        2. 增加信噪比计算
        3. 峰值验证
        4. 结合信号质量评估
        """
        if fps <= 0:
            return None
        
        # 使用你的FFT方法
        N = 60 * int(fps)
        m2 = int(2 ** np.ceil(np.log2(N)))
        
        # 补零到m2长度
        sig = np.asarray(sig)
        if sig.ndim > 1:
            sig = sig.flatten()
        if len(sig) < m2:
            sig = np.pad(sig, (0, m2 - len(sig)), 'constant')
        else:
            sig = sig[:m2]
        
        # FFT
        amp = np.fft.fft(sig, n=m2)
        pows = np.abs(amp) ** 2
        freqs = fps * np.arange(0, m2 // 2 + 1) / m2
        
        # 只取正频率部分
        pows = pows[:m2 // 2 + 1]
        freqs = freqs[:m2 // 2 + 1]
        
        # 【修改1】扩展心率范围到1.5-3.0 Hz (42-180 BPM)
        # 原来的1.6-3.0 Hz太窄了，会错过很多正常心率
        valid_mask = (freqs >= 1.5) & (freqs <= 3.0)
        valid_freqs = freqs[valid_mask]
        valid_pows = pows[valid_mask]
        
        if len(valid_freqs) == 0:
            return None
        
        # 【修改2】找到主频
        peak_idx = np.argmax(valid_pows)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_pows[peak_idx]
        
        # 【修改3】计算信噪比
        # 排除峰值±0.1Hz范围内的频率作为噪声
        noise_mask = np.abs(valid_freqs - peak_freq) > 0.1
        if np.any(noise_mask):
            noise_power = np.mean(valid_pows[noise_mask])
        else:
            noise_power = np.mean(valid_pows)
        
        # 安全计算信噪比
        if noise_power > 0 and peak_power > 0:
            snr_linear = peak_power / noise_power
            snr_db = 10 * np.log10(min(max(snr_linear, 1e-10), 1e10))
        else:
            snr_db = -999  # 无效信噪比
        
        # 【修改4】峰值验证
        # 检查峰值是否显著高于周围频率
        window_size = 3  # 检查周围3个频率点
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(valid_pows), peak_idx + window_size + 1)
        
        local_pows = valid_pows[start_idx:end_idx]
        if len(local_pows) > 1:
            local_mean = np.mean(local_pows[local_pows != peak_power])
            peak_prominence = peak_power / (local_mean + 1e-10)
        else:
            peak_prominence = 1.0
        
        # 【修改5】综合评估峰值质量
        # 结合信噪比、峰值显著性和信号质量
        quality_score = 0
        
        # 信噪比评分 (0-40分)
        if snr_db > 20:
            quality_score += 40
        elif snr_db > 10:
            quality_score += 30
        elif snr_db > 0:
            quality_score += 20
        elif snr_db > -10:
            quality_score += 10
        
        # 峰值显著性评分 (0-30分)
        if peak_prominence > 5:
            quality_score += 30
        elif peak_prominence > 3:
            quality_score += 20
        elif peak_prominence > 2:
            quality_score += 10
        
        # 信号质量评分 (0-30分)
        quality_score += self._signal_quality * 30
    
        
        # 【修改6】根据质量分数决定是否接受这个心率
        # if quality_score < 30:  # 质量太低，使用上次值
        #     return self.oldbpm if self.state_bpm else None
        
        # 【修改7】计算最终心率
        HR = int(np.ceil(peak_freq * 60))
        
        # # 【修改8】生理合理性检查
        # if HR < 40 or HR > 220:
        #     return self.oldbpm if self.state_bpm else 72
        
        # # 【修改9】根据质量调整置信度
        # # 质量越高，越相信新的测量值
        # confidence = min(quality_score / 100, 1.0)
        
        # # # 如果质量不是很高，与历史值进行加权平均
        # if confidence < 0.7 and self.state_bpm:
        #     HR = int(confidence * HR + (1 - confidence) * self.oldbpm)
        
        return HR
    def _smooth_heart_rate_with_history_and_signal_quality(self, new_bpm):
        """使用历史数据的平滑算法"""
        try:
            if not self.state_bpm:  # 第一次初始化
                self.bpm = new_bpm
                self.oldbpm = new_bpm
                self.state_bpm = True
                self.bpm_history = [new_bpm]
                return

            # 1. 生理范围检查
            if new_bpm < 40 or new_bpm > 220:

                return

            # 2. 异常值检测
            if len(self.bpm_history) >= 3:
                recent_avg = sum(self.bpm_history[-3:]) / 3
                if abs(new_bpm - recent_avg) > 15:
                    new_bpm = round(recent_avg)

            # 3. 添加到历史记录
            self.bpm_history.append(new_bpm)
            if len(self.bpm_history) > self.max_history:
                self.bpm_history.pop(0)

            # 4. 计算加权平均（最新的权重更大）
            weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # 最新的权重0.3
            if len(self.bpm_history) < 5:
                weights = weights[-len(self.bpm_history):]
            
            weighted_avg = sum(b * w for b, w in zip(self.bpm_history, weights))
            
            # 5. 信号质量调整
            if self._signal_quality > 0.7:
                # 高质量信号，允许更快变化
                self.bpm = round(0.7 * new_bpm + 0.3 * weighted_avg)
            else:
                # 低质量信号，更保守
                self.bpm = round(0.3 * new_bpm + 0.7 * weighted_avg)
            
            self.oldbpm = self.bpm

        except Exception as e:
            print(f"心率平滑异常: {e}")
            self.bpm = self.oldbpm if self.state_bpm else 72

    def _smooth_heart_rate_with_history_without_signal_quality(self, new_bpm):
        """使用历史数据的平滑算法（无信号质量依赖）"""
        try:
            if not self.state_bpm:  # 第一次初始化
                self.bpm = new_bpm
                self.oldbpm = new_bpm
                self.state_bpm = True
                self.bpm_history = [new_bpm]
                return

            # 1. 生理范围检查
            if new_bpm < 40 or new_bpm > 220:
                print(f"心率超出生理范围: {new_bpm}，忽略")
                return

            # 2. 异常值检测
            if len(self.bpm_history) >= 3:
                recent_avg = sum(self.bpm_history[-3:]) / 3
                if abs(new_bpm - recent_avg) > 18:
                    print(f"检测到异常心率: {new_bpm}，使用历史平均: {recent_avg:.1f}")
                    new_bpm = round(recent_avg)

            # 3. 添加到历史记录
            self.bpm_history.append(new_bpm)
            if len(self.bpm_history) > self.max_history:
                self.bpm_history.pop(0)

            # 4. 计算加权平均（最新的权重更大）
            weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # 最新的权重0.3
            if len(self.bpm_history) < 5:
                weights = weights[-len(self.bpm_history):]
            
            weighted_avg = sum(b * w for b, w in zip(self.bpm_history, weights))
            
            # 5. 固定权重混合
            self.bpm = round(0.6 * new_bpm + 0.4 * weighted_avg)
            self.oldbpm = self.bpm
            
            print(f"心率平滑(历史): {new_bpm} -> {self.bpm}")

        except Exception as e:
            print(f"心率平滑异常: {e}")
            self.bpm = self.oldbpm if self.state_bpm else 72
    def _smooth_heart_rate_with_signal_quality(self, new_bpm):
        """改进的心率平滑策略"""
        try:
            if not self.state_bpm:  # 第一次初始化
                self.bpm = new_bpm
                self.oldbpm = new_bpm
                self.state_bpm = True
                return

            # 1. 生理范围检查
            if new_bpm < 40 or new_bpm > 220:
                print(f"心率超出生理范围: {new_bpm}，忽略")
                return

            # 2. 计算变化量
            bpm_diff = new_bpm - self.oldbpm
            abs_diff = abs(bpm_diff)
            
            # 3. 异常值检测 - 变化超过20bpm认为是异常
            if abs_diff > 20:
                print(f"检测到异常心率变化: {self.oldbpm} -> {new_bpm}，使用上次值")
                return
            
            # 4. 动态阈值计算
            base_threshold = 8  # 基础阈值±8bpm
            dynamic_threshold = base_threshold + 7 * (1 - self._signal_quality)  # 根据信号质量浮动

            # 5. 分级处理策略
            if abs_diff <= base_threshold:
                # 小幅变化，直接接受
                self.bpm = new_bpm
            elif abs_diff <= dynamic_threshold:
                # 中等变化，加权平均
                weight = 0.7 - 0.3 * (abs_diff - base_threshold) / (dynamic_threshold - base_threshold)
                self.bpm = weight * new_bpm + (1 - weight) * self.oldbpm
            else:
                # 大幅变化，保守处理
                weight = max(0.3, 0.4 - 0.1 * (abs_diff - dynamic_threshold) / 10)
                self.bpm = weight * new_bpm + (1 - weight) * self.oldbpm

            # 6. 最终平滑 - 指数移动平均
            alpha = 0.3 + 0.4 * self._signal_quality  # 信号质量越高，变化越快
            self.bpm = alpha * self.bpm + (1 - alpha) * self.oldbpm
            
            # 7. 四舍五入到整数
            self.bpm = round(self.bpm)
            self.oldbpm = self.bpm

            print(f"心率平滑: {new_bpm} -> {self.bpm} (信号质量: {self._signal_quality:.2f})")

        except Exception as e:
            print(f"心率平滑异常: {e}")
            # 保持上次值不变
            self.bpm = self.oldbpm if self.state_bpm else 72

    def _smooth_heart_rate_without_signal_quality(self, new_bpm):
        """改进的心率平滑策略（无信号质量依赖）"""
        try:
            if not self.state_bpm:  # 第一次初始化
                self.bpm = new_bpm
                self.oldbpm = new_bpm
                self.state_bpm = True
                return

            # 1. 生理范围检查
            if new_bpm < 40 or new_bpm > 220:
                print(f"心率超出生理范围: {new_bpm}，忽略")
                return

            # 2. 计算变化量
            bpm_diff = new_bpm - self.oldbpm
            abs_diff = abs(bpm_diff)
            
            # 3. 异常值检测 - 变化超过25bpm认为是异常
            if abs_diff > 25:
                print(f"检测到异常心率变化: {self.oldbpm} -> {new_bpm}，使用上次值")
                return
            
            # 4. 固定阈值分级处理
            small_threshold = 5      # 小变化阈值±5bpm
            medium_threshold = 12    # 中等变化阈值±12bpm
            
            if abs_diff <= small_threshold:
                # 小幅变化，直接接受
                self.bpm = new_bpm
            elif abs_diff <= medium_threshold:
                # 中等变化，加权平均
                weight = 0.7 - 0.2 * (abs_diff - small_threshold) / (medium_threshold - small_threshold)
                self.bpm = weight * new_bpm + (1 - weight) * self.oldbpm
            else:
                # 大幅变化，保守处理
                weight = max(0.2, 0.5 - 0.3 * (abs_diff - medium_threshold) / 13)
                self.bpm = weight * new_bpm + (1 - weight) * self.oldbpm

            # 5. 最终平滑 - 固定指数移动平均
            alpha = 0.4  # 固定平滑因子
            self.bpm = alpha * self.bpm + (1 - alpha) * self.oldbpm
            
            # 6. 四舍五入到整数
            self.bpm = round(self.bpm)
            self.oldbpm = self.bpm

            print(f"心率平滑: {new_bpm} -> {self.bpm}")

        except Exception as e:
            print(f"心率平滑异常: {e}")
            # 保持上次值不变
            self.bpm = self.oldbpm if self.state_bpm else 72

    def _smooth_heart_rate_dual_filter(self, new_bpm):
        """双重滤波平滑算法（无信号质量依赖）"""
        try:
            if not self.state_bpm:  # 第一次初始化
                self.bpm = new_bpm
                self.oldbpm = new_bpm
                self.state_bpm = True
                self.bpm_history = [new_bpm]
                return

            # 1. 生理范围检查
            if new_bpm < 40 or new_bpm > 220:
                print(f"心率超出生理范围: {new_bpm}，忽略")
                return

            # 2. 第一层滤波：异常值检测和修正
            if len(self.bpm_history) >= 3:
                recent_avg = sum(self.bpm_history[-3:]) / 3
                if abs(new_bpm - recent_avg) > 20:
                    print(f"检测到异常心率: {new_bpm}，修正为: {recent_avg:.1f}")
                    new_bpm = round(recent_avg)

            # 3. 第二层滤波：变化量限制
            bpm_diff = new_bpm - self.oldbpm
            abs_diff = abs(bpm_diff)
            
            # 限制单次最大变化
            max_change = 15  # 单次最大变化15bpm
            if abs_diff > max_change:
                if bpm_diff > 0:
                    new_bpm = self.oldbpm + max_change
                else:
                    new_bpm = self.oldbpm - max_change
                print(f"限制心率变化: {self.oldbpm} -> {new_bpm}")

            # 4. 第三层滤波：滑动平均
            self.bpm_history.append(new_bpm)
            if len(self.bpm_history) > self.max_history:
                self.bpm_history.pop(0)

            # 计算加权平均
            weights = [0.1, 0.15, 0.2, 0.25, 0.3]
            if len(self.bpm_history) < 5:
                weights = weights[-len(self.bpm_history):]
            
            weighted_avg = sum(b * w for b, w in zip(self.bpm_history, weights))
            
            # 5. 最终输出：混合当前值和历史平均
            self.bpm = round(0.7 * new_bpm + 0.3 * weighted_avg)
            self.oldbpm = self.bpm
            
            print(f"心率平滑(双重): 原始{new_bpm} -> 平滑{self.bpm}")

        except Exception as e:
            print(f"心率平滑异常: {e}")
            self.bpm = self.oldbpm if self.state_bpm else 72
    