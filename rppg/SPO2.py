import numpy as np
import scipy
from scipy.signal import butter, lfilter


def rgb_to_ycgcr(rgb_signal):
    """
    在此处将RGB信号转化为YCgCr信号
    """
    Y = np.array([])
    Cg = np.array([])
    Cr = np.array([])
    temp = np.divide(rgb_signal,255)
    for i in range(120):
        rgb = temp[:,i]
        Y = np.append(Y , 16 + (65.481 * rgb[0]) + (128.533 * rgb[1]) + 
                      (24.966 * rgb[2]))
        Cg = np.append(Cg , 128 + (-81.085 * rgb[0]) + (112 * rgb[1]) + 
                       (-30.915 * rgb[2]))
        Cr = np.append(Cr , 128 + (112 * rgb[0]) + (-93.786 * rgb[1]) + 
                       (-18.214 * rgb[2]))
    return np.array([Y , Cg , Cr])




def extract_ac_signal(ppg_signal, fs=30, lowcut=0.7, highcut=3):
  """
  提取血氧饱和度信号中的交流成分 (AC)。

  Args:
    ppg_signal: PPG信号数据，二维数组，形状为 (channels, frames)
    fs: 采样频率，默认为 30Hz
    lowcut: 低频截止频率，默认为 0.7Hz
    highcut: 高频截止频率，默认为 3Hz

  Returns:
    ac_signal: AC信号数据，二维数组，形状与 ppg_signal 相同
  """
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(N=5, Wn=[low, high], btype='band')
  ac_signal = lfilter(b, a, ppg_signal, axis=0)
  return ac_signal

def find_peaks_and_valleys(signal):
  """
  找到信号中的峰值和谷值。

  Args:
    signal: 一维信号数据

  Returns:
    peaks: 峰值索引列表
    valleys: 谷值索引列表
  """
  peaks, _ = scipy.signal.find_peaks(signal)
  valleys, _ = scipy.signal.find_peaks(-signal)
  return peaks, valleys

def calculate_peak_to_valley_ratios(signal, peaks, valleys):
  """
  计算信号中每个峰值和谷值的比值。

  Args:
    signal: 一维信号数据
    peaks: 峰值索引列表
    valleys: 谷值索引列表

  Returns:
    peak_to_valley_ratios: 谷值/峰值比值列表
  """
  peak_to_valley_ratios = np.log(abs(max(signal[valleys])) / max(signal[peaks]))
  return peak_to_valley_ratios

def smooth_peak_to_valley_ratios(ratios, window_size=10):
  """
  平滑谷值/峰值比值数据。

  Args:
    ratios: 谷值/峰值比值列表
    window_size: 平滑窗口大小

  Returns:
    smoothed_ratios: 平滑后的谷值/谷峰值比值列表
  """
  smoothed_ratios = scipy.signal.medfilt(ratios, kernel_size=window_size)
  return smoothed_ratios

def caculate_spo2(rgb_signal):
   YCgCr = rgb_to_ycgcr(rgb_signal=rgb_signal)
   #转换为AC分量
   ac_siganl = extract_ac_signal(YCgCr) 
   
   
   #提取Cg和Cr两个通道的信号
   Cg_signal = ac_siganl[1 ,:]
   Cr_signal = ac_siganl[2 ,:]

   # 找到峰值和谷值
   peaks_cg, valleys_cg = find_peaks_and_valleys(Cg_signal)
   peaks_cr, valleys_cr = find_peaks_and_valleys(Cr_signal)
   print(f"Here is the peaks and valleys value:{peaks_cg} & {valleys_cg}")

   # 计算谷值/峰值
   ratios_cg = calculate_peak_to_valley_ratios(Cg_signal, peaks_cg,
                                                valleys_cg)
   ratios_cr = calculate_peak_to_valley_ratios(Cr_signal, peaks_cr,
                                                valleys_cr)
   # 计算Rcgcr
   Rcgcr = ratios_cr / ratios_cg
   #计算SPO2
   spo2 = 11.8805 * Rcgcr + 89.1914
   
   return spo2

