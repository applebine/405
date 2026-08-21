import numpy as np
from scipy.signal import welch , butter, filtfilt, find_peaks

def filt_the_heart_rate_signal(hr_signal,lowcut,highcut,fs,order = 2):
  nyquist = 0.5 * fs
  low = lowcut / nyquist
  high = highcut / nyquist
  b, a = butter(order, [low ,high], btype="band")
  return filtfilt(b, a, hr_signal)


def caculate_SDNN_and_RMSSD(timestamp):
   """
   此处在时域上进行分析计算
   """
   #计算相邻差值
   differences = np.diff(timestamp)
   rr = np.sum(differences) / len(differences)
   sum = 0
   for rri in differences:
     sum = sum + np.square(rri - rr)
   sdnn = np.sqrt(sum / len(differences))

   rmssd_diff = np.diff(differences)
   rmssd = np.sqrt(np.sum(rmssd_diff **2) / len(rmssd_diff))

   return sdnn,rmssd
     
def caculate_LF_HF_LFHF(frequencies , power_spectrum):
  lf_band = (frequencies >= 0.04) & (frequencies <= 0.15)
  lf_power = np.sum(power_spectrum[lf_band]) / np.sum(power_spectrum)


  hf_band = (frequencies >= 0.15) & (frequencies <= 0.4)
  hf_power = np.sum(power_spectrum[hf_band]) / np.sum(power_spectrum)
  


  lf_hf_ratio = lf_power / hf_power
  return [lf_power, hf_power, lf_hf_ratio]
  
def caculate_HRV(hr_signal,fs,timestamp):
    peaks,_ = find_peaks(hr_signal)
    #print(f"Here is the {peaks} and {timestamp}")
    peak_timestamp = np.array([])
    for i in peaks:
        peak_timestamp = np.append(peak_timestamp,timestamp[i])
    ##print(f"Here is the peak_stamp {peak_timestamp}")
#计算出时域参数
    sdnn,rmssd = caculate_SDNN_and_RMSSD(timestamp=peak_timestamp)
    #print(f"The SDNN is {sdnn} \nThe RMSSD is {rmssd}")

#将滤波后的信号转化为功率谱信号
    frequencies , power_spectrum = welch(hr_signal,fs=fs,nperseg=100*fs,window="hann",nfft=100*fs)
    #将功率谱归一化
    
    ##print(f"Frequencies are {frequencies}")

#计算出频域的参数
    spectrum = caculate_LF_HF_LFHF(frequencies=frequencies, power_spectrum=power_spectrum)
    #print(f"{spectrum[0]}\n{spectrum[1]}\n{spectrum[2]}")
    return [sdnn,rmssd,spectrum]