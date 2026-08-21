import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

#Cfr:10-40BPM
#Cfp:40-240BPM

def caculate_cn(rgb_mean_list):
    """
    计算出三通道归一化差分信号
    """
    differences = np.diff(rgb_mean_list,axis=1)
    sums = np.add(rgb_mean_list[ : , :-1], rgb_mean_list[:, 1:])
    result = differences / sums
    return result


def caculate_weight(cfp):
    """
    使用Cfp计算权重，即Cn经过40-240BPM滤波后的脉搏信号
    """
    xs_vector = np.array([0.77, -0.51, 0])
    ys_vector = np.array([0.77,0.51,-0.77])
    xs = np.dot(xs_vector,cfp)
    ys = np.dot(ys_vector,cfp)
    xs_std = np.std(xs)
    ys_std = np.std(ys)
    a = xs_std / ys_std
    c = 1 / np.sqrt(6 * np.square(a) - 20 * a + 20)
    vector = np.array([2 - a, 2 * a - 4, a])
    return c * vector
# 设计带通滤波器
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    filtered_rgb_signal = np.zeros_like(data)
    for channel in range(data.shape[0]):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_rgb_signal[channel,:] = filtfilt(b, a, data[channel,:])
    filtered_data = filtfilt(b, a, data)
    return filtered_data



def caculate_the_respiratory_rate(rgb_mean,fs):
    
    print(rgb_mean.shape)
    #计算Cn
    Cn = caculate_cn(rgb_mean)
    
    #计算Cfp
    Cfp = apply_bandpass_filter(Cn,40 / 60,240 / 60,fs)
    
    #计算Cfr
    Cfr = apply_bandpass_filter(Cn,10 / 60,40 / 60,fs)

    #计算权重Weight
    W = caculate_weight(Cfp)
    
    #计算出呼吸信号
    r_signal = np.dot(W, Cfr)
 
    #使用峰值检测输出呼吸次数(注意时间跨度。)
    peaks, _ = find_peaks(r_signal,distance=2.5,wlen=3)
    bpm = 0
    if len(peaks) != 0:
        bpm = (peaks[0] - 1) * 2
    return int(bpm)

