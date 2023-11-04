"""
复用方法
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_cps(R):
    # 从概率矩阵提取变点位置
    last_cp = 0
    cps_detected = []
    _T = len(R[0])
    for i in range(10,_T):
        if R[i].argmax() < i-last_cp:
            
            cps_detected.append(i)
            last_cp = i
    return cps_detected

def duplicate_cps(cps):
    # 去除重复的变点
    new_cps = [cps[0]]
    for i in range(1,len(cps)):
        if cps[i]-cps[i-1]>10:
            new_cps.append(cps[i])
    return new_cps
        
        
def cp_add(to_list, from_list, offset):
    # 将from_list元素整合到to_list，并考虑偏移offset
    for _ in from_list:
        to_list.append(_ + offset)
    return to_list

def cover_rate(cps_true, cps_pred,T):
    # 计算覆盖度
    cps_t = list(cps_true)
    cps_p = list(cps_pred)
    cr = .0
    for i in range(1,len(cps_t)):
        end_true = cps_t[i]
        start_true = cps_t[i-1]
        _A = end_true - start_true
        _ = []
        for j in range(1, len(cps_p)):
            end_pred = cps_p[j]
            start_pred = cps_p[j-1]
            _cin = min(end_pred, end_true) - max(start_pred, start_true)
            _cup = max(end_pred, end_true) - min(start_pred, start_true)
            _.append(_cin / _cup)
        cr += max(_)*_A
    return cr/T

def f1_score(cps_true, cps_pred,T):
    # 计算F1
    # 计算TP
    M = 5
    tp = .0
    for cp_pred in cps_pred:
        is_match = 0
        for cp in cps_true:
            if abs(cp_pred - cp) <= M:
                is_match = 1
                break
        tp += is_match
    prec = tp / len(cps_pred)
    recall = tp / len(cps_true)
    f1 = 2*prec*recall/(prec+recall)
    return f1
        

def plot_posterior(T, data, cps, cps_pred, R, pmean, pvar):
    # 画图
    fig, axes = plt.subplots(2, 1, figsize=(18,9))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
    # Plot predictions.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(R.T, aspect='auto', cmap='gray_r', origin='lower',
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
        #ax2.axvline(cp, c='red', ls='dotted')
    for cp in cps_pred:
        ax1.axvline(cp, c='blue', ls='dotted')
        ax2.axvline(cp, c='blue', ls='dotted')
        
    plt.tight_layout()
    plt.show()   