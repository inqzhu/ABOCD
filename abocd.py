'''
ABOCD方法
'''
import numpy as np
import copy
from scipy.stats import norm
from scipy.special import logsumexp

class ABOCDetector(object):
    
    def __init__(self, window_learn):
        # 初始化设置
        # 初始化 hazard
        self.hazard = 1/100
        # 初始化时间游标
        self.t = 1
        # 学习率
        self.lr = 0.001
        # 默认梯度下降轮数
        self.epochs = 10
        # 采用快速估计超参数，还是采用梯度下降估计
        self.op = 'fast'
        
        # 初始化一个概率模型，用于计算预测概率 P(x|...)
        [new_mean, new_var0, new_varx] = self.update_par(window_learn)
        self.model = GaussianUnknownMean(new_mean, new_var0, new_varx)

        # 预测的均值
        self.pmean = []
        # 预测的方差
        self.pvar = []

        # 初始化passing message
        self.log_message = np.array([0])  

        # 初始化概率矩阵(对数化后的)
        self.log_R = {}
        self.log_R[0] = np.array([[0]])
        


    def update(self, x, new_hazard=None):
        # 基于输入的新数据更新信息
        
        # 若有外部数据（如已实现波动率）输入，则根据输入数据确定hazard
        # 否则采用默认hazard
        if new_hazard:
            log_H = np.log(new_hazard)
            log_1mH = np.log(new_hazard)
        else:
            log_H = np.log(self.hazard)
            log_1mH = np.log(1 - self.hazard)

        # 计算当前的后验均值、方差
        self.pmean.append( np.sum(np.exp(self.log_R[self.t-1][ :self.t]) * \
            self.model.mean_params[ :self.t]) )
        self.pvar.append( np.sum(np.exp(self.log_R[self.t-1][ :self.t]) * \
            self.model.var_params[ :self.t]) )
        
        # 输出当前关于x的条件概率
        log_pis = self.model.log_pred_prob(self.t, x)

        # 计算增长概率（r_t = r_(t-1) + 1）
        log_growth_probs = log_pis + self.log_message + log_1mH

        # 计算变点概率(r_t = 0)
        log_cp_prob = logsumexp(log_pis + self.log_message + log_H)

        # 对概率归一化
        new_log_joint = np.append(log_cp_prob, log_growth_probs)
        self.log_R[self.t]  = new_log_joint
        self.log_R[self.t] -= logsumexp(new_log_joint)

        # 更新参数
        self.model.update_params(self.t, x)

        # 保留信息用以下一轮迭代
        self.log_message = new_log_joint

        self.t += 1

    def update_par(self, window):
        # 更新超参数
        # 初始估计
        xs = np.array(window)
        new_mean = np.median(xs)
        new_varx = new_var0 = xs.std()
        _est = np.array([new_mean, new_var0, new_varx])
        if self.op == 'SGD':
            # 梯度下降，优化学习超参数设置；否则采用快速初始估计
            for epoch in range(self.epochs):
                grads = self.abocd_grads(_est, window)
                _est = _est - self.lr * grads
            [new_mean, new_var0, new_varx] = _est
        return [new_mean, new_var0, new_varx]
    
    def abocd_loss(self, data, model, hazard):
        # 给定一段数据，计算loss
        T = len(data)
        log_R = -np.inf * np.ones((T+1, T+1))
        log_R[0, 0] = 0      
        pmean = np.empty(T)   
        pvar = np.empty(T)   
        log_message = np.array([0])   
        log_H = np.log(hazard)
        log_1mH = np.log(1 - hazard)
    
        ll = []
        for t in range(1, T+1):
            x = data[t-1]
            pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
            pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * model.var_params[:t])
            log_pis = model.log_pred_prob(t, x)
            log_growth_probs = log_pis + log_message + log_1mH
            log_cp_prob = logsumexp(log_pis + log_message + log_H)
            new_log_joint = np.append(log_cp_prob, log_growth_probs)
            ll.append(new_log_joint)
            log_R[t, :t+1]  = new_log_joint
            log_R[t, :t+1] -= logsumexp(new_log_joint)
            model.update_params(t, x)
            log_message = new_log_joint
        return ll[-1][-1]
    
    def abocd_grads(self, _est, data, delta=10**(-7)):
        # 计算梯度
        [mean0, var0, varx] = _est
        test_model = GaussianUnknownMean(mean0, var0, varx)
        basic_loss = self.abocd_loss(data, test_model, self.hazard)
        grads = copy.deepcopy(_est)
        
        for p in range(len(_est)):
            test_est = copy.deepcopy(_est)
            test_est[p] += delta
            [mean0, var0, varx] = test_est
            test_model = GaussianUnknownMean(mean0, var0, varx)
            test_loss = self.abocd_loss(data, test_model, self.hazard)
            grads[p] = (test_loss-basic_loss)/delta
        return grads
        
        
    def report(self):
        # 输出 rt 概率
        T = self.t
        log_R = -np.inf * np.ones((T+1, T+1))
        log_R[0][0] = 0
        
        for t in range(1,T):
            log_R[t-1][:t] = self.log_R[t-1]
        R = np.exp(log_R)
        return R


class GaussianUnknownMean:
    # 对数据分布设置的高斯模型
    # 结合先验、后验，实际为T分布模型
    
    def __init__(self, mean0, var0, varx):
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def update(self, new_mean, new_varx):
        self.mean0 = new_mean
        self.var0 = new_varx
        self.varx = new_varx #### ****
        
    def log_pred_prob(self, t, x):
        # 输出给定参数时，x的条件概率
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, x):
        # 更新参数
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params 
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        # 利用精度更新计算方差
        return 1./self.prec_params + self.varx

