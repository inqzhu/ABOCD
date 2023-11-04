import numpy as np
import random
import math 

def generate_data(case_name, T, cps):
	# 产生模拟数据
	# T-时序长度
	# cps-变点序列

	# case 1
	# 正态分布仅发生均值变化
	if case_name == 'normal-mean-change':
		X = np.zeros(T)
		cps = [0] + list(cps)
		_mean = 0
		for i in range(1,len(cps)):
			_n = cps[i]-cps[i-1]
			_X = np.random.randn(_n) + _mean
			X[cps[i-1]: cps[i]] = _X
			_mean += 2
	# case 2
	# 正态分布，仅方差变化
	elif case_name == 'normal-var-change':
		X = np.zeros(T)
		cps = [0] + list(cps)
		_sigma = 1
		for i in range(1,len(cps)):
			_n = cps[i]-cps[i-1]
			_X = np.random.randn(_n) * _sigma
			X[cps[i-1]: cps[i]] = _X
			_sigma *= 2
	# case 3
	elif case_name == 'poisson':
		X = np.zeros(T)
		cps = [0] + list(cps)
		_lambda = 1
		for i in range(1,len(cps)):
			_n = cps[i]-cps[i-1]
			_X = np.random.poisson(_lambda,size=_n)
			X[cps[i-1]: cps[i]] = _X
			_lambda += 1
	# case 4
	elif case_name == 'poisson2':
		X = []
		cps = [0]
		_lambda = 1
		pr = math.exp(-_lambda)
		for t in range(T):
			_X = np.random.poisson(_lambda,size=1)[0]
			X.append(_X)

			_r = random.random()
			if _r < pr and t-cps[-1]>30:
				_lambda += 1
				pr = math.exp(-_lambda)
				cps.append(t)
		X = np.array(X)

	return X, cps

		
