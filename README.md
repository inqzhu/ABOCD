# ABOCD
Adaptive Bayesian Online Changepoint Detection for Financial Time Series

ABOCD is a Python-based package for online changepoint detection. Based on the framework of Bayesian online changepoint detection, we develop an adaptive online changepoint detection method with sliding window. Through the smoothing mechanism of the sliding window, our method updates the hyper parameters required for changepoint detection in each window. As a result, the hyper parameters are adaptive to the dynamic changes of the data environment. 

An example of how to use this package is illustrated in `example.ipynb`. 
Descriptions of all files are given below:

| File name | Description | 
| ----------- | ----------- | 
| example.ipynb | An example of using ABOCD |
| abocd.py | The ABOCD detector |
| sim.py | Used for generate simulated dataset | 
| func.py | Auxiliary functions |


