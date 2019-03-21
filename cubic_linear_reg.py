# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:53:56 2019

@author: gela
"""

from Nafo import linear_regression as lr
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 40).reshape(-1,1)
y = 2*x**3 - 8*x**2 + 3
y = y.flatten()
plt.plot(x,y)
x_test = np.append(x, np.ones((40,1)), axis = 1)



Model1 = lr.Linear_Regression()
pred, cost = Model1.fit(x_test, y, epoch = 50000)
plt.plot(cost)

plt.plot(x,y)
plt.plot(x, pred)

x_test = np.append(x_test, x**2, axis = 1)
x_test = np.append(x_test, x**3, axis = 1)

Model2 = lr.Linear_Regression()
pred2, cost2 = Model2.fit(x_test, y, epoch = 500000)
plt.plot(cost2)

plt.plot(x,y)
plt.plot(x, pred2)

