#%%
import numpy as np
import matplotlib.pyplot as plt
# %%

rate = np.linspace(100, 1000)
# depth = lambda x: x, 

def depth(x):
    return x, x*(-5/9)+5000/9

plt.plot(*depth(rate))
plt.plot(1000, 0, 'go', label='no water')
plt.plot(100, 500, 'ro', label='detection threshold')

plt.xlabel('counts per second')
plt.ylabel('water depth rock')
plt.axis([1100, 0, -50, 550])

plt.legend()
