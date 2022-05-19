#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from uncertainties import ufloat
from uncertainties import unumpy

# rate = np.linspace(1000, 10000)
# depth = lambda x: x, 
# def depth(x):
#     return x, x*(-5/9)+5000/9

fig = plt.figure(figsize=(7, 5))

results = loadtxt('results-counts.txt', delimiter=';')

muon_rate = 1.0857  #1/day
depth = results[:,1]
counts = results[:,0]
STATISTICS = 1e7

counts_err = unumpy.uarray(counts, np.sqrt(counts))

detektorrate = muon_rate*counts_err/STATISTICS

plt.style.use('seaborn-whitegrid')
plt.errorbar(
    depth, unumpy.nominal_values(detektorrate), 
    yerr=unumpy.std_devs(detektorrate), fmt='vr', label='simulation data')
#%
# plt.plot(*depth(rate))
# plt.plot(1000, 0, 'go', label='no water')
# plt.plot(100, 500, 'ro', label='detection threshold')

plt.ylabel('counts per day')
plt.xlabel('water depth')
# plt.xlabel('counts per second')
# plt.ylabel('water depth rock')
# plt.xlim([900, -100])
# plt.axis([950, -100, 1.97 *1e6, 2.4 *1e6])

plt.legend()
plt.savefig('results_plot.pdf', dpi=1000)
#%%

a = counts / STATISTICS
N_0 = 1e4
N_0_err = unumpy.uarray(N_0, np.sqrt(N_0))

detektorrate_2 = muon_rate*(N_0_err*a)/N_0
print(detektorrate_2)
