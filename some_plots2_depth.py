#%%
import numpy as np
import matplotlib.pyplot as plt
# %%

rate = np.linspace(0, 1400)
# depth = lambda x: x, 

def dichte(x, y):
    return x, y

def p(a,b, rho):
    return dichte(np.linspace(a, b), np.linspace(rho, rho))

fig = plt.figure(figsize=(6, 6))

plt.plot(*p(0, 411.2, 2.55), label='Sandstein')
plt.plot(*p(411.2, 415, 2.32), label='Tonstein')
plt.plot(*p(415, 466.2, 2.55), label='Sandstein')
plt.plot(*p(466.2, 477.7, 2.32), label='Tonstein')
plt.plot(*p(477.7, 481.5, 2.85), label='Dolomitstein')
plt.plot(*p(481.5, 485.2, 2.71), label='Mergelstein')
plt.plot(*p(485.2, 509.5, 4.99), label='Anhydrit')
plt.plot(*p(509.5, 519.2, 2.71), label='Mergelstein')
plt.plot(*p(519.2, 1204, 2.32), label='Tonstein')
# plt.plot(1000, 0, 'go', label='no water')
# plt.plot(100, 500, 'ro', label='detection threshold')

plt.xlabel(r'undergound depth / $\mathrm{m}$')
plt.ylabel(r'density / $g / cm^3$')
# plt.xlabel(fr'propagated distance $\,/\, \mathrm{{{xlabel_unit}}} $')
# plt.axis([0, 1204, 0, ])
plt.xlim(390,540)
plt.ylim(0,5.7)


plt.legend(loc='upper left')
# plt.savefig('geology_modell_full.png', dpi=1000)
plt.savefig('geology_modell_detail.png', dpi=1000)
