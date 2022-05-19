#%%
import numpy as np
import matplotlib.pyplot as plt


rate = np.linspace(0, 1400)
# depth = lambda x: x, 

def dichte(x, y):
    return x, y

# def p(a,b, rho):
#     return dichte(np.linspace(a, b), np.linspace(rho, rho))

def p(a,b, rho):
    return (np.linspace(a, b), np.linspace(rho, rho))



from numpy import loadtxt

#read text file into NumPy array
z = loadtxt('data/z.txt')
z = list(z)
dichte = loadtxt('data/dichte.txt')
dichte = list(dichte)
gesteinstyp = loadtxt('data/gesteinstyp.txt', dtype='str')
gesteinstyp = list(gesteinstyp)

#%%
fig = plt.figure(figsize=(8, 6))

for i in range(13):
    # plt.plot(*p(z[i], z[i+1], dichte[i]), label=gesteinstyp[i])
    plt.plot(np.linspace(z[i], z[i+1]), np.linspace(dichte[i], dichte[i]), label=gesteinstyp[i])
    # plt.plot(*p(schichten[i]), label='Sandstein')

# plt.plot(1000, 0, 'go', label='no water')
# plt.plot(100, 500, 'ro', label='detection threshold')

plt.xlabel(r'Tiefe / $\mathrm{m}$')
plt.ylabel(r'Dichte / $g / cm^3$')

# plt.xlabel(fr'propagated distance $\,/\, \mathrm{{{xlabel_unit}}} $')
# plt.axis([0, 1204, 0, ])

# plt.xlim(390,540)
# plt.ylim(0,5.7)


plt.legend(loc='upper right')
# plt.legend(bbox_to_anchor=(1.0, 1.05))
plt.savefig('geology_modell_full.pdf', dpi=1000)

# plt.savefig('geology_modell_detail.pdf', dpi=1000)
