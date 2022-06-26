#%%
import numpy as np
from scipy.integrate import quad, dblquad, tplquad
from uncertainties import ufloat

# def I(n):
#     return dblquad(lambda t, x: np.exp(-x*t)/t**n,
#      0, np.inf, 
#      lambda x: 1, lambda x: np.inf)


# muons_per_x = dblquad(
#     lambda E, theta: 0.14*E^(-2.7)*(
#         1    /(1+1.1*E*np.cos(theta)/(115))+
#         0.054/(1+1.1*E*np.cos(theta)/(850)))*
#         2*np.pi, 
#     600, 200*1e3,
#     lambda theta: np.cos(np.radians(0)), lambda theta: np.cos(np.radians(30))
#     )
emin = 200
emin = 600
emax = np.inf
emax = 200*1e4
thetamin=0
thetamax=30

# emin = 1
# emax = 200*1e3
# thetamin=0
# thetamax=80
scale_factor = 1e0
# W = cos(theta)
# muons_per_x = dblquad(
#     lambda W, E: 0.14*E**(-2.7)*
#             (
#                 1    /(1+1.1*E*W/(115))+
#                 0.054/(1+1.1*E*W/(850))
#             )*
#         2*np.pi*
#         (-1), 
#     np.cos(np.radians(thetamin)), np.cos(np.radians(thetamax)),
#     lambda E: emin, lambda E: emax
#     )

f = lambda theta, phi, E: 0.14*E**(-2.7)*(
                    1    /(1+1.1*E*np.cos(theta)/(115))+
                    0.054/(1+1.1*E*np.cos(theta)/(850))
                )*np.sin(theta)*scale_factor 

muons_per_x = tplquad(f , 
    emin, emax,
    lambda E: 0, lambda E: 2*np.pi,
    lambda E, phi: np.radians(thetamin), lambda E, phi: np.radians(thetamax)
    )
Detektor_area = 10000
Detektor_area = 75
result = ufloat(muons_per_x[0]*60*60*24*Detektor_area, muons_per_x[1])

# print(f'{muons_per_x[0]/scale_factor} +- {muons_per_x[1]/scale_factor} / 1/cm^2*min')
print(f'{result/scale_factor} / 1/Tag')