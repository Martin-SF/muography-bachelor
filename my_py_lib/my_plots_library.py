import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# import inspect
# function_name = inspect.stack()[0][3]

# def pp_get_pos(pos_obj):
#     # a = np.empty(3, dtype=float)
#     a = [float]*3
#     a[0] = pos_obj.x
#     a[1] = pos_obj.y
#     a[2] = pos_obj.z
#     return a
def pp_get_pos(pos_obj):
    return [pos_obj.x, pos_obj.y, pos_obj.z]

def plot_energy_std(energy_array, binsize = 20, 
                        name = 'plot_energy_std', xlabel_unit = 'MeV', show=True, **kwargs):
    # expecting energy in MeV
    plt.xscale('log')
    plt.xlabel(fr'$E \,/\, \mathrm{{{xlabel_unit}}}$')
    plt.ylabel("# of particles")
    plt.title(name)
    bins = np.geomspace(min(energy_array), max(energy_array), binsize)
    plt.hist(energy_array, bins=bins, log=True, **kwargs)
    plt.savefig(f'{name}.pdf', bbox_inches="tight")
    if (show):
        plt.show()

def plot_distances_std(distances_array, binsize = 20, 
                        name = 'plot_distances_std', xlabel_unit = 'cm', show=True, **kwargs):
    # expecting distances in cm
    # plt.xscale('log')
    plt.xlabel(fr'propagated distance $\,/\, \mathrm{{{xlabel_unit}}} $')
    plt.ylabel("# of particles")
    plt.title(name)
    bins = np.linspace(min(distances_array), max(distances_array), binsize)
    _ = plt.hist(distances_array, bins=bins, log=True, **kwargs)
    plt.savefig(f'{name}.pdf', bbox_inches="tight")
    if (show):
        plt.show()

def cuboid_data(o, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)
    
def plotCubeAt(pos=(0,0,0), size=(1,1,1), ax=None, **kwargs):
    # from https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
    # Plotting a cube element at position pos
    pos = list(pos)
    # setting the position to the center of the cube
    for i in range(3):
        pos[i] -= size[i]/2
    if (ax !=None):
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, zorder=10000000000, **kwargs)

def plot_3D_start_end(dataset, detector_pos, detector_size, elev=30.0, azim=30, 
                        alpha=0.1, name = 'plot_3D_start_end', title = 'plot_3D_start_end', dpi=400, show=True, **kwargs):
    # original function by dominik baar
    # fig = plt.figure(figsize=(16, 16))
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.view_init(elev=elev, azim=azim)
    ls = dataset[:,0:3].reshape((-1,2,3))
    # name = '\n\n\n\n\n\n\n\n\n'+name
    ax.set_title(title, pad = -200)
    # plotCubeAt(pos=detector_pos, size=detector_size, ax=ax, color='r', alpha=1)
    ax.plot(dataset[:,0], dataset[:,1], dataset[:,2], '.', c='orange', markersize=4)
    # collection = Line3DCollection(ls, linewidths=0.5, colors='blue', alpha=alpha, zorder=0, label = f'# of particles {len(dataset)/2:.0f}')
    # ax.add_collection(collection)
    plotCubeAt(pos=detector_pos, size=detector_size, ax=ax, color='black', alpha=1)

    ax.plot(dataset[0,0], dataset[0,1], dataset[0,2], '.', c='black', markersize=40, zorder=4, label='position of 1st particle')
    ax.legend()
    # ax.set_aspect('equal')

    plt.savefig(f'{name}.pdf', bbox_inches="tight", dpi = dpi)
    if (show):
        plt.show()
