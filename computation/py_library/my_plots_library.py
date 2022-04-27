import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# import inspect
# function_name = inspect.stack()[0][3]
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14


# def pp_get_pos(pos_obj):
#     # a = np.empty(3, dtype=float)
#     a = [float]*3
#     a[0] = pos_obj.x
#     a[1] = pos_obj.y
#     a[2] = pos_obj.z
#     return a


def pp_get_pos(pos_obj):
    return [pos_obj.x, pos_obj.y, pos_obj.z]


# expecting energy in MeV
def plot_energy_std(energy_array, binsize = 20, 
                        name = 'plot_energy_std', xlabel_unit = 'MeV', show=True, **kwargs):
    plt.xscale('log')
    plt.xlabel(fr'$E \,/\, \mathrm{{{xlabel_unit}}}$')
    plt.ylabel("# of particles")
    plt.title(name)
    bins = np.geomspace(min(energy_array), max(energy_array), binsize)
    plt.hist(energy_array, bins=bins, log=True, label=name, **kwargs)
    # plt.savefig(f'figures/{name}.pdf', bbox_inches="tight")
    # if (show):
    #     plt.show()


# expecting distances in cm
def plot_distances_std(distances_array, binsize = 20, 
                        name = 'plot_distances_std', xlabel_unit = 'cm', show=True, **kwargs):
    # plt.xscale('log')
    plt.xlabel(fr'propagated distance $\,/\, \mathrm{{{xlabel_unit}}} $')
    plt.ylabel("# of particles")
    plt.title(name)
    bins = np.linspace(min(distances_array), max(distances_array), binsize)
    plt.hist(distances_array, bins=bins, log=True, **kwargs)
    # plt.savefig(f'figures/{name}.pdf', bbox_inches="tight")
    # if (show):
    #     plt.show()

# versatile histogram plotting function
def plot_hist(array,
                ylabel = '# of muons',
                x_label1 = 'physical property', 
                xlabel_unit = 'unit', 
                name='plot_hist_log',
                label=None,
                xlog=False, 
                binsize=None,
                show_or_multiplot=True, 
                savefig=False,
                **kwargs):
    plt.xlabel(fr'${{{x_label1}}} \,/\, \mathrm{{{xlabel_unit}}}$')
    plt.ylabel(f"{ylabel}")
    plt.title(name)
    if (binsize != None):
        if (xlog):
            bins = np.geomspace(min(array), max(array), binsize)
        else:
            bins = np.linspace(min(array), max(array), binsize)
    else:
        if (xlog):
            bins = np.geomspace(min(array), max(array), 30)
        else:
            bins = binsize

    if (xlog):
        plt.xscale('log')
    
    plt.hist(array, bins=bins, log=xlog, label=label, **kwargs)
    plt.legend()
    if (show_or_multiplot):
        plt.show()
    if (savefig and not (show_or_multiplot)):
        plt.savefig(f'figures/{name}.pdf', bbox_inches="tight")


def plot_3D_start_end(dataset, 
            detector_pos, 
            detector_size, 
            elev=30.0, 
            azim=30, 
            alpha=0.1, 
            name = 'plot_3D_start_end', 
            title = 'plot_3D_start_end', 
            dpi=400, 
            show=True, 
            **kwargs):
    # original function by dominik baar
    # fig = plt.figure(figsize=(16, 16))
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, pad = -200)

    # üöpt 
    ax.plot(dataset[:,0], dataset[:,1], dataset[:,2], '.', c='orange', markersize=4)

    # ls = dataset[:,0:3].reshape((-1,2,3))
    # collection = Line3DCollection(ls, linewidths=0.5, colors='blue', alpha=alpha, zorder=0, label = f'# of particles {len(dataset)/2:.0f}')
    # ax.add_collection(collection)

    plotCubeAt(pos=detector_pos, size=detector_size, ax=ax, color='black', alpha=1)

    ax.plot(dataset[0,0], dataset[0,1], dataset[0,2], '.', c='black', markersize=40, zorder=4, label='particle origin')
    ax.legend()
    # ax.set_aspect('equal')

    plt.savefig(f'figures/{name}.pdf', bbox_inches="tight", dpi = dpi)
    if (show):
        plt.show()


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
    

'''
test = np.array(
    [
        [0, 0, 0], [1, 1, 1], [0, 0, 0],
        [2,  2, 2], [0, 0, 0], [3, 3, 3],
        [0, 0, 0], [4, 4, 4], [0, 0, 0],
        [5, 5, 5], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [99, 99, 99]
    ]
)

'''