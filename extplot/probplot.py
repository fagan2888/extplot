import numpy as np
import scipy.stats as ss
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

def plot_3d_probs(Wx,X,Y,Z, cutoff, resolution, ax=None, color='b', alpha=1.0, 
               xlim=None, ylim=None, zlim=None):
    '''
    Create a 3D plot showing an isosurface for a probability value of 'cutoff'

    The vertices of the polygons are generated using the Marching Cubes algorithm
    as implemented in the PyMCubes library:

        https://github.com/pmneila/PyMCubes

    :param Wx: The probability values on a grid.
    :param X: The X grid (generated using meshgrid)
    :param Y: The Y grid (generated using meshgrid)
    :param Z: The Z grid (generated using meshgrid)
    :param cutoff: The cutoff value for the isosurface.
    :param resolution: The width, height and length of the grid (as a single scalar value)
    :param ax: An axis object. If None, then we create our own
    :param color: The color for the surface
    :param alpha: How transparent the surface should be
    :param xlim: The limits of the x-axis (and data)
    :param ylim: The limits of the y-axis (and data)
    :param zlim: The limits of the z-axis (and data)
    '''
    import mcubes
    vertices, triangles = mcubes.marching_cubes(Wx, cutoff)

    scaled_vertices = np.array([xlim[0] + (xlim[1] - xlim[0]) * vertices[:,0] / resolution, 
                            ylim[0] + (ylim[1] - ylim[0]) * vertices[:,1] / resolution, 
                            zlim[0] + (zlim[1] - zlim[0]) * vertices[:,2] / resolution]).T
    verts = [[scaled_vertices[i] for i in t] for t in triangles]

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    #print "ax 2", ax
    
    if ax is None:
        #print "here3"
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1,projection='3d')
        
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=alpha))

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])

    ax.set_xlabel('$d$')
    ax.set_ylabel('$\phi$')
    ax.set_zlabel('$\psi$')
    
    return ax
    #plt.show()

def plot_2d_probsurface(data, resolution=20, ax = None, 
                        xlim=None, ylim=None, kde=None,
                        zdir='', offset=0):
    '''
    Plot a 2D probability surface on a 3D plot.

    :param data: A 2D numpy array.
    :param resolution: The width, height and length of the grid (as one scalar value)
    :param ax: The axis object
    :param xlim: The limits of the x-axis
    :param ylim: The limits of the y-axis
    :param kde: The kernel density estimate of the data. If None, then generate it automatically from the passed data
    :param zdir: The location of the z-axis ('x', 'y', or 'z')
    :param offset: Where to draw the contours along the z-axis
    '''
    if kde is None:
        # create a function to calcualte the density at a particular location
        kde = ss.gaussian_kde(data.T)
    
    # calculate the limits if there are no values passed in
    # passed in values are useful if calling this function
    # systematically with different sets of data whose limits
    # aren't consistent
    if xlim is None:
        xlim = (min(data[:,0]), max(data[:,0]))
    
    if ylim is None:
        ylim = (min(data[:,1]), max(data[:,1]))

    # create some tick marks that will be used to create a grid
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    
    # wrap the KDE function and vectorize it so that we can call it on
    # the entire grid at once
    def calc_prob(x,y):
        return kde([x,y])[0]
    calc_prob = np.vectorize(calc_prob)
    
    # check if we've received a plotting surface
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
    
    # create the grid and calculate the density at each point
    X,Y = np.meshgrid(xs, ys)
    Z = calc_prob(X,Y) 
    
    # plot the contour
    if zdir != '':
        if zdir == 'x':
            cont = ax.contour(Z,X,Y, zdir=zdir, offset=offset)
        if zdir == 'y':
            cont = ax.contour(X,Z,Y, zdir=zdir, offset=offset)
        if zdir == 'z':
            cont = ax.contour(X,Y,Z, zdir=zdir, offset=offset)
    else:
        cont = ax.contour(X,Y,Z, zdir=zdir, offset=offset)
        
    if data is not None:
        point_values = kde(data.T)
        #print point_values
        #print cont
        if zdir != '':
            ax.scatter(data[:,0], data[:,1], zs=offset, c=point_values, zdir=zdir, cmap=cont.cmap, norm=cont.norm)
        else:
            ax.scatter(data[:,0], data[:,1], c=point_values, cmap=cont.cmap, norm=cont.norm)
        #ax.scatter(data[:,1], data[:,0], c=point_values, zdir=zdir)
    
    return {"xlab": None, "ylab": None, "contour": cont}


def plot_3d_and_2d_distribution(data, resolution=20, ax = None, color='b', filename=None, title=None, xlim=None, ylim=None, zlim=None, cutoff=None):
    '''
    Draw both an isosurface as well as the corresponding 2D contour plots all in one go.
    
    :param data: A 3D numpy array of data points.
    :param resolution: The number of points in each dimension of the grid.
    :param ax: An axis object to plot to
    :param color: The color to draw the isosurface in
    :param filename: A filename to store this graph to
    :param title: The title of this graph
    :param xlim: The limits of the x-axis
    :param ylim: The limits of the y-axis
    :param zlim: The limits of the z-axis
    :param cutoff: The probability value along which to draw the isosurface
    '''

    #sub_data = data[:,cols]
    prob = ss.gaussian_kde(data.T)
    
    def calc_all_prob(x,y,z):
        return prob([x,y,z])[0]
    
    calc_all_prob = np.vectorize(calc_all_prob)

    if xlim is None:
        xlim = (min(data[:,0]), max(data[:,0]))

    if ylim is None:
        ylim = (min(data[:,1]), max(data[:,1]))

    if zlim is None:
        zlim = (min(data[:,2]), max(data[:,2]))
    
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    zs = np.linspace(zlim[0], zlim[1], resolution)
    
    X,Y,Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    W = calc_all_prob(X,Y,Z)

    if cutoff is None:
        cutoff = np.mean(W.flatten())
    
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = Axes3D(fig)
        
    #print W
    ax = plot_3d_probs(W,X,Y,Z,cutoff, resolution, ax=ax, alpha=0.4, color=color, 
                    xlim=xlim, ylim=ylim, zlim=zlim)
    
    '''
    if sorted(cols) == [0,1]:
        zdir = 'z'
    elif sorted(cols) == [0,2]:
        zdir = 'y'
    elif sorted(cols) == [1,2]:
        zdir = 'x'
    else:
        print >>sys.stderr, "Z-direction undefined, cols:", sorted(cols)
        return
    '''
        
    for cols, zdir,offset, plot_xlim, plot_ylim in [((0,1), 'z', xlim[0], xlim, ylim), 
                              ((0,2), 'y', zlim[1], xlim, zlim), 
                              ((1,2), 'x', ylim[0], ylim, zlim)]:
        pl = plot_2d_probsurface(data[:,cols], xlim =plot_xlim, ylim=plot_ylim, ax = ax, 
                            zdir=zdir, offset=offset, resolution=resolution)
        
    
    if fig is not None:
        cbar = fig.colorbar(pl['contour'], shrink=0.5)
        cbar.set_label('Probability Density', rotation=270, labelpad=20)
        
    if title is not None:
        ax.set_title(title, y=1.03)

    if filename is not None:
        # blah
        plt.savefig(filename, bbox_inches='tight')

    return {"xlab": None, "ylab": None, "contour": pl['contour'], 'fig': fig, 'ax': ax}

def plot_3d_and_2d_distdiff(data1, data2, resolution=20, ax = None, color='b', cutoff=0.008, filename=None, title=None):
    '''
    Plot the difference between two density distributions. If probability density of the
    datasets is W1 and W2, respectively, the the plotted data will be W1 / (W1 + W2)

    :param data1: A 3D dataset
    :param data2: A 3D dataset
    :param resolution: The width of the grid
    :param ax: An axis object to plot to. A new one is created if this is passed as null.
    :param color: The color of the isosurface
    :param cutoff: The probability value along which to draw the isosurface
    :param filename: A filename to save the plot to
    :param title: The title of the plot
    '''
    #sub_data = data[:,cols]
    prob1 = ss.gaussian_kde(data1.T)
    prob2 = ss.gaussian_kde(data2.T)
    
    def calc_all_prob(x,y,z,prob):
        return prob([x,y,z])[0]
    
    calc_all_prob = np.vectorize(calc_all_prob)
    
    xs = np.linspace(0, 30, resolution)
    ys = np.linspace(0, 3.14, resolution)
    zs = np.linspace(0, 3.14, resolution)
    
    X,Y,Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    #W = calc_all_prob(X,Y,Z)
    W1 = calc_all_prob(X,Y,Z, prob1)
    W2 = calc_all_prob(X,Y,Z, prob2)
    
    W = W1 / (W1 + W2)
    
    fig = None
    #print "ax:", ax
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = Axes3D(fig)
        
    #print W
    plot_probs(W,X,Y,Z,cutoff, resolution, ax=ax, alpha=0.4, color=color)
    
    for cols, zdir,offset, xlim, ylim in [((0,1), 'z', 0, (0, 30), (0,3.14)), 
                              ((0,2), 'y', 3.14, (0, 30), (0,3.14)), 
                              ((1,2), 'x', 0, (0, 3.14), (0,3.14))]:
        
        kde1 = ss.gaussian_kde(data1[:,cols].T)
        kde2 = ss.gaussian_kde(data2[:,cols].T)
        
        kde = lambda x: (kde1(x) * (len(data1) / float(len(data2))))  / (kde1(x) + kde2(x))
            
        print "cols:", cols, offset, zdir
        pl = plot_2d_probsurface(data = None, xlim =xlim, ylim=ylim, ax = ax, 
                            zdir=zdir, offset=offset, resolution=resolution, kde=kde)
        
    if title is not None:
        ax.set_title(title, y=1.03)
    
    if fig is not None:
        cbar = fig.colorbar(pl['contour'], shrink=0.5)
        cbar.set_label('Probability Density', rotation=270, labelpad=20)

    if filename is not None:
        # blah
        plt.savefig(filename, bbox_inches='tight')
        
    return {"xlab": None, "ylab": None, "contour": pl['contour'], 'fig': fig, 'ax': ax}

