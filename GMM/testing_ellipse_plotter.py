# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:44:28 2021

Making sure the Gaussian mixture plotter actually works

@author: benpg
"""

from utils import draw_ellipse
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal as N

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

mean = np.array([1.,2.])
cov = np.array([[2,0.1],[0.1,0.5]])

sample_points = N(mean,cov,100000)

plt.plot(sample_points[:,0],sample_points[:,1],'rx', alpha=0.1)

x, y = draw_ellipse(mean,cov,num_sd=1)
plt.plot(x,y,'k')
x, y = draw_ellipse(mean,cov,num_sd=2)
plt.plot(x,y,'k')
x, y = draw_ellipse(mean,cov,num_sd=3)
plt.plot(x,y,'k')

# It doesn't fucking work!!!


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    #return ax.add_patch(ellipse)
    return ellipse

fig = plt.figure()
x,y = sample_points[:,0],sample_points[:,1]
ax = fig.add_subplot(1,1,1)
ell = confidence_ellipse(x,y,ax)
ax.add_patch(ell)
fig.show()

