# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:15:41 2021

@author: benpg

PLAYING WITH AXES
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
color='g', alpha=0.5)

# ellipse = mpl.patches.Ellipse()

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

# create the figure and the axis in one shot
fig, ax = plt.subplots(1,figsize=(6,6))

art = mpatches.Circle([0,0], radius = 1, color = 'r')
#use add_patch instead, it's more clear what you are doing
ax.add_patch(art)

art = mpatches.Circle([0,0], radius = 0.1, color = 'b')
ax.add_patch(art)

#set the limit of the axes to -3,3 both on x and y
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)

plt.show()
