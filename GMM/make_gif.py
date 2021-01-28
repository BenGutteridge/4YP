# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:47:48 2021

@author: benpg

GIF creator
"""

import imageio, os

def make_gif(gifname, filedir):
    images = []
    # for filename in os.listdir(filedir):
    #     images.append(imageio.imread(filedir+filename))
    # imageio.mimsave(gifname+'.gif', images)
    
        
    with imageio.get_writer(gifname+'.gif', mode='I') as writer:
        for filename in os.listdir(filedir):
            image = imageio.imread(os.path.join(filedir,filename))
            writer.append_data(image)