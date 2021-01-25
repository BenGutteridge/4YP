import matplotlib as mpl
import sys
from IPython.display import Image
import imageio
import os
import datetime
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
from numpy.linalg import det, inv

def make_gif(gifname, filedir):
  gifname = str(datetime.datetime.now())[:-7].replace(':',';')
  with imageio.get_writer(gifname+'.gif', mode='I') as writer:
      for filename in sorted(os.listdir(filedir)):
          image = imageio.imread(os.path.join(filedir,filename))
          writer.append_data(image)

  Image(open("/content/%s.gif"%gifname,'rb').read())

def draw_ellipse(mu, cov, conf=.95):
    chiscale = scipy.stats.chi2.isf(1-conf,2)
    v, w = np.linalg.eigh(cov)
    v = 2. * np.sqrt(chiscale) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mu, v[0], v[1], 180. + angle)
    return ell

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def plot_GMM(X, mu, lam, pi, centres, covs, K, title, savefigpath=False):
    plt.figure()
    plt.xlim(-5,10)
    plt.ylim(-5,15)
    plt.plot(X[:,0], X[:,1], 'kx', alpha=0.2)
    
    legend = ['Datapoints']
    
    for k in range(K):
        plt.plot(mu[k][0],mu[k][1],'ro')
        cov = inv(lam[k])
        ell = draw_ellipse(mu[k], cov)
        ell.set_alpha(pi[k])
        ell.set_edgecolor('m')
        ell.set_fill(True)
        splot = plt.subplot(1, 1, 1)
        splot.add_artist(ell)
        
    # Plotting the ellipses for the GMM that generated the data
    for i in range(len(centres)):
        true_ell = draw_ellipse(centres[i], covs[i])
        true_ell.set_edgecolor('g')
        true_ell.set_fill(False)
        splot.add_artist(true_ell)
    # legend.append('Data generation GMM 1, var1=%.2f, var2=%.2f, cov=%.2f' %(covs[0][0,0],covs[0][1,1],covs[0][1,0]))
    # legend.append('Data generation GMM 2, var1=%.2f, var2=%.2f, cov=%.2f' %(covs[1][0,0],covs[1][1,1],covs[1][1,0]))
    plt.title(title)
    if isinstance(savefigpath, str):
        plt.savefig(savefigpath)
    else:
        plt.legend(legend)
        plt.show()
		
