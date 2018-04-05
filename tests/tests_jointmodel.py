"""Simple test suite for JMCtools package"""

# Some trickery for relative imports, see: https://stackoverflow.com/a/27876800
if __name__ == '__main__':
    if __package__ is None:
        import sys
        import os
        sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import JMCtools as jt
        import JMCtools.distributions as jtd
    else:
        import JMCtools as jt
        import JMCtools.distributions as jtd
 
import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np

# Test basic constructions

gauss2D  = jtd.JointModel([sps.norm(loc=3,scale=1), sps.norm(loc=10,scale=2)])
gauss2Db = jtd.JointModel([sps.norm(loc=5,scale=1), sps.norm(loc=15,scale=2)])

mix = jtd.MixtureModel([gauss2D,gauss2Db],[0.3,0.7])

print( mix.pdf([3,5]) )
print( mix.rvs((5,))  )# size has to be a tuple, I didn't make a 0d special case.
print( mix.rvs((3,4)) )

# Plots?

# 2D pdf
x = np.arange(0,10,0.1)
y = np.arange(0,20,0.2)
X, Y = np.meshgrid(x, y)
PDF = mix.pdf([X,Y])
im = plt.imshow(PDF, interpolation='bilinear', aspect='auto',
                origin='lower', extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
plt.show()

# 2D scatter plot of random samples
samples = mix.rvs((10000,))
plt.scatter(*samples,lw=0,s=1)
plt.show()
