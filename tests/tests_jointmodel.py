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


# Can't actually do this right now
gauss2D  = jtd.JointModel([sps.norm(loc=3,scale=1), sps.norm(loc=10,scale=2)], frozen=True)
gauss2Db = jtd.JointModel([sps.norm(loc=5,scale=1), sps.norm(loc=15,scale=2)], frozen=True)

# This works though
#gauss2D  = jtd.JointModel([sps.norm, sps.norm], [{"loc":3,"scale":1}, {"loc":10,"scale":2}])
#gauss2Db = jtd.JointModel([sps.norm, sps.norm], [{"loc":5,"scale":1}, {"loc":15,"scale":2}])

mix = jtd.MixtureModel([gauss2D,gauss2Db],[0.3,0.7])

print( gauss2D.rvs(3) )
print( mix.pdf([3,5]) )
print( mix.rvs((3,4)) )
print( mix.rvs(5)  )# size has to be a tuple, I didn't make a 0d special case.
# Plots?

# 2D pdf
x = np.arange(0,10,0.1)
y = np.arange(0,20,0.2)
X, Y = np.meshgrid(x, y)
# last dimension needs to index components
data = np.stack([X,Y],axis=-1)
print("data.shape:",data.shape)
PDF = mix.pdf(data)
print("PDF.shape:", PDF.shape)
im = plt.imshow(PDF, interpolation='bilinear', aspect='auto',
                origin='lower', extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
plt.show()

# 2D scatter plot of random samples
samples = mix.rvs(10000)
print("samples.shape:",samples.shape)
plt.scatter(*samples.T,lw=0,s=1)
plt.show()
