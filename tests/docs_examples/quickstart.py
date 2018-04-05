# make_joint
import JMCtools.distributions as jtd
import scipy.stats as sps
import numpy as np

joint = jtd.JointModel([sps.norm,sps.norm])
# sample_pdf
parameters = [{'loc': 3, 'scale': 1}, 
              {'loc': 1, 'scale': 2}]
samples = joint.rvs((10000,),parameters)
# check_pdf
# Compute 2D PDF over grid
nxbins=100
nybins=100
x = np.linspace(-2,8,nxbins)
y = np.linspace(-6,10,nybins)
X, Y = np.meshgrid(x, y)
dxdy = (x[1]-x[0]) * (y[1]-y[0])
PDF = joint.pdf([X,Y],parameters)

# Construct smallest intervals containing certain amount of probability
outarray = np.ones((nxbins,nybins))
sb = np.argsort(PDF.flat)[::-1]
outarray.flat[sb] = np.cumsum(PDF.flat[sb] * dxdy)

# Make plot!
import matplotlib.pyplot as plt
fig= plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
ax.contourf(X, Y, outarray, alpha=0.3, levels=[0,0.68,0.95,0.997])
ax.scatter(*samples,lw=0,s=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig("example_2D_joint.svg")
# build_model 

