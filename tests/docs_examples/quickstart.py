# make_joint
import JMCtools.distributions as jtd
import scipy.stats as sps
import numpy as np

joint = jtd.JointModel([sps.norm,sps.norm])
# sample_pdf
null_parameters = [{'loc': 3, 'scale': 1}, 
                   {'loc': 1, 'scale': 2}]
samples = joint.rvs((10000,),null_parameters)
# check_pdf
# Compute 2D PDF over grid
nxbins=100
nybins=100
x = np.linspace(-2,8,nxbins)
y = np.linspace(-6,10,nybins)
X, Y = np.meshgrid(x, y)
dxdy = (x[1]-x[0]) * (y[1]-y[0])
PDF = joint.pdf([X,Y],null_parameters)

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
import JMCtools.models as jtm

def pars1(a):
   return {'loc': a, 'scale':1}

def pars2(b):
   return {'loc': b, 'scale':2}

model = jtm.ParameterModel(joint,[pars1,pars2])
# block_structure
print(model.blocks)
# alt_model
def pars3(a,b):
   return {'loc': a+b, 'scale':1}

model2 = jtm.ParameterModel(joint,[pars3,pars2])
print(model2.blocks)
# sim_data
null_parameters = {'a':3, 'b':1}
Ntrials = 10000
Ndraws = 1
data = model.simulate((Ntrials,Ndraws),null_parameters)
# find_MLEs
# Set starting values and step sizes for Minuit search
options = {'a':3, 'error_a':1, 'b': 1, 'error_b': 2}
Lmax, pmax = model.find_MLE_parallel(options,data,method='minuit',Nprocesses=3)
# Note that Lmax are the log-likelihoods of the MLEs, 
# and pmax are the parameter values.
# compute_stats
Lnull = model.logpdf(null_parameters)
LLR = -2*(Lnull - Lmax) # log-likelihood ratio

# Plot!
n, bins = np.histogram(LLR, bins=100, normed=True)
q = np.arange(0,9,0.01)
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',label="Minuit",c='r')
ax.plot(q,sps.chi2.pdf(q, 2),c='k',label="Asymptotic") 
ax.set_xlabel("LLR")
ax.set_ylabel("pdf(LLR)")
ax.set_ylim(0.001,2)
ax.set_xlim(0,9)
ax.set_yscale("log")
ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':14})

fig.savefig('quickstart_LLR.svg')
