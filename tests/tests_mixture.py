"""Checking properties of MixtureModels via ParameterModel objects"""

# Some trickery for relative imports, see: https://stackoverflow.com/a/27876800
if __name__ == '__main__':
    if __package__ is None:
        import sys
        import os
        sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import JMCtools as jt
        import JMCtools.distributions as jtd
        import JMCtools.models as jtm
        import JMCtools.common as c
    else:
        import JMCtools as jt
        import JMCtools.models as jtm
        import JMCtools.common as c
 
import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import time

# My screen is now really high-res so make images bigger by default
plt.rcParams["figure.dpi"] = 3*72

# Test 1 -- Define mixture model as mixture of two Gaussians
mix = jtd.MixtureModel([sps.norm,sps.norm])

# Test that we can get the pdf with chosen parameters for Gaussians plus mixing parameters
eta = 0.8
w=[eta, 1-eta]
pars=[{'loc':3,'scale':1},{'loc':5,'scale':2}]
x = np.linspace(0,10,100)
p = mix.pdf(x,weights=w,parameters=pars)

fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,p)
ax.set_xlabel("x")
ax.set_ylabel("pdf(x)")
fig.savefig("mix_test_1_pdf.png")

# --- Test 2: 2D, using JointModel to define each 2D gaussian
mix = jtd.MixtureModel([jtd.JointModel([sps.norm,sps.norm]),
                        jtd.JointModel([sps.norm,sps.norm])])

eta = 0.8
w=[eta, 1-eta]
# Sub-models are JointModels, so those each require a list of dictionaries as parameters
# But MixtureModel also takes a list of dictionaries to pass to each sub-model, so there are
# two layers of lists of dictionaries here.
pars=[{'parameters':[{'loc':2,'scale':1.5},{'loc':5,'scale':2}]},
      {'parameters':[{'loc':1,'scale':1},{'loc':5.5,'scale':1}]}]
nxbins=100
nybins=100
x = np.linspace(-3,13,nxbins)
y = np.linspace(-3,13,nybins)
X, Y = np.meshgrid(x, y)
# Let's also sample from this distribution. Easier to freeze it first.
mix_frozen = mix(weights=w,parameters=pars)

dxdy = (x[1]-x[0]) * (y[1]-y[0])
data = np.stack([X,Y],axis=-1) # components need to be in last axis
PDF = mix_frozen.pdf(data) * dxdy # add in bin weights
samples = mix_frozen.rvs((10000,))

# Construct smallest intervals containing certain amount of probability
outarray = np.ones((nxbins,nybins))
sb = np.argsort(PDF.flat)[::-1]
outarray.flat[sb] = np.cumsum(PDF.flat[sb])  #cumulative sum of probabilities

fig= plt.figure()
ax = fig.add_subplot(111)
ax.contourf(X, Y, outarray, alpha=0.3, levels=[0,0.68,0.95,0.997])
ax.scatter(*samples.T,lw=0,s=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig("mix_test_2_pdf.png")

# --- Test 3: Using model from test 2, construct test statistic for testing mixing parameter

# Only one "block" of parameters for this model
# This is a fairly serious limitation of the mixture model; even if the models being
# mixed have a nice block-diagonal structure, the mixture does not.
def pars(eta,locx):
   weights=[1-eta,eta]
   pars=[{'parameters':[{'loc':locx,'scale':1.5},{'loc':5,'scale':2}]},
         {'parameters':[{'loc':1,'scale':1},{'loc':5.5,'scale':1}]}]
   return {'weights': weights, 'parameters': pars}

# Need to put the mixture model into a trival JointModel in order to use it with ParameterModel
# Will need to have a bunch of samples from the mixture distribution for asymptotic formulae to work
# This is a good test of how well this framework scales...

def pars(eta,locx):
   weights=[1-eta,eta]
   pars=[{'parameters':[{'loc':locx,'scale':1.5},{'loc':5,'scale':2}]},
         {'parameters':[{'loc':1,'scale':1},{'loc':5.5,'scale':1}]}]
   return {'weights': weights, 'parameters': pars}

# Put it in a JointModel so we can make a ParameterModel out of it
mixj = jtd.JointModel([(mix,2),(mix,2)])
parmix = jtm.ParameterModel(mixj,[pars,pars])

# The asymptotic formula won't work for just one sample per experiment. Need a bunch of
# samples.
Nevents = 9 # Will draw from our distribution this many times per trial
# Hmm, too many events seems to actually screw it up! This might be because
# it becomes possible to locate the MLE's at higher accuracy than the
# discretisation scale used here. Probably need to implement a real
# minimizer in order to get this correct. 
# But that is a bit tough to do repeatedly!
# Can we use e.g. minuit in some python extension module code?

# Define the null hypothesis
null_parameters = {'eta':0., 'locx':2} #, 'locx2': 3}

# Get some test data (will be stored internally)
# It takes some RAM to do this for lots of samples at once, so
# let's do in batches

# Set ranges for parameter "scan"
ranges0 = {'eta':0, 'locx':[-10,10]}#, 'locx2':[-15,15]}
options0 = {"ranges": ranges0, "N": 40}

ranges = {'eta':[0,1], 'locx':[-10,10]}#, 'locx2':[-15,15]}
options = {"ranges": ranges, "N": 40}

min_defs = {'pedantic': False, 'print_level': -1} 
m_options = {'eta':0.5, 'locx':2, 'error_eta': 0.01, 'error_locx':0.05, 'limit_eta':(0,1), 'limit_locx':(-10,10), **min_defs}
m0_options = {'eta':0, 'fix_eta':True, 'error_eta': 0, 'locx':2, 'error_locx':0.1, 'limit_eta':(0,1), 'limit_locx':(-10,10), **min_defs}

fixed = ['eta']

Ntrials = 5000 
print("{0:.2e} events to simulate in total...".format(Nevents*Ntrials))

# Simulate all the data up-front so that we can compare MLE-finding results
data = parmix.simulate(Ntrials,Nevents,null_parameters)

print("Structure of simulated data (elements should be {0}: {1}".format(data[1], c.get_data_structure(data[0])))

# Number of MLE scans to perform in parallel
Nproc = 3

print("parmix.blocks:",parmix.blocks)

# Get grid scan MLEs
Lmax_grid_all, pmax_grid = parmix.find_MLE_parallel(options,data,method='grid',Nprocesses=Nproc)
Lmax0_grid_all, pmax0_grid = parmix.find_MLE_parallel(options0,data,method='grid',Nprocesses=Nproc)

# Get Minuit MLEs
#Lmax_all, pmax_all   = parmix.find_MLE_parallel(m_options,data,method='minuit',Nprocesses=Nproc,seeds=pmax_grid)
#Lmax0_all, pmax0_all = parmix.find_MLE_parallel(m0_options,data,method='minuit',Nprocesses=Nproc,seeds=pmax0_grid)

# Lmax should always be greater than Lmax0, so enforce this (to correct small numerical errors)
#m = Lmax_all < Lmax0_all
#Lmax_all[m] = Lmax0_all[m]

# Plot simple MLLR test statistic
# should be delta(0) + chi2(k=(2-1=1))
#MLLR = -2*(Lmax0_all - Lmax_all)

MLLR_grid = -2*(Lmax0_grid_all - Lmax_grid_all)

#n, bins = np.histogram(MLLR, bins=50, normed=True)
n_g, bins_g = np.histogram(MLLR_grid, bins=50, normed=True)
q = np.arange(0,15,0.01)
fig= plt.figure(figsize=(6,8))
ax = fig.add_subplot(211)
#ax.plot(bins[:-1],n,drawstyle='steps-post',label="Minuit",c='r')
ax.plot(bins_g[:-1],n_g,drawstyle='steps-post',label="grid",c='b')
ax.plot(q,0.5*sps.chi2.pdf(q, 1),c='k') 
ax.set_xlabel("MLLR")
ax.set_ylabel("pdf(MLLR)")
ax.set_ylim(0,2)
ax.set_xlim(0,9)
ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':16})

# Log scale
ax = fig.add_subplot(212)
#ax.plot(bins[:-1],n,drawstyle='steps-post',label="Minuit",c='r')
ax.plot(bins_g[:-1],n_g,drawstyle='steps-post',label="grid",c='b')
ax.plot(q,0.5*sps.chi2.pdf(q, 1),c='k') 
ax.set_xlabel("MLLR")
ax.set_ylabel("pdf(MLLR)")
ax.set_ylim(0.001,2)
ax.set_xlim(0,9)
ax.set_yscale("log")
ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':16})

fig.savefig('mixture_MLLR.png')

# Hmm not really working, did we violate Chernoff's theorem?
#
# Or perhaps we just aren't in the asymptotic regime. We are effectively just testing with one sample at a time,
# so that's not very asymptotic.
#
# Let us test with a bunch of samples then.
# Need to construct the joint model for all samples.
#
# Ok does much better with more samples, but still not perfect... just asymptotics? Or our discretisation?
# Multiplying the number of samples stretches the abilities of these list-based objects quite a bit,
# for big datasets I guess it is necessary to construct a dedicated pdf/rvs object that can be sampled from/
# have it's pdf evaluated quickly.



