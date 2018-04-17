"""Checking that we can handle scipy.stats multivariate objects"""

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

# First, test broadcasting rules for sps.multivariate_normal
test = sps.multivariate_normal

# Use 2D gaussian, with various means
means = np.random.uniform(size=(100,2))

cov = np.array([[1,0],[0,1]]) # single common covariance matrix
print(means)

#x = test.rvs(mean=means,cov=cov[np.newaxis,:,:])
#print("x.shape:",x.shape)


# Will fit a 1D gaussian and a 2D multivariate gaussian
joint = jtd.JointModel([sps.norm,(sps.multivariate_normal,2)])

# Parameter functions:
def pars1(x1):
    return {'loc': x1, 'scale': 1}

def pars2(x1,x2):
    #print("x1,x2:",x1,x2)
    return {'mean': np.array([x1,x2]).flatten(), # Structure here has to be a bit odd. Not sure why x1, x2 have strange dimensions.
             'cov': [[1,0],[0,1]]}

model = jtm.ParameterModel(joint,[pars1,pars2])

null_parameters = {'x1': 0, 'x2': 0}

Ntrials = int(1e4)
Ndraws = 1
samples = model.simulate(Ntrials,Ndraws,null_parameters)

# Check the structure of the data we get out of this
print(c.get_data_structure(samples[0]))

trial0 = samples[0]

print(trial0.shape)

# Ok now that works, but Ndraws still now last. What happens when we extract the pdf?
pdf = model.logpdf(null_parameters,trial0)

# Seems fine. But let's check that the Objective object used to feed minimisers knows
# how to correctly multiple the Nevents pdfs
# This part is checking internal implentation details, users don't generally need to
# deal with these objects.
objf = jtm.Objective(next(iter(model.blocks)),trial0)
print("objf:", objf.func_code)
print(objf(0,0))

# Might be ok? Try fitting.
options = {'x1':0,'x2':0,'error_x1':1,'error_x2':2}
Lmax, pmax = model.find_MLE_parallel(options,samples,method='minuit',Nprocesses=3) 

# Check that best fit parameters match the analytic MLEs
data = samples[0]
X = data[...,0].flatten()
Y = data[...,1].flatten()
Z = data[...,2].flatten()
# We have to flatten since the data is always at least 2D due to the (Ntrials,Nevents) structure

abf = {'x1': (X+Y)/2, 'x2': Z}

#for i in range(Ntrials):
#    print("found,analytic: 'x1': {0}, {1}; 'x2': {2}, {3}".format(pmax['x1'][i],abf['x1'][i],
#                                                                  pmax['x2'][i],abf['x2'][i]))
# These seem to be complete rubbish. Something wrong in parameter selection?

# Check if analytic best fits reproduce the numerically found MLE
# Unfortunately broadcasting doesn't seem to work for the mean vector in scipy.multivariate_normal, which is
# pretty shit. Also means the grid minimiser won't work on it. For now we can evaluate in a list comprehension.
MLEpdf = np.array([model.logpdf({key: val[i] for key,val in pmax.items()},
                                    samples[i]) for i in range(Ntrials)])
#for i in range(Ntrials):
#    print("Lmax, MLE_logpdf:", Lmax[i], MLEpdf[i])

# See if these have the correct distribution. Should basically be 2 independent Gaussian constraints,
# so the log pdf minus normalisation junk should be chi^2 distributed with 2 DOF.
# Null model with zero data should give the normalisation factors
# Edit: Hmm ok no it seems to be 1 DOF. I suppose because we maximise some parameters.
zero_data = [0,0,0]
print(model.logpdf(null_parameters,zero_data))
chi2 = -2*(Lmax - model.logpdf(null_parameters,zero_data))

n, bins = np.histogram(chi2, bins=50, normed=True)
q = np.arange(0,15,0.01)
fig= plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',label="Minuit",c='r')
ax.plot(q,sps.chi2.pdf(q, 1),c='k') 
ax.set_xlabel("LLR")
ax.set_ylabel("pdf(LLR)")
#ax.set_ylim(0.001,2)
#ax.set_xlim(0,9)
ax.set_yscale("log")
ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':16})

fig.savefig('multivariate_LLR.png')

# Looks good!
