"""Some demos of the basic usage of ParameterModel objects"""

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
    else:
        import JMCtools as jt
        import JMCtools.models as jtm
 
import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np

# My screen is now really high-res so make images bigger by default
plt.rcParams["figure.dpi"] = 3*72

      
# Create a JointModel, and define a mapping from some parameter
# space into the parameters of the JointModel distribution functions

# Freaky inter-dependent model that probably screws with Wilk's theorem regularity conditions
#def pars2_A(mu1,mu2):
#    return {"loc":mu1+mu2, "scale":1}
#
#def pars2_B(mu2,mu3):
#    return {"loc":mu2+mu3, "scale":1}
#
#def pars2_C(mu4):
#    return {"loc":mu4, "scale":1}

# Simple model for testing
def pars2_A(mu1):
    return {"loc":mu1, "scale":1}

def pars2_B(mu2):
    return {"loc":mu2, "scale":1}

def pars2_C(mu3):
    return {"loc":mu3, "scale":1}

# # Construct transformed distributions
# submodels = [(jtd.TransDist(sps.norm,pars2_A),['mu1']),
#              (jtd.TransDist(sps.norm,pars2_B),['mu2']),
#              (jtd.TransDist(sps.norm,pars2_C),['mu3'])]
# 
# # Combine transformed distributions into one object that manages their parameter dependency structure
# parmodel = jtm.ParameterModel.fromList(submodels)

# Could also construct it like this:
# jointmodel = jtd.JointModel([jtd.TransDist(sps.norm,pars2_A),
#                              jtd.TransDist(sps.norm,pars2_B),
#                              jtd.TransDist(sps.norm,pars2_C)])
# parmodel = jtm.ParameterModel(jointmodel,[['mu1'],['mu2'],['mu3']])
# 
# Or like this:
parmodel = jtm.ParameterModel([jtd.TransDist(sps.norm,pars2_A),
                               jtd.TransDist(sps.norm,pars2_B),
                               jtd.TransDist(sps.norm,pars2_C)]
                              ,[['mu1'],['mu2'],['mu3']])

# Define the null hypothesis
null_parameters = {'mu1':0, 'mu2':0, 'mu3':0}

# Get some test data (will be stored internally)
parmodel.simulate(10000,null_parameters)

# Set ranges for parameter "scan"
ranges = {}
for p in null_parameters.keys():
   ranges[p] = (-5,5)        

# N gives the number of grid points in each parameter direction
options = {"ranges": ranges, "N": 50}

# Get maximum likelihood estimators for all parameters
Lmax, pmax = parmodel.find_MLE(options)

print("results:", Lmax.shape, [p.shape for p in pmax.values()])

# Plot simple MLLR test statistic
# Should be chi-squared distributed with (3-0)=3 degrees of freedom for 'simple' model
Lnull = parmodel.logpdf(null_parameters)

MLLR = -2*(Lnull - Lmax)

n, bins = np.histogram(MLLR, bins=50, normed=True)
q = np.arange(0,15,0.01)
fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',label="T")
ax.plot(q,sps.chi2.pdf(q, 3),c='k')
ax.set_xlabel("MLLR")
ax.set_ylabel("pdf(MLLR)")
#ax.set_ylim(0,1)
fig.savefig('test_parametermodel_MLLR1.png')

# Suppose we want to profile some nuisance parameters out of the null model
# We can then get 'Lmax' with only the null parameters of interest fixed

# Set ranges for parameter "scan"
ranges0 = {'mu1': 0, 'mu2': (-5,5), 'mu3': (-5,5)}
options0 = {"ranges": ranges0, "N": 50}
Lmax0, pmax0 = parmodel.find_MLE(options0)

# LLR should now be chi-squared distributed with (3-2)=1 degrees of freedom
MLLR2 = -2*(Lmax0 - Lmax)
n, bins = np.histogram(MLLR2, bins=50, normed=True)
q = np.arange(0,15,0.01)
fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',label="T")
ax.plot(q,sps.chi2.pdf(q, 1),c='k')
ax.set_xlabel("MLLR")
ax.set_ylabel("pdf(MLLR)")
fig.savefig('test_parametermodel_MLLR2.png')

# One more, with two fixed parameters

# Set ranges for parameter "scan"
ranges0 = {'mu1': 0, 'mu2': 0, 'mu3': (-5,5)}
options0 = {"ranges": ranges0, "N": 50} # deviates a little from chi2 at low MLLR values, due to discretisation. Higher N reduces this effect.
Lmax0, pmax0 = parmodel.find_MLE(options0)

# LLR should now be chi-squared distributed with (3-1)=2 degrees of freedom
MLLR3 = -2*(Lmax0 - Lmax)
n, bins = np.histogram(MLLR3, bins=50, normed=True)
q = np.arange(0,15,0.01)
fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',label="T")
ax.plot(q,sps.chi2.pdf(q, 2),c='k')
ax.set_xlabel("MLLR")
ax.set_ylabel("pdf(MLLR)")
fig.savefig('test_parametermodel_MLLR3.png')






