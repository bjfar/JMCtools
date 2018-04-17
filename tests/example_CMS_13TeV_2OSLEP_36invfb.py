"""Application of this framework to the realistic problem of the
   analysis described in https://arxiv.org/abs/1709.08908,
   i.e. the CMS search including two opposite-sign, same-flavour 
   leptons, jets, and MET at 13 TeV.

   Simplified likelihood construction follows 
   https://cds.cern.ch/record/2242860
"""

import JMCtools as jt
import JMCtools.distributions as jtd
import JMCtools.models as jtm
import JMCtools.common as c

import scipy.stats as sps
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import time
import pickle

regenerate = True
Nsamples = int(1e2)
tag = "1e2" # tag for saved data

# My screen is now really high-res so make images bigger by default
plt.rcParams["figure.dpi"] = 3*72

# First, the data from the CMS paper that is required to build the
# joint pdf:
# (I actually just copied this out of
#  gambit/ColliderBit/src/analyses/Analysis_CMS_13TeV_2OSLEP_36invfb.cpp
#  and made it Python format)

# Observed event counts
CMS_o = [57., 29., 2., 0., 9., 5., 1.]

# Background estimates (means)
CMS_b = [54.9, 21.6, 6., 2.5, 7.6, 5.6, 1.3]
   
# Background uncertainties, same-flavor signal regions
#CMS_bsame = [7., 5.6, 1.9, 0.9, 2.8, 1.6, 0.4]

# Covariance matrix for nuisance parameter measurements
CMS_cov = [
  [ 52.8, 12.7,  3.0,  1.2,  4.5,  5.1,  1.2 ],
  [ 12.7, 41.4,  3.6,  2.0,  2.5,  2.0,  0.7 ],
  [  3.0,  3.6,  1.6,  0.6,  0.4,  0.3,  0.1 ],
  [  1.2,  2.0,  0.6,  1.1,  0.3,  0.1,  0.1 ],
  [  4.5,  2.5,  0.4,  0.3,  6.5,  1.8,  0.4 ],
  [  5.1,  2.0,  0.3,  0.1,  1.8,  2.4,  0.4 ],
  [  1.2,  0.7,  0.1,  0.1,  0.4,  0.4,  0.2 ],
]

# For this CMS analysis, a simplified joint PDF can be 
# constructed as a product of numerous independent Poisson 
# distributions, which model events in various signal regions, and
# a large multivariate normal distribution which encodes correlations
# between constraints on nuisance parameters for the Poisson
# distributions. So our first task is to build this joint PDF.

N_regions = len(CMS_o)

poisson_part = [sps.poisson for i in range(N_regions)]
correlations = [(sps.multivariate_normal,7)]

# Define the parameter structure
#
# As it turns out, it is useful to do this in a loop, but that makes
# it difficult for the parameter introspection to work correctly.
# Therefore I will have to do something a bit ugly to dynamically 
# create these functions and put them in a list...

func_template = """\
def f_{i}(mu, s_{i}, theta_{i}):\n\
    l = np.atleast_1d(mu*s_{i} + CMS_b[{i}] + theta_{i})\n\
    m = (l<0)\n\
    l[m] = 0  #Poisson cannot have negative mean \n\
    return {{'mu': l}}\n\
\n\
f = f_{i} # Temporary alias name
"""

poisson_fs = []
for i in range(N_regions):
    func_def = func_template.format(i=i)
    if i==0:
        print("Defined function:") 
        print(func_def)
    exec(func_def)
    poisson_fs += [f]

# Version without signal scaling parameter 'mu', for use in uncorrelated
# fit. Each bin can then be fit independently.
func_template_nomu = """\
def fnomu_{i}(s_{i}, theta_{i}):\n\
    l = np.atleast_1d(s_{i} + CMS_b[{i}] + theta_{i})\n\
    m = (l<0) \n\
    l[m] = 0  #Poisson cannot have negative mean\n\
    return {{'mu': l}}\n\
\n\
f = fnomu_{i} # Temporary alias name
"""

poisson_fs_nomu = []
for i in range(N_regions):
    func_def = func_template_nomu.format(i=i)
    if i==0:
        print("Defined function:")
        print(func_def)
    exec(func_def)
    poisson_fs_nomu += [f]

# Parameters for the multivariate normal
# i.e the nuisance parameters.
# They are constructed such that the experimentally observed
# output of this multinormal is zero for all components (i.e.
# from control measurements of some kind)
# This observation constrains the nuisance parameters.
def multi_helper(*thetas):
    """Helper function to repack arguments into correct
    numpy array structure for sps.multivariate_normal
    """
    #print(thetas[0])
    means = np.array(thetas)
    #print(means.shape)
    return {'mean': means.flatten(),
             'cov': CMS_cov}

# String template to deal with the fact that thetas must all be passed as
# separate, correctly named arguments.
multi_template = """\
def multnorm_f({thetas}):\n\
    return multi_helper({thetas})
"""

thetas = ",".join(["theta_{0}".format(i) for i in range(N_regions)])
func_def = multi_template.format(thetas=thetas)
print("Defined function:") 
print(func_def)
exec(func_def)

# Let's also create a version which ignores the correlations
# This will be much faster to fit, and should still resemble
# the right answer, so can be a helpful cross-check.
# Just take the diagonal of the multivariate normal.
no_correlations = [sps.norm for i in range(N_regions)]

func_template_noc = """\
def c_{i}(theta_{i}):\n\
    return {{'loc': theta_{i},\n\
             'scale': np.sqrt(CMS_cov[{i}][{i}])}}\n\
f = c_{i} # Temporary alias name
"""

theta_fs_noc = []
for i in range(N_regions):
    func_def = func_template_noc.format(i=i)
    if i==0:
        print("Defined function:")
        print(func_def)
    exec(func_def)
    theta_fs_noc += [f]

# Create the joint PDF objects
joint     = jtd.JointModel(poisson_part + correlations)
joint_noc = jtd.JointModel(poisson_part + no_correlations)

# Connect the joint PDFs to the parameter structures
model     = jtm.ParameterModel(joint, poisson_fs + [multnorm_f])
model_noc = jtm.ParameterModel(joint_noc, poisson_fs_nomu + theta_fs_noc)

# Check the inferred block structures
print("model.blocks    :", model.blocks)
print("model_noc.blocks:", model_noc.blocks)

# Define null parameters
null_s = {"s_{0}".format(i): 0 for i in range(N_regions)}
null_theta = {"theta_{0}".format(i): 0 for i in range(N_regions)}
null_parameters = {"mu": 0 , **null_s, **null_theta}

# In order to perform some statistical test, we need a signal
# hypothesis. It is sort of cheating, but for testing let's just use
# the observed counts for this job. In reality we should use e.g. the
# predictions from our best fit MSSM point.
bf_s = np.array(CMS_o) - np.array(CMS_b)
# Another option: just make it look like the background.
# This will be a weaker test though since it is not tailored
# to the observed fluctuations. Of course the tailored test can
# be unreasonably strong I think.
#bf_s = np.array(CMS_b)
print("bf_s", bf_s)
# Some of these are negative which will cause problems: set them to zero
# in our fake signal. We'll also make the nominal signal bigger to helper the fitting routines
bf_s_fixed = np.zeros(bf_s.shape)
bf_s_fixed[bf_s>0] = bf_s[bf_s>0]
print("bf_s_fixed:", bf_s_fixed)
nominal_s = {"s_{0}".format(i): bf_s_fixed[i] for i in range(N_regions)}

# As a check, let's see what the best fit mu for each signal region independently would be
# under this null hypothesis
bf_mu = (np.array(CMS_o) - np.array(CMS_b)) / bf_s_fixed
print("bf_mu:", bf_mu)

# We now need to profile out the nuisance parameters for both null and
# alternate hypotheses
#
# We'll do this with Minuit, but it requires setting some options
#
theta_opt  = {'theta_{0}'.format(i) : 0 for i in range(N_regions)}
theta_opt2 = {'error_theta_{0}'.format(i) : 1.*np.sqrt(CMS_cov[i][i]) for i in range(N_regions)} # Get good step sizes from covariance matrix
fix_s = {'fix_s_{0}'.format(i): True for i in range(N_regions)}
m0_options = {'mu': 0, 'fix_mu': True, **theta_opt, **theta_opt2, **nominal_s, **fix_s} #actually the nominal_s does nothing here, equivalent to null_s since mu=0. Just emphasing that mu is the only difference between null and alternate.
#m_options = {'mu': 1, 'limit_mu': (0,1e99), 'error_mu': 0.1, **theta_opt, **theta_opt2, **nominal_s, **fix_s}

# Hmm, it seems we need to let signal strength fit go negative for this to work. 
# I guess that makes sense.
m_options = {'mu': 0.5, 'error_mu': 0.1, **theta_opt, **theta_opt2, **nominal_s, **fix_s} 

# Let's also fit a model where all the bins can be fit freely.
s_opt  = {'s_{0}'.format(i): 0 for i in range(N_regions)} # Maybe zero is a good starting guess
s_opt2 = {'error_s_{0}'.format(i) :  0.1*np.sqrt(CMS_cov[i][i]) for i in range(N_regions)} # Get good step sizes from covariance matrix. Should be similar for s_i I guess.
# This is a lot of parameters now, but probably minuit can handle it.
m2_options = {'mu': 1, 'fix_mu': True, **theta_opt, **theta_opt2, **s_opt, **s_opt2}  

# Settings for fit of uncorrelated model
verb = {'print_level': 1}
m_noc_options = {**theta_opt, **theta_opt2, **s_opt, **s_opt2}  

# Number of MLE scans to perform in parallel
Nproc = 5

# Sample from null distribution
if regenerate:
   samples = model.simulate(Nsamples,1,null_parameters)

   # Actually we can compute the MLEs for the uncorrelated model analytically.
   # We'll use them to check that the fit is working, and to seed the more
   # difficult model.
   seeds={}
   theta_seeds={}
   print("poisson samples shape:", samples[...,0].shape)
   print("multinorm samples shape:", samples[...,N_regions:].shape)
   bin_samples = samples[:,0,:N_regions].T
   theta_samples = samples[:,0,N_regions:].T
   for i in range(N_regions):
      theta_MLE = theta_samples[i]
      s_MLE = bin_samples[i] - CMS_b[i] - theta_MLE
      print('theta_MLE.shape, s_MLE.shape:', theta_MLE.shape, s_MLE.shape)
      seeds['theta_{0}'.format(i)] = theta_MLE
      seeds['s_{0}'.format(i)] = s_MLE 
      theta_seeds['theta_{0}'.format(i)] = theta_MLE # Just the theta seeds...
      print('data for bin {0}: {1}'.format(i,samples[:,0,i]))
   #print(samples)

   # Get MLEs using Minuit
   
   # Fit the simplified, uncorrelated model first. We can use the best fit signal strengths from this to
   # seed the fit of the properly correlated model.
   # To use the data in the uncorrelated model we actually have to modify it a bit
   # This is kind of undesirable, might need to review the structure of the sampled data
   # Couldn't we keep it in a big array after all? Does it matter what the substructure is if
   # we just know we have N variates in total?
   # Well, for now we have to do the following:
   #split_theta_samples = [np.array([theta_samples[i]]).T for i in range(N_regions)]
   #samples_noc = (samples[0][:N_regions] + split_theta_samples, samples[1]) 
   #print("samples_noc structure:", c.get_data_structure(samples_noc[0]))

   print("samples.shape:",samples.shape)
   print(samples)
   print("Fitting uncorrelated model...")
   Lmax_noc, pmax_noc = model_noc.find_MLE_parallel(m_noc_options,samples,method='minuit',Nprocesses=Nproc,seeds=seeds)

   # Check the Lmax for the analytically computed MLEs...
   #model.validate_data(samples,verbose=True)
   # Unfortunately broadcasting doesn't seem to work for the mean vector in scipy.multivariate_normal, which is
   # pretty shit. Also means the grid minimiser won't work on it. For now we can evaluate in a list comprehension.
   MLEpdf = np.array([model_noc.logpdf({key: val[i] for key,val in seeds.items()},
                                        samples[i]) for i in range(Nsamples)])

   for i in range(10):
      print("Lmax_nox, MLE_logpdf:", Lmax_noc[i], MLEpdf[i])
   # Ok so they match just fine...

   for i in range(10):
      print("theta_1_nox, theta_1_MLE:", pmax_noc['theta_1'][i], seeds['theta_1'][i])
 
   #for key in seeds.keys():
   #   print(key+":", seeds[key][-2], pmax_noc[key][-2])
   # ...So why are these completely wrong????
   # ...ok they aren't, just sometimes are completely wrong. Fit is unstable.
   # ...only last elements work? wtf?
   # Ahh ok, bugs in JMCtools, was not passing seeds on correctly. Fixed now.

   # Now fit the more complicated model
   print("Fitting null hypothesis...")
   Lmax0, pmax0 = model.find_MLE_parallel(m0_options,samples,method='minuit',Nprocesses=Nproc,seeds=theta_seeds)
   print("Fitting alternate hypothesis...")
   Lmax, pmax = model.find_MLE_parallel(m_options,samples,method='minuit',Nprocesses=Nproc,seeds=theta_seeds)
   print("Fitting alternate hypothesis 2 (free signal)...")
   Lmax2, pmax2 = model.find_MLE_parallel(m2_options,samples,method='minuit',Nprocesses=Nproc,seeds=seeds)

   print("Lmax_noc:", Lmax_noc) 
   print("Lmax0:",Lmax0)
   print("Lmax :",Lmax)
   print("Lmax2 :",Lmax2)
   print("pmax['mu']:",pmax['mu'])
   
   # Save data
   with open('saved_fit_CMS_2OS_{0}.pickle'.format(tag), 'wb') as handle:
       pickle.dump((Nsamples,samples,Lmax0,pmax0,Lmax,pmax,Lmax2,pmax2,Lmax_noc,pmax_noc), handle)

else:
   # Load data instead of sampling, if desired.
   with open('saved_fit_CMS_2OS_{0}.pickle'.format(tag), 'rb') as handle:
       Nsamples,samples,Lmax0,pmax0,Lmax,pmax,Lmax2,pmax2,Lmax_noc,pmax_noc = pickle.load(handle)

   # Recompute the analytic best fits for the uncorrelated model
   print("Recomputing analytic best fits for uncorrelated model")
   seeds={}
   theta_samples = samples[0][N_regions].T
   for i in range(N_regions):
      theta_MLE = theta_samples[i]
      s_MLE = samples[0][i][:,0] - CMS_b[i] - theta_MLE
      seeds['theta_{0}'.format(i)] = theta_MLE
      seeds['s_{0}'.format(i)] = s_MLE 
 
   #...and get the data into a form that fits model_noc
   # TODO: Would really be helpful if this was not necessary... but is a pretty
   # big change to JMCtools.
   split_theta_samples = [np.array([theta_samples[i]]).T for i in range(N_regions)]
   samples_noc = (samples[0][:N_regions] + split_theta_samples, samples[1])


# Null cannot ever be a better fit than alternate, so apply fix for numerical errors

m = Lmax < Lmax0
Lmax[m] = Lmax0[m]

m2 = Lmax2 < Lmax0
Lmax2[m2] = Lmax0[m2]

# Let's also compute a test statistics assuming there are no correlations
# The data is still produced from the 'real' model, so it will be interesting
# to see how big the mistake is if we test using the uncorrelated model.

# We didn't fit the null hypothesis for this, but we don't have to since we
# know the analytic best fit. We didn't even have to fit the full hypothesis
# actually, that was just a cross-check.

# Can feed all parameters in directly this time since there is no
# multivariate_normal object to screw up the broadcasting
#print("samples_noc:", c.get_data_structure(samples_noc[0]))
# As it turns out, we need to change the parameter array shape
# so that it doesn't auto-broadcast against the data array.
# Data array has shape (Nsamples,1), so we need the parameter
# array to be this same shape.
pars = {**seeds}
pars_null = {**seeds}
for i in range(N_regions):
    pars_null['s_{0}'.format(i)] = np.zeros((Nsamples,1))
    pars['s_{0}'.format(i)] = pars['s_{0}'.format(i)][...,np.newaxis]
    pars['theta_{0}'.format(i)] = pars['theta_{0}'.format(i)][...,np.newaxis]
    pars_null['theta_{0}'.format(i)] = pars_null['theta_{0}'.format(i)][...,np.newaxis]

Lmax0_noc = model_noc.logpdf(pars_null,samples)
Lmax_noc = model_noc.logpdf(pars,samples)

print("shapes:",Lmax_noc.shape, Lmax0_noc.shape)

#print("short way")
#for i in range(10):
#  print("Lmax_noc, Lmax0_noc:",Lmax_noc[i], Lmax0_noc[i])

#Lmax_noc = np.array([model_noc.logpdf({key: val[i] for key,val in seeds.items()},
#                      c.get_data_slice(samples_noc,i)) for i in range(Nsamples)])
#Lmax0_noc = np.array([model_noc.logpdf({key: val[i] for key,val in seeds_null.items()},
#                      c.get_data_slice(samples_noc,i)) for i in range(Nsamples)])
#print("shapes:",Lmax_noc.shape, Lmax0_noc.shape)


#print("long way")
#for i in range(10):
#  print("Lmax_noc, Lmax0_noc:",Lmax_noc[i], Lmax0_noc[i])

  
#print("Ls:",[(a,b,c,d) for a,b,c,d in zip(Lmax0,Lmax,Lmax2,Lmax_noc)])

# Plot test statistic distribution
LLR = -2*(Lmax0 - Lmax)
LLR2 = -2*(Lmax0 - Lmax2)
LLR_noc = -2*(Lmax0_noc - Lmax_noc)

def makeplot(ax, tobin, theoryf, log=True, label="", c='r', obs=None, pval=None):
    #print("tobin:", tobin)
    # get non-nan max and min values
    #m = np.isfinite(tobin)
    #ran = np.min(tobin[m]), np.max(tobin[m])
    ran = (0,25)
    n, bins = np.histogram(tobin, bins=50, normed=True, range=ran)
    print("Histogram y range:", np.min(n[n!=0]),np.max(n))
    q = np.arange(ran[0],ran[1],0.01)
    ax.plot(bins[:-1],n,drawstyle='steps-post',label=label,c=c)
    eta = 0.5
    if theoryf!=None:
        ax.plot(q, theoryf(q),c='k')
    ax.set_xlabel("LLR")
    ax.set_ylabel("pdf(LLR)")
    if log:
        #ax.set_ylim(np.min(n[n!=0]),10*np.max(n))
        ax.set_ylim(1e-3,10*np.max(n))
        ax.set_yscale("log")     
    else:
        pass
        ax.set_ylim(0,1.05*np.max(n))
    if obs!=None:
        # Draw line for observed value, and show p-value region shaded
        qfill = np.arange(obs,ran[1],0.01)
        if theoryf!=None:
           ax.fill_between(qfill, 0, theoryf(qfill), lw=0, facecolor=c, alpha=0.2)
        pval_str = ""
        if pval!=None:
           pval_str = " (p={0:.2g})".format(pval)
        ax.axvline(x=obs,lw=2,c=c,label="Observed ({0}){1}".format(label,pval_str))
    ax.set_xlim(ran[0],ran[1])

# Alright, looks pretty chi-squared.
# I guess that means we can use asymptotic formula for this after all.
# What then is the p-value?
# Let's draw the observed test statistic value on the plot.

observed_data = np.concatenate([np.array(CMS_o),np.zeros(len(CMS_o))],axis=-1)
print("observed_data.shape:",observed_data.shape)
# Need to add the Ntrials and Ndraws axes
observed_data = observed_data[np.newaxis,np.newaxis,:]
Lmax0_obs, pmax0_obs = model.find_MLE_parallel(m0_options,observed_data,method='minuit',Nprocesses=Nproc)
Lmax_obs, pmax_obs   = model.find_MLE_parallel(m_options,observed_data,method='minuit',Nprocesses=Nproc)
seed_obs = {'s_{0}'.format(i): [bf_s[i]] for i in range(N_regions)} # change seeds to fit observed data 
Lmax2_obs, pmax2_obs = model.find_MLE_parallel(m2_options,observed_data,method='minuit',Nprocesses=Nproc,seeds=seed_obs)
LLR_obs = -2 * (Lmax0_obs - Lmax_obs)
pval = 1 - sps.chi2.cdf(LLR_obs[0], 1) 
LLR2_obs = -2 * (Lmax0_obs - Lmax2_obs)
pval2 = 1 - sps.chi2.cdf(LLR2_obs[0], 7) 

# Need analytic best for uncorrelated model
seeds_obs = {}
seeds_obs_null = {}
s_MLE_obs = np.array(CMS_o) - np.array(CMS_b)
for i in range(N_regions):
   seeds_obs['theta_{0}'.format(i)] = 0
   seeds_obs['s_{0}'.format(i)] = s_MLE_obs[i]
   seeds_obs_null['theta_{0}'.format(i)] = 0
   seeds_obs_null['s_{0}'.format(i)] = 0
    
Lmax_obs_noc = model_noc.logpdf(seeds_obs,observed_data)
Lmax0_obs_noc = model_noc.logpdf(seeds_obs_null,observed_data)
LLR_obs_noc = -2 * (Lmax0_obs_noc - Lmax_obs_noc)

print("Lmax_obs_noc", Lmax_obs_noc)
print("Lmax0_obs_noc", Lmax0_obs_noc)

print("LLR_obs    :",LLR_obs)
print("LLR2_obs   :",LLR2_obs)
print("LLR_obs_noc:",LLR_obs_noc)
# Compare to numerical p-values? 
# Get these from (interpolated) empirical CDF of test statistics
def eCDF(x):
  return np.arange(1, len(x)+1)/float(len(x))

CDF1 = spi.interp1d([0]+list(LLR)+[1e99],[0]+list(eCDF(LLR))+[1])
CDF2 = spi.interp1d([0]+list(LLR2)+[1e99],[0]+list(eCDF(LLR2))+[1])
CDF_noc = spi.interp1d([0]+list(LLR_noc)+[1e99],[0]+list(eCDF(LLR_noc))+[1])


print("Lmax0_obs, Lmax_obs, Lmax2_obs, LLR_obs, LLR2_obs:", Lmax0_obs, Lmax_obs, Lmax2_obs, LLR_obs, LLR2_obs)

e_pval  = 1 - CDF1(LLR_obs)
e_pval2 = 1 - CDF2(LLR2_obs)
e_pval_noc = 1 - CDF_noc(LLR_obs_noc)

print("asympototic p-value of observed data:", pval)
print("empirical p-value of observed data  :", e_pval)
print("asymptotic p-value of observed data (2):", pval2)
print("empirical p-value of observed data (2) :", e_pval2)
print("empirical p-value of observed data (noc) :", e_pval_noc)

fig= plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
makeplot(ax, LLR, lambda q: sps.chi2.pdf(q, 1), log=True, label='mu', c='r', obs=LLR_obs, pval=pval)
makeplot(ax, LLR2, lambda q: sps.chi2.pdf(q, 7), log=True, label='free s', c='g', obs=LLR2_obs, pval=pval2)
#makeplot(ax, LLR_noc, None, log=True, label='no_cor', c='b', obs=LLR_obs_noc)

ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
fig.savefig('CMS_2OSLEP_LLR_{0}.png'.format(tag))

