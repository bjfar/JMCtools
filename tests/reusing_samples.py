"""This is an experiment to test whether anything useful can be
   done with samples from a parameter scan of a model, in terms
   of re-weighting them to re-estimate maximum likelihood
   estimators.

   The concept is as follows:

   1. Do global fit of model to some observed data, i.e. via
      MCMC or some such.
   2. Take the maximum likelihood point as null hypothesis
   3. Simulate a bunch of data under this hypothesis
   4. Re-fit the model under that data by re-computing the
      likelihoods of all samples
      (this is useful to do because the relevant physics only
       has to be computed in the first global fit; we just
       change the observed data and recompute the likelihood)
   5. Check if test statistics resemble asymptotic predictions,
      or whether the re-weighted likelihood surfaces are too 
      poorly sampled for this to work.

   Notes: I suspect this might work ok as described, but this is
   only actually useful for goodness-of-fit I think. If we want
   to discover a signal, we need to simulate data under a zero-signal
   null hypothesis, and then I think it is much more likely that
   the reweighted likelihoods will be extremely poorly sampled.
   Certainly the bigger fluctuation in the null data, the worse
   the sampling will be, since big fluctuations look like signals
   and would ideally drive the scan to certain regions of parameter
   space. It might work if the observed data only has 1 or 2 sigma
   significance, but I think the sampling will be too bad for larger
   fluctuations.
   On the other hand, perhaps it is good enough to estimate the
   asymptotic properties of the test statistics? And then we can
   use asymptotic formula to get an approximate significance?
"""
import JMCtools as jt
import JMCtools.distributions as jtd
import JMCtools.models as jtm
import JMCtools.plotting as jtp
import JMCtools.common as c
import pymc as mc
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import concurrent.futures

# Step 1: define model
# --------------------
# Let's use 3 normal distributions, nice and simple
# If this idea doesn't work on this then it won't
# work on anything.
test_model = jtd.JointModel([sps.norm,sps.norm,sps.norm])

# Parameter space setup
def pars1(mu1):
    return {"loc":mu1, "scale":1}

def pars2(mu2):
    return {"loc":mu2, "scale":1}

def pars3(mu3):
    return {"loc":mu3, "scale":1}

parfs = [pars1, pars2, pars3]
parmodel = jtm.ParameterModel(test_model,parfs)

# Define the null hypothesis
null_parameters = {'mu1':0, 'mu2':0, 'mu3':0}

# Step 2: Set observed data
# -------------------------
fit_data = np.array([0,0,0])[np.newaxis,np.newaxis,:]
obs_data = np.array([2,2,2])[np.newaxis,np.newaxis,:] 

# Step 3: Set up PyMC scan
# ------------------------

# Priors
mu1 = mc.Uniform('mu1', lower=-20, upper=20)
mu2 = mc.Uniform('mu2', lower=-20, upper=20)
mu3 = mc.Uniform('mu3', lower=-20, upper=20)

# Log(likelihood) function
@mc.stochastic(observed=True)
def custom_logl(value=obs_data, mu1=mu1, mu2=mu2, mu3=mu3):
    return parmodel.logpdf({'mu1':mu1, 'mu2':mu2, 'mu3':mu3},fit_data) 

M = mc.MCMC([mu1,mu2,mu3,custom_logl])
M.sample(iter=5000)

print("")
print(M.stats()['mu1']['mean'])
print(M.stats()['mu2']['mean'])
print(M.stats()['mu3']['mean'])

# Alrighty, good enough.
# Let's get the chain.
t_mu1 = M.trace('mu1')[:]
t_mu2 = M.trace('mu2')[:]
t_mu3 = M.trace('mu3')[:]

print("Chain length: {0}".format(len(t_mu1)))

# What if we fit the null model instead? We could consider doing a scan
# like this I guess, though it is a bit crazy...


# It's kind of dumb, but PyMC doesn't save a trace of the likelihood
# values. Can do some hacks to make it do so, but let's just manually
# recompute them. We have to recompute them lots of times anyway.

# Step 4: Simulate some data under the null hypothesis
# ----------------------------------------------------

Ntrials = int(1e4)
null_data = parmodel.simulate(Ntrials,null_parameters)
# Have to reshape the parameter arrays so they broadcast against the data
newshape = t_mu1.shape + tuple([1]*(len(null_data.shape)-1))

# We'll do this in a parallelised loop to save RAM
def loopfunc(i,data):
   reweighted_logls = parmodel.logpdf({
                    'mu1':t_mu1.reshape(newshape), 
                    'mu2':t_mu2.reshape(newshape), 
                    'mu3':t_mu3.reshape(newshape)
                    },data)
   
   # Now, find the maximum logl in the samples under each simulated dataset
   Lmax = np.max(reweighted_logls,axis=0)[...,0]
   return i, Lmax

Lmax = np.zeros(Ntrials)
Nprocesses = 3
chunksize = 100
Nchunks = Ntrials // chunksize
Nremainder = Ntrials % chunksize
with concurrent.futures.ProcessPoolExecutor(Nprocesses) as executor:
    for i, Lmax_chunk in executor.map(loopfunc, range(Nchunks), 
                                          jtm.ChunkedData(null_data,chunksize) 
                                      ):
        # Collect results
        print("\r","Getting MLEs for chunk {0} of {1}           ".format(i,Nchunks), end="")
        slicesize = chunksize
        if i==Nchunks and Nremainder!=0:
            slicesize = Nremainder
        start, end = i*chunksize, i*chunksize + slicesize
        Lmax[start:end] = Lmax_chunk
print("Done!")

# And get the original likelihood values, under the observed data, as well.
t_logl = parmodel.logpdf({
                 'mu1':t_mu1.reshape(newshape), 
                 'mu2':t_mu2.reshape(newshape), 
                 'mu3':t_mu3.reshape(newshape)
                 },obs_data)

# Finally, we need the logl of our null hypothesis under each simulated dataset
Lmax0 = parmodel.logpdf(null_parameters,null_data)[...,0]
LLR = -2 * (Lmax0 - Lmax) 

# Also need observed value of test statistic
LLR_obs = -2 * (parmodel.logpdf(null_parameters,obs_data) - np.max(t_logl))

# Compare asymptotic to empirical p-values
epval = c.e_pval(LLR,LLR_obs[0][0])
apval = 1 - sps.chi2.cdf(LLR_obs[0][0], 3)  

print("Empirical p-value :",epval)
print("Asymptotic p-value:",apval)

# Now we can compute our test statistic and plot its distribution!
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
jtp.plot_teststat(ax, LLR, lambda q: sps.chi2.pdf(q, 3), log=True, 
         c='b', obs=LLR_obs, pval=None, title="Reweighted (epval={0:.3}, apval={1:.3})".format(epval,apval))
ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
fig.savefig('reweighting_test.png')

# Hmm, surpisingly good! Not so good if we do the MCMC fit to the observed data though...
