"""Just messing around with some of the objects in this package"""

# Some trickery for relative imports, see: https://stackoverflow.com/a/27876800
if __name__ == '__main__':
    if __package__ is None:
        import sys
        import os
        sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import joint_simtools as jt
        import joint_simtools.distributions as jtd
    else:
        import joint_simtools as jt
        import joint_simtools.distributions as jtd
 
import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np
# My screen is now really high-res so make images bigger by default
plt.rcParams["figure.dpi"] = 3*72

mu_chi = np.arange(0,10,0.5) # Our "computable" parameter
mu_Delta = np.arange(0,10,0.5) # The super-flexible "filler" parameter

scale1 = 2
scales = [scale1]

model1 = jtd.JointModel([sps.norm])
def pars1(mu1):
    return [{"loc":mu1, "scale":1}]

# True parameters
muT_1 = 6
DT_1 = 0 

# Generate samples from null model
N = 10000
model0 = model1(pars1(muT_1+DT_1))
print("Is model0 frozen?", model0.frozen)
samples = model0.rvs((N,))

# See how this varies with the data-generating hypothesis.
DT_1b = 3 
model0b = model1(pars1(muT_1+DT_1b))
samplesb = model0b.rvs((N,))

def getppp(data):
    mu = np.arange(-5,15,0.1) # same range for all parameters
        
# Will do the profiling of each likelihood component separately, since the parameters are independent.
def gauss(x,loc,scale):
    return (x - loc)**2 / scale**2
    
def halfgauss(x,loc,scale):
    out = np.zeros((x+loc).shape)
    m = [(x < loc)]
    out[m] = (x - loc)[m]**2 / scale**2
    return out

# Use these analytically profiled functions to replace those in the model
#model1.set_logpdf([gauss])

def get_q_obs(data,mu_DeltaT_test):
    # Need higher resolution here dues to numerical issues when prof_model1_chi2 is really small
    mu = np.arange(-5,15,0.5) # same range for all parameters
    mu_full = mu[np.newaxis,:] + mu_DeltaT_test[:,np.newaxis]
    H1_chi2_min = np.min(-2*model1.logpdf(data[0][:,np.newaxis,np.newaxis],pars1(mu_full[np.newaxis,:,:])), axis=-1)
    #H1_chi2_min = np.min(gauss(data[0][:,np.newaxis,np.newaxis],mu_full[np.newaxis,:,:],scales[0]),axis=-1) # profile out mu parameter
   
    # No free parameters in null hypothesis now!
    #H0_chi2_min = gauss(data[0][:,np.newaxis],muT[0]+mu_DeltaT_test[np.newaxis,:],scales[0]) 
    H0_chi2_min = -2*model0.logpdf(data[0])
       
    Tval = H0_chi2_min - H1_chi2_min
    
    # Should have 2D array of test statistic values, Nsample * size(mu_DeltaT_test)
    return Tval

# Need some kind of object structure that simplifies finding profile likelihoods...
# Needs to therefore know what parameters we are profiling...
# Or simplest first; we just want global maximum, possibly leaving some parameters fixed
# How can we do this in an elegant way?
# Also want to be able to do it in a vectorised fashion, for many data realisations at once
# Can't just use giant cube of parameters, will overflow memory
# And we also need to make use of the independence of submodels if possible
# Use a mapping from parameters to which submodels require them?
# 
# e.g.
#
# Mapping from "theory parameters" to parameters of statistical distribution functions
# def pars(a1,a2,a3,a4):
#    out = [{"loc": a1+a2, "scale": 1},
#           {"loc": a2+a3, "scale":1},
#           {"loc": a4, "scale":1] 
#    return out
#
# Structure which encodes the dependencies of that mapping
# deps = [("a1","a2"),("a2","a3"),("a4")]
# where the index tells you which distribution function depends on those parameters
#
# So then if we want to maximise we know we can do it in two blocks, the a1,a2,a3 block and the a4 block.
# (where we have to merge blocks with overlapping parameter dependence)
#
# So, I think we want some kind of wrapper class, that contains both a JointModel and also
# this kind of parameter dependence mapping, and can do operations like maximisation with them?

class Block:
    """Record of dependency structure of a block of submodels in ParameterModel"""
    def __init__(self,deps,submodels,jointmodel=None,submodel_deps=None,parfs=None):
       self.deps = frozenset(deps) # parameter dependencies
       self.submodels = frozenset(submodels) # indices of submodels associated with this block 
       self.jointmodel = jointmodel
       self.submodel_deps = submodel_deps # parameter dependencies of each submodel in jointmodel
       self.parfs = parfs # functions to compute jointmodel.pdf arguments from parameters

    @classmethod  
    def fromBlock(cls,block,jointmodel=None,submodel_deps=None,parfs=None):
       deps = block.deps
       submodels = block.submodels
       if (block.jointmodel!=None) and (jointmodel!=None):
          raise ValueError("Tried to set 'jointmodel' for a copy of a Block that already has a 'jointmodel' set!")
       if block.jointmodel!=None:
          jointmodel = block.jointmodel
       return cls(deps,submodels,jointmodel,submodel_deps,parfs)

    # Methods for hashing in sets
    def __repr__(self):
       return "(deps={0}, submodels={1})".format([d for d in self.deps],[i for i in self.submodels])

    def __eq__(self, other):
       return self.deps == other.deps and self.submodels == other.submodels 

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.deps) ^ hash(self.submodels)

    def get_pdf_args(self,parameters):
        """Obtain the argument list required to evaluate the joint
           pdf of a block, from the more abstract parameters known
           to this object"""
       
        # For each submodel, we need to run its 'parf' function
        # But to do that we need to first know what parameters that
        # function needs.
        # Fortunately, this is stored in 'submodel_deps'
        args = []
        for deps,parf in zip(self.submodel_deps,self.parfs):
            pdict = {p: parameters[p] for p in deps} # extract just the parameters that this parf needs
            args += [parf(**pdict)]
        return args       
 
    def logpdf(self,x,parameters):
        args = get_pdf_args(parameters)
        return jointmodel.logpdf(x,**args)


class ParameterModel:     

    def __init__(self,jointmodel,parameter_funcs,x=None):
        """Note: once this object is constructed, it is best if you
           don't mess around with the internals of the stored JointModels
           and so on, or else you might screw up the internal
           consistency of the member routines in this object"""
        self.model = jointmodel
        self.parfs = parameter_funcs
        # Get parameter dependencies of each block
        self.submodel_deps = [f.__code__.co_varnames for f in self.parfs] 
        if x==None:
           self.x = [None for i in range(self.model.N_submodels)]
        else:
           validate_data(x)
           self.x = x
 
        # Compute grouping to use for e.g. MLE search
        # That is, find the minimal "blocks" of submodels that depend
        # on a minimal set of parameters
        # This search is not very efficient, but I don't anticipate there
        # being zillions of parameters. It can be improved if there is some
        # speed problem in the future.

        # Go through each set of dependencies of the submodels,
        # and construct a list of submodels that need to stay joined
        # due to overlapping parameter dependencies
        # Will have to do repeated passes until no more merges are required
        # block list is a list (well, set) of pairs; the submodels in a block, and their parameter dependencies
        # Starts off assuming all submodels form independent blocks. We will merge from there.
        #block_list = set([(frozenset(deps),frozenset([i])) for i,deps in enumerate(self.submodel_deps)])
        print(self.parfs)
        block_list = set([Block(deps,[i]) for i,deps in enumerate(self.submodel_deps)])
        merge_occurred = True
        while merge_occurred:
           merge_occurred = False
           new_block_list = set([])
           for block in block_list:
              block_deps = set([])
              block_submodels = set([])
              for parameter in block.deps:
                 matches = self.find_submodels_which_depend_on(parameter)
                 if len(matches)>0:
                    block_submodels.update(matches)
                    # Add all the parameters on which the newly added submodels depend
                    for i in matches:
                       block_deps.update(self.submodel_deps[i])
              #new_block_list.add((frozenset(block_deps),frozenset(block_submodels)))
              new_block_list.add(Block(block_deps,block_submodels))
           block_list = new_block_list

        # Break down full JointModel into individual JointModels for each block
        # and add these to the Blocks       
        final_block_list = set([])
        for block in block_list:
           smlist = list(block.submodels) # get as a list to ensure iteration order
           block_jointmodel = self.model.split(smlist)
           block_submodel_deps = [self.submodel_deps[i] for i in smlist]
           block_parfs = [self.parfs[i] for i in smlist]  
           final_block_list.add(Block.fromBlock(block,block_jointmodel,block_submodel_deps,block_parfs))

        self.blocks = final_block_list
        print(self.blocks)

    def get_pdf_args(self,parameters):
        """Obtain the argument list required to evaluate the joint
           pdf of a block, from the more abstract parameters known
           to this object"""
       
        # For each submodel, we need to run its 'parf' function
        # But to do that we need to first know what parameters that
        # function needs.
        # Fortunately, this is stored in 'submodel_deps'
        args = []
        for deps,parf in zip(self.submodel_deps,self.parfs):
            pdict = {p: parameters[p] for p in deps} # extract just the parameters that this parf needs
            args += [parf(**pdict)]
        return args
 
    def find_submodels_which_depend_on(self,parameter):
        """Return the indices of the submodels which depend on 'parameter'"""
        matches = set([])
        for i,d in enumerate(self.submodel_deps):
            if parameter in d:
                matches.add(i)
        return matches

    def validate_data(self,x):
        if len(x)!=self.model.N_submodels:
           raise ValueError("The length of the supplied list of data values does not match the number of submodels in the wrapped JointModel! This means that this data cannot possibly be generated by (or used to compute the pdf of) this model and so is invalid.")

    def set_data(self,x):
        """Set internally-stored data realisations to use for parameter 
           estimation etc."""
        validate_data(x)
        self.x = x

    def simulate(self,N,null_parameters=None):
        """Generate N simulated datasets to use for parameter estimation etc.
           null_parameters - parameters to use for the simulation.
             If None, hopes that 'model' is frozen with some pre-set parameters.
        """
        if null_parameters!=None:
           args = self.get_pdf_args(null_parameters)
        else:
           args = {}
        print('args:',args)
        self.x = self.model.rvs(N,args)  

    def find_MLE(self,options,x=None):
        """Find the global maximum likelihood estimate of the model parameters,
           using data as x
           If x is a (list of) arrays, then returns MLEs for each element of that
           array. That is, this function is vectorised with respect to data
           realisations.
           options["ranges"] - dictionary giving ranges to scan in each parameter direction.
           options["N"] - number of parameter points to use in each dimension of grid search
           (TODO: choose algorithm to use for search)

           We search for the MLE by breaking up the problem into independent
           pieces, since we know which submodels of the joint pdf depend
           on which parameters, so we can do the maximisation in several steps.
        """
        if x==None:
           x = self.x # Use pre-generated data if none provided
        else:
           validate_data(x)

        for block in self.blocks:
           block_x = [x[i] for i in block.submodels] # Select data relevant to this submodel
           MLE_pars = find_MLE_for_block(block,options,block_x)


    def find_MLE_for_block(self,block,options,block_x=None):
        """Find MLE for a single block of parameters
           Mostly for internal use
        """
        # Construct some ND cube of parameters and maximise over it?
        # This is the simplest thing that is vectorisable, but will run
        # out of RAM if more than a couple of parameter dimensions needed at once
        # Oh well just try it for now. We can make it fancier later.
        
        # block.deps contains a list of parameters on which this
        # block of submodels depends, e.g. ["p1","p2","p3"]
        # block.submodels contains a list of indices of the submodels
        # which belong to this block
        # The first task, then, is to compute the parameter cube that
        # we want to scan
        N = options["N"]
        ranges = options["ranges"]
        p1d = []
        for par in block.deps:
           p1d = [ np.linspace(*ranges[par],N) ]
        PARS = np.meshgrid(*p1d)
        pdict = {}
        for i,par in enumerate(block.deps):
           pdict[par] = PARS[i]

        block_logpdf = block.logpdf(block_x,parameters[...,np.newaxis])

        print(block_logpdf.shape)


        # Do the mapping from parameter space to the arguments of
        # each submodel of the jointmodel 
#        submodel_args = [
#
#        # Do minimization on the above grid
#        H1_chi2_min = np.min(block.jointmodel.logpdf(block_x,
#
#    H1_chi2_min = np.min(gauss(data[0][:,np.newaxis],mu[np.newaxis,:],scales[0]),axis=-1) # profile out mu_x parameter
#
#    for x,s in zip(data[1:],scales[1:]):
#        H1_chi2_min += np.min(halfgauss(x[:,np.newaxis],mu[np.newaxis,:],s),axis=-1) # profile out each half-gaussian parameter
   



       # Actually rather than interpolate, let's just find the minima for now
#        H1_chi2_min = np.min(np.min(H1_chi2_all,axis=0),axis=0)


        
# Test ParameterModel constructor

model1 = jtd.JointModel([sps.norm])
def pars1(mu1):
    return [{"loc":mu1, "scale":1}]

parmodel = ParameterModel(model1,[pars1])


model2 = jtd.JointModel([sps.norm,sps.norm,sps.norm])
def pars2_A(mu1,mu2):
    return {"loc":mu1+mu2, "scale":1}

def pars2_B(mu2,mu3):
    return {"loc":mu2+mu3, "scale":1}

def pars2_C(mu4):
    return {"loc":mu4, "scale":1}

# Can infer parameter dependencies of blocks from this list of functions
parfs = [pars2_A, pars2_B, pars2_C]

parmodel2 = ParameterModel(model2,parfs)

null_parameters = {'mu1':0, 'mu2':0, 'mu3':0, 'mu4':0}
parmodel2.simulate(100,null_parameters)
        
quit()

# Now, we need the test statistic values under all the values of mu_deltaT 
# that have non-zero posterior probability.
mu_DeltaT_test = np.arange(0,15,1)

qobs = get_q_obs([samples],mu_DeltaT_test)
qobs2 = get_q_obs([samples2],mu_DeltaT_test)

# Now we use these to compute p-values and take the average over the conditonal posteriors

pvals = 1-sps.chi2.cdf(qobs,1)
pvals2 = 1-sps.chi2.cdf(qobs2,1)

mu_DeltaT_post = make_model0(mu_DeltaT_test[np.newaxis,:]).pdf(samples[:,np.newaxis])
mu_DeltaT_post2 = make_model0(mu_DeltaT_test[np.newaxis,:]).pdf(samples2[:,np.newaxis])

print(qobs.shape)
print(mu_DeltaT_post.shape)

pppvals = np.sum(pvals * mu_DeltaT_post * 1,axis=-1) #don't forget volume element, 1 in this case!
pppvals2 = np.sum(pvals2 * mu_DeltaT_post2 * 1,axis=-1)

# Let's compare these to the p-values we would compute if we actually knew mu_DeltaT
# These are just what you get for a delta function posterior
pvals_0 = pvals[:,0]
pvals_5 = pvals2[:,3]

n, bins = np.histogram(pppvals, bins=30, normed=True)
n2, bins = np.histogram(pppvals2, bins=bins, normed=True)
n0, bins = np.histogram(pvals_0, bins=bins, normed=True)
n5, bins = np.histogram(pvals_5, bins=bins, normed=True)
#x = np.arange(0,10,0.01)
fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',c='r',label="mu_DeltaT={0}".format(mu_DeltaT))
ax.plot(bins[:-1],n2,drawstyle='steps-post',c='b',label="mu_DeltaT={0}".format(mu_DeltaT2))
ax.plot(bins[:-1],n0,drawstyle='steps-post',c='g',label="FIXED mu_DeltaT={0}".format(mu_DeltaT_test[0]))
ax.plot(bins[:-1],n5,drawstyle='steps-post',c='m',label="FIXED mu_DeltaT={0}".format(mu_DeltaT_test[3]))
ax.set_xlabel("pppval")
ax.set_ylabel("pdf(pppval)")
plt.legend(loc=2, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
#ax.set_ylim(0,1)

logbins = np.logspace(-5,1,30)
n, bins = np.histogram(pppvals, bins=logbins, normed=True)
n2, bins = np.histogram(pppvals2, bins=bins, normed=True)
n0, bins = np.histogram(pvals_0, bins=bins, normed=True)
n5, bins = np.histogram(pvals_5, bins=bins, normed=True)
#x = np.arange(0,10,0.01)
fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',c='r',label="mu_DeltaT={0}".format(mu_DeltaT))
ax.plot(bins[:-1],n2,drawstyle='steps-post',c='b',label="mu_DeltaT={0}".format(mu_DeltaT2))
ax.plot(bins[:-1],n0,drawstyle='steps-post',c='g',label="FIXED mu_DeltaT={0}".format(mu_DeltaT_test[0]))
ax.plot(bins[:-1],n5,drawstyle='steps-post',c='m',label="FIXED mu_DeltaT={0}".format(mu_DeltaT_test[3]))
ax.set_xlabel("pppval")
ax.set_ylabel("pdf(pppval)")
ax.set_xscale("log")
ax.set_yscale("log")
plt.legend(loc=2, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
#ax.set_ylim(0,1)
plt.show()

# Ok now select samples where it is obvious that mu_DeltaT needs to be zero.
m = (samples - mu_l1T < -2*scale1) #i.e. 3 sigma below the mean. Way below predicted WIMP component
                                    # so should be clear that mu_DeltaT=0 is best
m2 = (samples2 - mu_l1T < -2*scale1) #i.e. 3 sigma below the mean. Way below predicted WIMP component
                                    # so should be clear that mu_DeltaT=0 is best

logbins = np.logspace(-5,1,30)
n, bins = np.histogram(pppvals[m], bins=logbins, normed=True)
n2, bins = np.histogram(pppvals2[m2], bins=bins, normed=True)
n0, bins = np.histogram(pvals_0[m], bins=bins, normed=True)
n5, bins = np.histogram(pvals_5[m2], bins=bins, normed=True)
#x = np.arange(0,10,0.01)
fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(bins[:-1],n,drawstyle='steps-post',c='r',label="mu_DeltaT={0}".format(mu_DeltaT))
ax.plot(bins[:-1],n2,drawstyle='steps-post',c='b',label="mu_DeltaT={0}".format(mu_DeltaT2))
ax.plot(bins[:-1],n0,drawstyle='steps-post',c='g',label="FIXED mu_DeltaT={0}".format(mu_DeltaT_test[0]))
ax.plot(bins[:-1],n5,drawstyle='steps-post',c='m',label="FIXED mu_DeltaT={0}".format(mu_DeltaT_test[3]))
ax.set_xlabel("pppval")
ax.set_ylabel("pdf(pppval)")
ax.set_xscale("log")
ax.set_yscale("log")
plt.legend(loc=2, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
#ax.set_ylim(0,1)
plt.show()

# Check that the posterior looks like we expect...
fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(mu_DeltaT_test,mu_DeltaT_post[m,:][0,:],drawstyle='steps-post',c='r')
ax.set_xlabel("mu_DeltaT")
ax.set_ylabel("posterior")
plt.legend(loc=2, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
