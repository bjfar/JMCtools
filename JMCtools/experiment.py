"""Classes related to defining what an 'experiment' needs to provide in order to be
   analysed in the jpvc framework"""

import numpy as np
import copy
import JMCtools.distributions as jtd
import JMCtools.models as jtm
import JMCtools.common as c
import scipy.stats as sps
from functools import partial
import matplotlib.pyplot as plt
import six
import sys

def diag_MLEs(e, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples, tag):
    """Very simple diagnositic function which simply plots the
       distributions of the MLEs for all parameters under both the 
       full and hypothesis. Should work for any Experiment."""
    Npars = len(pmax.keys())

    fig = plt.figure(figsize=(3*Npars,3))

    for i, (par, vals) in enumerate(pmax.items()):
        pos = i+1
        vals = np.array(vals)
        vals = vals[np.isfinite(vals)] # remove nans from failed fits
        n, bins = np.histogram(vals, normed=True)
        ax = fig.add_subplot(1,Npars,pos)
        ax.plot(bins[:-1],n,drawstyle='steps-post')
        ax.set_title(par)
    plt.tight_layout()
    fig.savefig('{0}_diagnostic_MLEs_full_{1}.png'.format(e.name,tag))
    plt.close(fig)

    # Now the null MLEs
    Npars0 = len(pmax0.keys())

    fig = plt.figure(figsize=(3*Npars0,3))

    for i, (par, vals) in enumerate(pmax0.items()):
        pos = i+1
        vals = np.array(vals)
        vals = vals[np.isfinite(vals)] # remove nans from failed fits
        n, bins = np.histogram(vals, normed=True)
        ax = fig.add_subplot(1,Npars0,pos)
        ax.plot(bins[:-1],n,drawstyle='steps-post')
        ax.set_title(par)
    plt.tight_layout()
    fig.savefig('{0}_diagnostic_MLEs_null_{1}.png'.format(e.name,tag))
    plt.close(fig)


class Test:
    """Defines information required for doing fits (i.e. finding MLEs) for a
       particular kind of statisical test. Mostly just a box for fit
       options, plus some extra bits and pieces."""
    def __init__(self,null_options,
                      full_options,
                      null_seeds,
                      full_seeds,
                      DOF,
                      diagnostics=None,
                      tag=""):
        if diagnostics is None:
            diagnostics = []
        self.null_options      = null_options
        self.full_options      = full_options
        self.null_seeds        = null_seeds
        self.full_seeds        = full_seeds
        self.DOF               = DOF
        try:
            self.diagnostics = [partial(diag_MLEs,tag=tag)] + diagnostics
        except TypeError:
            six.reraise(TypeError,TypeError("Failed to construct Test with tag {0}. Error while concatenating diagnostic function list (the experiment may not have supplied a list? saw diagnostic={1})".format(tag,diagnostics))) 

# Parameter transformation function for scaling signal parameters with 'mu'
def mu_parameter_mapping(mu,scale_with_mu,**kwargs):
    """Parameters that are 'replaced' by mu should be fixed to
       some nominal values using functools.partial, as should
       the list 'scale_with_mu'""" 
    out_pars = {}
    for parname, parval in kwargs.items():
        if parname in scale_with_mu:
            out_pars[parname] = mu * parval
        else:
            out_pars[parname] = parval
    return out_pars
 
class Experiment:
    def __init__(self,name,joint_pdf,observed,DOF,Nproc=3):
        """Basic information:
           name
           joint pdf
           observed data
           degrees of freedom (of general model, i.e. used in gof test)
	   Nproc - Number of mpi processes to use for parallelising repeated fits
        """
        self.name = name
        self.joint_pdf = joint_pdf
        oshape = np.array(observed).shape
        if len(oshape)==1:
           self.observed_data = np.array(observed)[np.newaxis,np.newaxis,:]
        elif len(oshape)==3 and oshape[:2]==(1,1):
           self.observed_data = observed # correct shape already
        else:
           raise ValueError("Shape problem with observed data supplied to experiment {0}. Shape was {1}, but should be either 1D (Ncomponents) or 3D (Ntrials=1, Ndraws=1, Ncomponents)".format(name,oshape))
        self.DOF = DOF
        self.tests = {}
        self.general_model = jtm.ParameterModel(self.joint_pdf)
        self.Nproc = Nproc

    @classmethod  
    def fromExperimentList(cls,experiments,name="Monster",common_pars=[]):
        """Create a 'super' experiment by merging together data from a list
           of Experiments"""

        # Create new 'general' ParameterModel combining 
        # all submodels of all experiments
        #print("experiments:", experiments)
        parmodels = []
        for e in experiments:
            #print("{0} blocks: ".format(e.name), e.general_model.blocks)
            parmodels += [e.general_model]
        joint_dist, renaming = cls.create_jointmodel(parmodels,common_pars)

        # Join observed data along last axis
        obs_data_list = []
        for e in experiments:
            obs_data_list += [e.observed_data]
        #observed_data = np.concatenate(
        #         [o.reshape(1,1,-1) for o in obs_data_list],axis=-1) 
        observed_data = np.concatenate([o.reshape(1,1,-1) for o in obs_data_list],axis=-1) 
   
        mergedE = Experiment(name,joint_dist,observed_data,DOF=1) # Need to overwrite this DOF once we compute what it is supposed to be
        #print("mergedE.general_model.blocks:", mergedE.general_model.blocks)

        # Merge together generic information for all tests
        for t in ['mu', 'gof']:
            null_opt = {}
            full_opt = {}
            null_seeds_fs = []
            full_seeds_fs = []
            DOF = 0
            dims = []
            scale_with_mu = []
            #print(experiments)
            #print(renaming)
            for e, rename in zip(experiments,renaming):
                try:
                    test = e.tests[t]
                except KeyError:
                    # This type of test is undefined for this experiment; skip it
                    continue 
                null_opt.update(cls.rename_options(test.null_options,rename))
                full_opt.update(cls.rename_options(test.full_options,rename))
                null_seeds_fs += [test.null_seeds]
                full_seeds_fs += [test.full_seeds]
                DOF += test.DOF
                dims  += [np.sum(e.general_model.model.dims)]
                #print(e.name, e.general_model.model.dims, dims)
                if t is 'mu':
                    # Slightly hacky since my parameter renaming function operates on dictionaries...
                    swm_dict = {par: None for par in test.scale_with_mu}
                    renamed_swm = cls.rename_options(swm_dict,rename)
                    scale_with_mu += renamed_swm.keys()
                    # Consolidate the test signal
                    #test_signal.update(cls.rename_options(test.test_signal,rename)) 
         
            #print("options:")
            #print("null_opt:", null_opt)
            #print("full_opt:", full_opt)
            if t is 'gof':
                nseedfs, nexact = cls.get_seeds_prep(null_seeds_fs)
                fseedfs, fexact = cls.get_seeds_prep(full_seeds_fs)
                mergedE.DOF = DOF
                mergedE.define_gof_test(null_opt,full_opt,
                                     (partial(cls.get_seeds,dims=dims,get_seed_fs=nseedfs,renaming=renaming), nexact),
                                     (partial(cls.get_seeds,dims=dims,get_seed_fs=fseedfs,renaming=renaming), fexact), 
                                     )
            if t is 'mu':
                nseedfs, nexact = cls.get_seeds_prep(null_seeds_fs)
                mergedE.define_mu_test(null_opt,
                                    (partial(cls.get_seeds,dims=dims,get_seed_fs=nseedfs,renaming=renaming), nexact),
                                    scale_with_mu,
                                    diagnostics=[cls.make_mu_diag()])
        return mergedE 

    @classmethod
    def make_mu_diag(cls):
        """Diagnostic function for mu test"""
        def dmu(e, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples):
            # Plot distribution of fit values. E.g. if 'mu' isn't
            # roughly Gaussian then we don't expect the asymptotic
            # distribution of the test statistics to be followed either.
            fig = plt.figure(figsize=(4,3))
            pos = 1
            val = pmax['mu'] # Best fit value of 'mu' in each pseudoexperiment
            val = val[np.isfinite(val)] # remove nans from failed fits
            #val = val[val>=0] # remove non-positive MLEs, these can't be log'd 
            #print(key, val)
            n, bins = np.histogram(val, bins=20, normed=True)
            ax = fig.add_subplot(1,1,pos)
            ax.plot(bins[:-1],n,drawstyle='steps-post',label="mu")
            ax.set_title('mu')
            plt.tight_layout()
            fig.savefig('{0}_diagnostic_mu.png'.format(e.name))
            plt.close(fig)
        return dmu

    @classmethod
    def rename(cls,key,rename_dict):
        newkey = False
        if key in rename_dict.keys():
            newkey = rename_dict[key]
        return newkey

    # Helper functions for fromExperimentList constructor
    @classmethod
    def rename_options(cls,options,renaming):
        #print("renaming:", renaming)
        # Work out renaming
        rename_dict = {}
        for instruction in renaming:
            new,old = instruction.split(' -> ')
            rename_dict[old] = new
        #print("renaming dict:", rename_dict)
        # Collect and rename options/parameters
        # This is tricky in general since options could have
        # weird names depending on parameters. So for now this
        # only works for the Minuit options.
        # We replace 'par' in the following:
        # 'par'
        # 'error_par'
        # 'fix_par'
        new_opts = {}
        for key,val in options.items():
            newkey = cls.rename(key,rename_dict)
            if not newkey:
                words = key.split("error_")
                newkey = cls.rename(words[0],rename_dict)
                if newkey:
                    newkey = 'error_'+newkey
            if not newkey:
                words = key.split("fix_")
                newkey = cls.rename(words[0],rename_dict)
                if newkey:
                    newkey = 'fix_'+newkey
            if not newkey:
                newkey = key # No substitution found, key unchanged
            new_opts[newkey] = val
            #print("Renamed {0} to {1}".format(key, newkey))
        return new_opts 

    @classmethod
    def create_jointmodel(cls,parmodels,common_pars=[]):
        """Create a single giant JointDist out of a list of
        ParameterModels"""
        #print("In create_joint_model")
        #print("parmodels:", parmodels)
        all_submodels = []
        all_dims = []
        all_renaming = []
        for i,m in enumerate(parmodels):
            # Collect submodels and perform parameter renaming to avoid
            # collisions, except where parameters are explicitly set
            # as being common.
            all_renaming += [[]]
            for submodel in m.model.submodels:
                temp = jtd.TransDist(submodel) # Need this to figure out parameter names
                renaming = ['Exp{0}_{1} -> {1}'.format(i,par) for par in temp.args if par not in common_pars] 
                #print(renaming, temp.args, common_pars)
                all_renaming[i] += renaming  
                all_submodels += [jtd.TransDist(submodel,renaming_map=renaming)]
            all_dims += m.model.dims
            #print("m:",m)
            #print("all_dims", m.model.dims, all_dims)
        new_joint = jtd.JointDist(list(zip(all_submodels,all_dims)))
        return new_joint, all_renaming
    
    def make_mu_model(self,slist): 
        """Create giant ParameterModel for a signal hypothesis 's'
           Might have to re-think this to more easily handle various
           types of signals, especially e.g. signals for unbinned
           likelihoods."""
        jointmodels = []
        for s,f in zip(slist,self.make_mu_model_fs):
            jointmodels += [f(s)]
        return self.create_jointmodel(jointmodels,common_pars=['mu'])[0] 
  
    @classmethod
    def get_seeds_prep(cls,get_seed_fs):
        """Preparation work for construction of joined seed function
           Checks if all the seeds are considered exact, and separates these
           flags from the actual seed functions."""
        all_exact = True
        all_funcs = []
        for seed_func in get_seed_fs:
           try:
              seeds, exact = seed_func
           except TypeError:
              seeds, exact = seed_func, False # Not exact if no flag claiming otherwise.
           all_exact = all_exact and exact # Can only claim seeds are exact if they are ALL exact
           all_funcs += [seeds]
        return all_funcs, exact

    @classmethod  
    def get_seeds(cls,samples,signal,dims,get_seed_fs,renaming):
        #print("samples:",samples)
        #print("samples.shape:",samples.shape)
        #print("dims:",dims)
        datalist = c.split_data(samples,dims)
        #print("datalist.shapes:",[d.shape for d in datalist])
        seeds = {}
        for data, seedf, rename in zip(datalist,get_seed_fs, renaming):
           seeds.update(cls.rename_options(seedf(data,signal),rename)) # Make sure to rename seeds to match renamed parameters!
        return seeds

    # end fromExperimentList helper functions

    def define_gof_test(self,null_options,full_options,null_seeds,full_seeds,diagnostics=None):
        """Set information related to 'goodness of fit' test.
           Required:
             Nuisance parameter default (null hypothesis) values.
             Options for null hypothesis fit (nuisance-only fit)
             Options for full fit (all free parameters)
             Function to produce good starting guesses for null fit (nuisance parameters)
             Function to produce good starting guesses for fit for all parameters
        """
        try:
            self.tests['gof'] = Test(
                                 null_options,
                                 full_options,
                                 null_seeds,
                                 full_seeds,
                                 DOF=self.DOF,
                                 diagnostics=diagnostics,
                                 tag='gof')
        except TypeError:
            six.reraise(TypeError,TypeError("Error constructing 'gof' test for experiment '{0}'. Input parameters were:\n\
   null_options={1}\n\
   full_options={2}\n\
   null_seeds={3}\n\
   full_seeds={4}\n\
   DOF={5}\n\
   diagnostics={6}\n\
   tag={7}\n\
".format(self.name,
         null_options,
         full_options,
         null_seeds,
         full_seeds,
         self.DOF,
         diagnostics,
         'gof')))
        
    def define_mu_test(self,null_options,null_seeds,scale_with_mu,diagnostics=None):
        """Set information related to testing via 'mu' signal strength
           parameter. This requires a bit different information that the 'gof'
           case since we need to know what parameters to scale using 'mu', but don't
           need options for the 'full' fit (since we can work them out here).
        """

        null_opt = {**null_options, 'mu': 0, 'fix_mu': True}
        full_opt = {**null_options, 'mu': 1, 'fix_mu': False, 'error_mu': 0.1}

        try:
           nseeds, nexact = null_seeds
        except TypeError:
           nseeds, nexact = null_seeds, False
        # The seeds are never exact for the mu test. At least not yet.

        self.tests['mu'] = Test(
                                null_opt,
                                full_opt,
                                (nseeds,False),
                                (nseeds,False),
                                DOF=1,
                                diagnostics=diagnostics,
                                tag='mu')
        self.tests['mu'].scale_with_mu = scale_with_mu # List of parameter to scale with mu

    def define_musb_test(self,null_options,mu1_seeds,mu0_seeds,scale_with_mu,asimov,diagnostics=None):
        """Set information related to testing via 'mu' signal strength
           parameter, (s+b vs b-only version)
           This is much the same as mu_test, however instead of testing mu_BF 
           vs mu=0, we test mu=1 vs mu=0.
           There is therefore no fitting of mu required, we only have to
           fit the nuisance parameters.
           asimov - function that computes asimov data for a given input mu and signal shape.
        """
        if asimov is None:
            raise ValueError("Received NoneType for 'asimov' input. This should be a function\
 that calculates the asimov data for this experiment, for a given signal hypothesis")

        # The naming here is slightly confusing for the 'musb' test, since we use the
        # same LLR regardless of whether mu=0 or mu=1 is the null hypothesis (we can
        # work out the asymptotic distribution under either)
        # But to match arxiv:1007.1727 we want mu=1 in the numerator and mu=0 in the
        # denominator of the likelihood ratio, which corresponds to the choice below.
        null_opt = {**null_options, 'mu': 1, 'fix_mu': True} # numerator
        full_opt = {**null_options, 'mu': 0, 'fix_mu': True} # denominator

        try:
           nseeds, nexact = mu1_seeds
        except TypeError:
           nseeds, nexact = mu1_seeds, False
        try:
           fseeds, fexact = mu0_seeds
        except TypeError:
           fseeds, fexact = mu0_seeds, False
 
        self.tests['musb'] = Test(
                                null_opt,
                                full_opt,
                                (nseeds,nexact),
                                (fseeds,fexact),
                                DOF=0,
                                diagnostics=diagnostics,
                                tag='musb')
        self.tests['musb'].scale_with_mu = scale_with_mu # List of parameter to scale with mu
        self.tests['musb'].asimov = asimov

    def make_mu_model(self,signal):
        """Create ParameterModel object for fitting with mu_test"""
        if not 'mu' in self.tests.keys():
            raise ValueError("Options for 'mu' test have not been defined for experiment {0}!".format(self.name))

        # Currently we cannot apply the transform func directly to the JointDist object,
        # so we have to pull it apart, apply the transformation to eah submodel, and then
        # put it all back together.
        transformed_submodels = []
        for submodel, dim in zip(self.joint_pdf.submodels, self.joint_pdf.dims):
            args = c.get_dist_args(submodel)
            # Pull out the arguments that aren't getting scaled by mu, and replace them with mu.
            new_args = [a for a in args if a not in self.tests['mu'].scale_with_mu] + ['mu']
            # Pull out the arguments that ARE scaled by mu; we only need to provide these ones,
            # the other signal arguments are for some other submodel.
            sig_args = [a for a in args if a in self.tests['mu'].scale_with_mu]
            my_signal = {a: signal[a] for a in sig_args} # extract subset of signal that applies to this submodel
            transform_func = partial(mu_parameter_mapping,scale_with_mu=self.tests['mu'].scale_with_mu,**my_signal)
            trans_submodel = jtd.TransDist(submodel,transform_func,func_args=new_args)
            #print('in make_mu_model:', trans_submodel.args)
            transformed_submodels += [(trans_submodel,dim)]
        #print("new_submodels:", transformed_submodels)
        new_joint = jtd.JointDist(transformed_submodels)
        return jtm.ParameterModel(new_joint)

    @classmethod
    def chi2_pval(cls,LLR,LLR_obs,DOF):
        """Compute p-value for chi2 test using pre-generated LLR samples
           (or just asymptotically if LLR is None"""
 
        #print("LLR_obs (chi2_pval):",LLR_obs)

        # Asymptotic p-value
        apval = np.atleast_1d(1 - sps.chi2.cdf(LLR_obs, DOF)) 

        if np.all(np.isnan(apval)):
           apval = None # bit easier to work with

        # Empirical p-value
        if LLR is not None:
           epval = c.e_pval(LLR,LLR_obs)
        else:
           epval = None
        return apval, epval


    def do_gof_test(self,test_parameters,samples=None,observed=None):
        model = self.general_model
        # Test parameters fix the hypothesis that we are
        # testing. I.e. they fix some parameters during
        # the 'null' portion of the parameter fitting.
        extra_null_opt = {**test_parameters}
        for key in test_parameters.keys():
            extra_null_opt["fix_{0}".format(key)] = True # fix these parameters
        LLR, LLR_obs = self.get_LLR_all(model,'gof',samples,extra_null_opt,signal=test_parameters,observed=observed)
        # p-values
        DOF = self.tests['gof'].DOF
        apval, epval = self.chi2_pval(LLR,LLR_obs,DOF)

        return model, LLR, LLR_obs, apval, epval, DOF

    def do_mu_test(self,nominal_signal,samples=None):
        model = self.make_mu_model(nominal_signal)
        #print("Blocks: ", model.blocks)
        LLR, LLR_obs = self.get_LLR_all(model,'mu',samples)

        # p-values
        DOF = self.tests['mu'].DOF
        apval, epval = self.chi2_pval(LLR,LLR_obs,DOF) 
      
        return model, LLR, LLR_obs, apval, epval, DOF

    @classmethod
    def sb_pval(cls,LLR,LLR_obs,LLRA,nullmu):
        """Compute p-value for s+b/b test ('musb'), with asymptotics
           as described in e.g. Eq. 73, 74 in arXiv:1007.1727"""
        # Note, LLR is actually q = -2 * log(L1/L0)
        # Asymptotic distribution of q is a Gaussian with mean and
        # variance computed from LLRA as follows (Eq. 73, 74 in 1007.1727)
        # Need to multiply by a sign depending on hypotheis
        if nullmu is 0:
           sign=1
        else:
           sign=-1
        #var_mu = sign * 1. / LLRA
        Eq = LLRA
        Varq = sign * 4 * LLRA
        # Now we have the parameters, can compute p-value
        # (trying to exclude background-only hypothesis)
        #print("LLR_obs (sb_pval):",LLR_obs)
        #print("LLRA (sb_pval):",LLRA)
        apval = np.atleast_1d(sps.norm.cdf((LLR_obs - Eq) / np.sqrt(Varq))) # Eq. 76, 1007.1727
        if nullmu is not 0:
           apval = 1 - apval # integration goes the other way in signal+background case

        if np.all(np.isnan(apval)):
           apval = None # bit easier to work with

        #print("LLR_obs:", LLR_obs)
        #print("apval:", apval)

        # Empirical p-value
        # Signal-like direction is negative, therefore we want to integrate from -infty to LLR_obs for p-value
        # in mu=0 case (probability of more signal-like test statistic than was observed)
        # This is the opposite of the chi-squared case, thus we want to reverse the
        # ordering of the sorted LLR sequence in the empirical CDF calculation
        # (but if we are testing mu=1 then ordering is as in chi^2 case!)
        if LLR is not None:
            if nullmu is 0:
                epval = c.e_pval(LLR,LLR_obs,reverse=True)
            else:
                epval = c.e_pval(LLR,LLR_obs)
        else:
            epval = None
        return apval, epval, Eq, Varq


    def do_musb_test(self,nominal_signal,samples=None,nullmu=0,observed=None):
        """By default, assume the null hypothesis is that mu=0, i.e. we are trying
           to exclude the background-only hypothesis. For trying to exclude the
           nominal signal hypothesis, need to set nullmu=1.
           If observed=None will use compute p-values for observed data as defined
           in the Experiment. If 'observed' is supplied then p-values will be computed
           for that data instead"""
        model = self.make_mu_model(nominal_signal)
        #print("Blocks: ", model.blocks)
        LLR, LLR_obs = self.get_LLR_all(model,'musb',samples,signal=nominal_signal,LLR_must_be_positive=False,observed=observed)
  
        # Determine asymptotic distribution from Asimov data (see e.g. arXiv:1007.1727)
        # Also gets MLEs for nuisance parameters under Asimov data
        nA, nuis_MLEs = self.tests['musb'].asimov(mu=nullmu,signal=nominal_signal) # Get asimov data for background-only hypothesis
        #print("nA:", nA)
        # Evaluate test statistic under asimov data
        nA = np.atleast_1d(nA)[np.newaxis,np.newaxis,:]
        #print("nA:", nA)
        #for key in nuis_MLEs:
        #   nuis_MLEs[key] = np.atleast_1d(nuis_MLEs[key]) # Fix up shape
        #Lmax0, pmax0, Lmax, pmax = self.get_LLR(model,'musb',nA,nuis_MLEs,nuis_MLEs)
        #LLRA = -2*(Lmax0 - Lmax)
        print("Fitting with Asimov data",file=sys.stderr)
        LLRA, LLRA_obs = self.get_LLR_all(model,'musb',nA,signal=nominal_signal,LLR_must_be_positive=False,observed=observed)
 
        apval, epval, Eq, Varq = self.sb_pval(LLR,LLR_obs,LLRA,nullmu)
 
        #print("nA:", nA)
        #print("LLR_obs:", LLR_obs)
        #print("LLRA:", LLRA)
        #print("apval:", apval)
        #print("epval:", epval)
        #print("Eq:", Eq)
        #print("Varq:", Varq)

        return model, LLR, LLR_obs, apval, epval, LLRA, Eq, Varq
 
    def compute_LL(self,model,samples,fixed_pars,opt,seeds,exact=False,Nproc=3):
        """Compute log-likelihood for input parameters/options/samples"""
        test_mode = False # Compute both numerical and exact results, for comparison. Only does anything if exact=True
        if exact:
            print("Seeds are exact MLEs; skipping minimisation",file=sys.stderr)
            pmax_exact = seeds
            pmax_exact.update(fixed_pars)
            # need to loop over parameters, otherwise it will automatically evaluate
            # every set of parameters for every set of samples. We need them done
            # in lock-step.
            Nsamples = samples.shape[0]
            Lmax_exact = np.zeros(Nsamples)
            for i,X in enumerate(samples):
                if i % 50 == 0:
                    print("\r","Processed {0} of {1} samples...           ".format(i,Nsamples), end="", file=sys.stderr)
                pars = {}
                for par, val in pmax_exact.items():
                    try:
                        pars[par] = val[i]
                    except (TypeError, IndexError):
                        pars[par] = val # Fixed parameters aren't arrays
                #print()
                #print("seeds:",seeds)
                Lmax_exact[i] = model.logpdf(pars,X)
                if ~np.isfinite(Lmax_exact[i]): 
                   print("NaN detected!")
                   print("pars:",pars)
                   print("Lmax_exact[i]:", Lmax_exact[i])
                   print("X:",X)
                   raise ValueError("NaN logpdf detected!") 
            print()
        if not exact or test_mode:
            Lmax_fit, pmax_fit = model.find_MLE_parallel(opt,samples,method='minuit',
                                               Nprocesses=Nproc,seeds=seeds)
        if exact and test_mode:
            #print("pmax_fit:", pmax_fit)
            #print("pmax_exact:", pmax_exact)
            for i in range(Nsamples):
                print("Exact vs Fit for sample {0}".format(i))
                print("   Lmax_exact: {0}".format(Lmax_exact[i]))
                print("   Lmax_fit  : {0}".format(Lmax_fit[i]))
                for key in pmax_fit.keys():
                    if key in fixed_pars.keys():
                        print("   {0} fixed to {1}".format(key,fixed_pars[key]))
                    else:
                        print("   {0} exact: {1}".format(key, pmax_exact[key][i])) 
                        print("   {0} fit  : {1}".format(key, pmax_fit[key][i]))

        # This is after all the testing stuff to ensure we don't accidentally set them equal via reference voodoo
        if exact:
            Lmax = Lmax_exact
            pmax = pmax_exact
        else:
            Lmax = Lmax_fit
            pmax = pmax_fit                    
        return Lmax, pmax

    def get_LLR(self,model,test,observed_data,seeds0,seeds,nexact=True,fexact=True,extra_null_opt={},signal={},Nproc=3):
        """Get log-likelihood ratio for selected test, using supplied data and non-fixed parameters"""
        # Get options for fitting routines
        null_opt = self.tests[test].null_options
        if extra_null_opt: null_opt.update(extra_null_opt)
        full_opt = self.tests[test].full_options

        # Extract fixed parameter values from options,
        # or from signal parameters specified as fixed by options
        null_fixed_pars = {}
        for opt in null_opt:
            words = opt.split("fix_")
            if words[0] == opt:
                continue # option is not fix_something, skip it
            else:
                parname = words[1]
                # Search for a value to fix this parameter to
                # Go first with parameters supplied by user, then
                # revert to defaults set in experiment setup options
                null_fixed_pars[parname] = c.search_dicts(parname,signal,null_opt)

        full_fixed_pars = {}
        for opt in full_opt:
            words = opt.split("fix_")
            if words[0] == opt:
                continue # option is not fix_something, skip it
            else:
                parname = words[1]
                full_fixed_pars[parname] = c.search_dicts(parname,signal,full_opt)

        # Manually force numerical minimization for testing
        #fexact, nexact = False, False
        print("Fitting null hypothesis...",file=sys.stderr)
        Lmax0, pmax0 = self.compute_LL(model,observed_data,null_fixed_pars,null_opt,seeds0,nexact,Nproc)
        print("Fitting alternate hypothesis...",file=sys.stderr)
        Lmax,  pmax  = self.compute_LL(model,observed_data,full_fixed_pars,full_opt,seeds,fexact,Nproc)
  
        return Lmax0, pmax0, Lmax, pmax

    def get_LLR_all(self,model,test,samples=None,extra_null_opt={},signal=None,LLR_must_be_positive=True,observed=None):
        """Get log-likelihood ratio for selected statistical test
           Uses seed functions defined as part of test to help fit (or directly compute) MLEs for parameters.
           Therefore needs 'signal' parameters to give to this seed function."""

        print("Fitting experiment {0} in '{1}' test".format(self.name,test),file=sys.stderr)

        # Get seed calcuation functions for fitting routines, tailored to simulated (or any) data
        # Come from definitions in tests
        null_seeds = self.tests[test].null_seeds
        full_seeds = self.tests[test].full_seeds

        # Check if seeds actually give exact MLEs for parameters
        try:
           nseeds, nexact = null_seeds
        except TypeError:
           nseeds, nexact = null_seeds, False
        try:
           fseeds, fexact = full_seeds
        except TypeError:
           fseeds, fexact = full_seeds, False
         
        # Manual check of exactness of seeds
        #if fexact==False or nexact==False:
        #   raise ValueError("Non-exact seeds detected for experiment {0}, with test {1}!".format(self.name,test)) 
 
        if samples is not None: #if no samples provided, just do asymptotic calculations
           # Get seeds for fits (or get analytic MLEs if seeds are exact)
           #print("samples:",samples)
           #print("signal:", signal)
           #print("nseeds:", nseeds)
           #print("Computing seeds0")
           seeds0 = nseeds(samples,signal) # null hypothesis fits depend on signal parameters
           #print("seeds0", seeds0)
           #print("Computings seeds")
           #print("I think here the signal has not been modified to have mu=0")
           seeds  = fseeds(samples,signal) # In mu fit case, the seeds also depend on the signal
           #print("seeds", seeds)

           #print("Fitting free parameters...")
           extra_null_opt.update(signal) # pass on the signal parameters too, can sometimes be used to fix extra parameters
           Lmax0, pmax0, Lmax, pmax = self.get_LLR(model,test,samples,seeds0,seeds,nexact,fexact,extra_null_opt,signal)

           #print()
           #print("extra_null_opt:", extra_null_opt)
           #print("Lmax0",Lmax0)
           #print("pmax0",pmax0)
           #print("Lmax",Lmax)
           #print("pmax",pmax)
 
           # Run diagnostics functions for this experiment + test
           dfuncs = self.tests[test].diagnostics
           if dfuncs:
               print("Running extra diagnostic functions",file=sys.stderr)
               for f in dfuncs:
                   f(self, Lmax0, pmax0, seeds0, Lmax, pmax, seeds, samples)
 
           # Likelihood ratio test statistics
           LLR = -2*(Lmax0 - Lmax)
           #for a,b,c,d,e,f in zip(LLR, Lmax0, Lmax, samples, seeds0, seeds):
           #    print("LLR: ", a, ", Lmax0:", b, ", Lmax:", c, ", sample:", d, ", seeds0:", e, ", seeds:", f)

           # Check if seeds were good guesses:
           # print("seeds0:", seeds0)
           # for par in seeds0.keys():
           #     print(par)
           #     for a,b in zip(np.atleast_1d(pmax0[par]), np.atleast_1d(seeds0[par])):
           #         print("   pmax0 : ", a)
           #         print("   seeds0: ", b)
           # for par in seeds.keys():
           #     print(par)
           #     for a,b in zip(np.atleast_1d(pmax[par]), np.atleast_1d(seeds[par])):
           #         print("   pmax : ", a)
           #         print("   seeds: ", b)

           # print("Lmax:", Lmax)
           # print("Lmax0:", Lmax0)
           # print("LLR:", LLR)

           # Correct (hopefully) small numerical errors
           if LLR_must_be_positive:
               LLR[LLR<0] = 0
        else:
           LLR = None

        # Also fit the observed data so we can compute its p-value 
        print("Fitting with observed data...",file=sys.stderr)
        if observed is None:
            odata = self.observed_data
        else:
            print("Overriding observed data with supplied values...",file=sys.stderr)
            odata = observed
        #print("odata:", odata)
        seeds0_obs = nseeds(odata,signal) # null hypothesis fits depend on signal parameters
        seeds_obs  = fseeds(odata,signal)

        #print("seeds_obs:", seeds_obs)
        #print("seeds0_obs:", seeds0_obs)
        Lmax0_obs, pmax0_obs, Lmax_obs, pmax_obs = self.get_LLR(model,test,odata,seeds0_obs,seeds_obs,nexact,fexact,extra_null_opt,signal)
        LLR_obs = -2 * (Lmax0_obs - Lmax_obs)
 
        # print("Lmax_obs :", Lmax_obs)
        # print("Lmax0_obs:", Lmax0_obs)
        # print("LLR      :", LLR)  
        # print("samples:",samples)
        # print("signal:", signal)
 
        if LLR_must_be_positive:
            LLR_obs[LLR_obs<0] = 0
        # and LLR_obs<0:
        #            raise ValueError("LLR_obs < 0 for experiment {0}, but this is forbidden! Debug data follows:\n\
        #  LLR_obs   = {1}\n\
        #  Lmax0_obs = {2}\n\
        #  Lmax_obs  = {3}\n\
        #  odata     = {4}\n\
        #  pmax0_obs = {5}\n\
        #  pmax_obs  = {6}\n\
        #  null_fixed_pars = {7}\n\
        #  null_opt  = {8}\n\
        #  full_fixed_pars = {9}\n\
        #  full_opt  = {10}\n\
        #  seeds0_obs = {11}\n\
        #  seeds_obs = {12}\n\
        #".format(self.name,LLR_obs,Lmax0_obs,Lmax_obs,odata,pmax0_obs,pmax_obs,null_fixed_pars,null_opt,full_fixed_pars,full_opt,seeds0_obs,seeds_obs))
        #print("LLR_obs:", LLR_obs)
        #print("odata:", odata)
        return LLR, LLR_obs




