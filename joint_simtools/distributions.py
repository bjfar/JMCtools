import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as sps
from scipy.stats import rv_continuous
import scipy.optimize as spo
from scipy.integrate import quad

class ListModel:
    """
    Base class for freezable statistics objects (similar to those in scipy.stats)
    that are built from lists of scipy.stats (or similar) objects 
    """
    def __init__(self, submodels, parameters=None):
        """If parameters!=None, will freeze this model and assume that all submodels are frozen too"""
        self.submodels = submodels
        self.N_submodels = len(self.submodels)
        self.parameters = parameters
        if self.parameters==None or self.parameters==False:
            self.frozen = False
        else:
            self.frozen = True
 
    def _check_parameters(self, parameters=None):
        """Validate and return parameters
           Note that this is just the parameters of *this* object,
           which might be e.g. mixing parameters. Parameters
           of the submodels don't need to be supplied if they
           are supposed to be frozen, so we don't store them
           here in that case.
        """
        if self.frozen and parameters!=None:
            raise ValueError("This distribution is frozen! You are not permitted to alter the parameters used to compute the pdf of a frozen distribution object.")
        elif not self.frozen and parameters==None:
            raise ValueError("This distribution is not frozen, but no parameters were supplied to compute the pdf! Please provide some.")
        elif self.frozen and parameters==None:
            parameters = self.parameters
        elif not self.frozen and parameters!=None:
            pass # just use what was passed in 
        return parameters

    def _freeze_submodels(self, submodel_parameters):
        """Get list of all submodels, frozen with the supplied parameters"""
        if self.frozen:
            raise ValueError("This distribution is already frozen! You cannot re-freeze it with different parameters")
        else:
            out_submodels = []
            for submodel, pars in zip(self.submodels,submodel_parameters):
                out_submodels += [submodel(**pars)] # Freeze all submodels
        return out_submodels

# Handy class for sampling from mixture models in scipy.stats
class MixtureModel(ListModel):
    def __init__(self, submodels, weights=None, *args, **kwargs):
        super().__init__(submodels, weights)

    def __call__(self, weights=None, submodel_parameters=None):
        """Construct a 'frozen' version of the distribution
           Need to fix all parameters of all submodels if doing this.
        """
        if self.frozen:
            raise ValueError("This distribution is already frozen! You cannot re-freeze it with different parameters")
        self._check_parameters(submodel_parameters)
        weights = self._check_parameters(weights)
        frozen_submodels = self._freeze_submodels(submodel_parameters)
        return MixtureModel(frozen_submodels, weights) # Copy of this object, but frozen

    def pdf(self, x, weights=None, submodel_parameters=None):
        self._check_parameters(submodel_parameters)
        weights = self._check_parameters(weights)
        if weights==None:
            raise ValueError("No mixing weights supplied!")
        if submodel_parameters==None:
            submodel_parameters = [{} for i in range(len(self.submodels))]
        try:
            _pdf = weights[0] * np.exp(self.submodels[0].logpdf(x,**submodel_parameters[0]))
        except AttributeError:
            _pdf = weights[0] * np.exp(self.submodels[0].logpmf(x,**submodel_parameters[0]))
        for w,submodel,pars in zip(weights[1:],self.submodels[1:],submodel_parameters[1:]):
            try:
                _pdf += w*np.exp(submodel.logpdf(x,**pars))
            except AttributeError:
                _pdf += w*np.exp(submodel.logpmf(x,**pars))
        return _pdf
    
    def logpdf(self, *args, **kwargs):
        return np.log(self.pdf(*args,**kwargs)) # No better way to do this for mixture model

    def rvs(self, size, weights=None, submodel_parameters=None):
        #print('MixtureModel.rvs: ', weights, submodel_parameters)
        self._check_parameters(submodel_parameters)
        weights = self._check_parameters(weights)
        if weights==None:
            raise ValueError("No mixing weights supplied!")
        if submodel_parameters==None:
            submodel_parameters = [{} for i in range(len(self.submodels))]
        submodel_choices = np.random.choice(range(len(self.submodels)), p=weights, size=size)
        submodel_samples = [submodel.rvs(size=size,**pars) for submodel,pars in zip(self.submodels,submodel_parameters)]
        _rvs = np.choose(submodel_choices, submodel_samples)
        return _rvs
    
class PowerMixtureModel(rv_continuous):
    def __init__(self, submodels, weights, domain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = weights
        self.norm = 1
        self.domain = domain
        # Need to compute normalisation factor
        # Can do this numerically, but need user to specify the domain to use
        # No discrete distributions allowed, for now.
        # Also only mixtures of 1D models are allowed for now.
        res = quad(self.pdf,*self.domain) # this is a problem if we want to allow parameters to change...
        self.norm = res[0]
        #print "self.norm = ",self.norm
        
    def pdf(self, *args, **kwargs):
        return np.exp(self.logpdf(*args,**kwargs))
   
    def get_norm(self, x, weights=None, submodel_parameters=None):
        """Compute normalisation for pdf for many parameters in parallel (possibly)"""
        # do it by Monte Carlo integration?
        # Even if data vector is long, should only have to do this once for a given array of parameters
     
        # We are going to use 'x' to determine the shape of the parameter array
        # This means that 'x' must be input after it is already broadcasted against
        # other parameters etc. 
        # TODO: Finish this? It is pretty hard... don't actually need x here,
        # but need to somehow infer which direction of the parameter array actually
        # represent varying parameter values. We don't want to re-do the MC integration
        # for directions in which it is the data that changes. But that is hard to
        # infer from the parameter arrays alone.

        # tol = 0.001
        # Nbatch = 1000 # Number of random numbers per parameter point per iteration
        # while err > tol:
        #     # Generate batch of random numbers in self.domain
        #     for low,high in self.domain:
        #        np.random.uniform(low,high,s

        # self.pdf(x, weights, submodel_parameters) 

    def logpdf(self, x, weights=None, submodel_parameters=None):
        if weights==None:
            weights = self.weights # TODO check if frozen
        if submodel_parameters==None:
            submodel_parameters = [{} for i in range(len(self.submodels))]
        _logpdf = weights[0] * self.submodels[0].logpdf(x,**submodel_parameters[0])
        for w,submodel,pars in zip(weights[1:],self.submodels[1:],submodel_parameters[1:]):
            _logpdf += w * submodel.logpdf(x,**pars)
        return _logpdf - np.log(self.norm)
        
    def rvs(self, size):
        raise ValueError("Sorry, random samples cannot be drawn from this distribution, it is too freaky")
        # TODO: If the computation of the norm via MC integration is implemented, then it basically gives a set of samples
        # from the distribution at the same time. So these functions should be semi-combined.
        return None
    
class JointModel(ListModel):
    """Class for constructing a joint pdf from independent pdfs, which can be sampled from.
       Has a feature for overriding the pdf of submodels so that, for example, portions of
       the joint pdf may be profiled or marginalised analytically to speed up fitting routines.
    """
    def __init__(self, submodels, submodel_parameters=None, submodel_logpdf_replacements=None, *args, **kwargs):        
        super().__init__(submodels, submodel_parameters)
        # Here we DO need to store the submodel parameters, because we sometimes use them to evaluate
        # analytic replacements for the pdfs of the submodels

        if submodel_logpdf_replacements==None:
            self.submodel_logpdf_replacements = [None for i in range(len(self.submodels))]
        else:
            self.submodel_logpdf_replacements = submodel_logpdf_replacements

    def split(self,selection):
        """Create a JointModel which is a subset of this one, by splitting off the submodels 
           with the listed indices into a separate object"""
        if self.frozen:
            out = JointModel([self.submodels[i] for i in selection],
                          parameters = [self.parameters[i] for i in selection],
                          frozen = True,
                          submodel_logpdf_replacements = [self.submodel_logpdf_replacements[i] for i in selection]
                         )
        else:
            # If not frozen, cannot supply parameters
            out = JointModel([self.submodels[i] for i in selection],
                          frozen = False,
                          submodel_logpdf_replacements = [self.submodel_logpdf_replacements[i] for i in selection]
                         )
        return out
 
    def __call__(self, submodel_parameters=None):
        """Construct a 'frozen' version of the distribution
           Need to fix all parameters of all submodels if doing this.
        """
        if self.frozen:
            raise ValueError("This distribution is already frozen! You cannot re-freeze it with different parameters")
        submodel_parameters = self._check_parameters(submodel_parameters)
        frozen_submodels = self._freeze_submodels(submodel_parameters)
        return JointModel(frozen_submodels, submodel_parameters, self.submodel_logpdf_replacements) # Copy of this object, but frozen

    def submodel_logpdf(self,i,x,parameters={}):
        """Call logpdf (or logpmf) of a submodel, automatically detecting where parameters
           should come from"""
        if self.frozen and parameters!={}:
           raise ValueError("This distribution is frozen! You are not permitted to alter the parameters used to compute the pdf of a frozen distribution object.")
        elif not self.frozen and parameters=={}:
           raise ValueError("This distribution is not frozen, but no parameters were supplied to compute the pdf! Please provide some.")
        try:
             _logpdf = self.submodels[i].logpdf(x,**parameters)
        except AttributeError:
             _logpdf = self.submodels[i].logpmf(x,**parameters)
        return _logpdf 

    def submodel_pdf(self,i,x,parameters=None):
        return np.exp(self.submodel_logpdf(i,x,parameters))

    def pdf(self, x, parameters=None):
        """Provide data to pdf as a list of arrays of same
           length as the submodels list. Each array will be
           passed straight to those submodels, so broadcasting
           can be done at the submodel level. 
           This does mean that JointModel behaves DIFFERENTLY
           to "basic" distribution functions from scipy. So
           you should not construct JointModels with JointModels
           as submodels. Just make a new one from scratch in that
           case, since the submodels have to be statistically
           independent anyway.
           NOTE: could make a special constructor for JointModel
           to merge together pre-existing JointModels.

           parameters - list of dictionaries of parameters to be
           passed to the submodel pdf functions. If submodel is
           'frozen' then its corresponding element of 'parameters'
           should be an empty dictionary.
        """
        # Validate the supplied parameters (if any)
        submodel_parameters = self._check_parameters(submodel_parameters)
        #print("parameters:",parameters)
        # Use first submodel to determine pdf array output shape
        if self.submodel_logpdf_replacements[0]!=None:
            _pdf = np.exp(self.submodel_logpdf_replacements[0](x[0],**submodel_parameters[0]))
        else:
            _pdf = self.submodel_pdf(0,x[0],submodel_parameters[0])
        # Loop over rest of the submodels
        for i,(xi,submodel,alt_logpdf,pars) in enumerate(zip(x[1:],self.submodels[1:],self.submodel_logpdf_replacements[1:],submodel_parameters[1:])):
            #print(pars)
            if alt_logpdf!=None:
                _pdf *= np.exp(alt_logpdf(xi,**pars))
            else:
                _pdf *= self.submodel_pdf(i+1,xi,pars)
        return _pdf
    
    def logpdf(self, x, submodel_parameters=None):
        """As above but for logpdf
        """
        submodel_parameters = self._check_parameters(submodel_parameters)
    
        # If pdf is frozen, need to 'mute' parameters for submodels whose pdf's have not been replaced by analytic expressions 
        if self.frozen:
            for i in range(len(self.submodels)):
                if self.submodel_logpdf_replacements[i]==None:
                    submodel_parameters[i] = {}

        #print("JointModel.logpdf: x = ",x)
        #print("len(self.submodels):",len(self.submodels))
        # Use first submodel to determine pdf array output shape
        if self.submodel_logpdf_replacements[0]!=None:
            _logpdf = self.submodel_logpdf_replacements[0](x[0],**submodel_parameters[0])
        else:
            _logpdf = self.submodel_logpdf(0,x[0],submodel_parameters[0])
        # Loop over rest of the submodels
        if len(self.submodels)>1:
            for i,(xi,submodel,alt_logpdf,pars) in enumerate(zip(x[1:],self.submodels[1:],self.submodel_logpdf_replacements[1:],submodel_parameters[1:])):
                #print(pars)
                if alt_logpdf!=None:
                    _logpdf += alt_logpdf(xi,**pars)
                else:
                    _logpdf += self.submodel_logpdf(i+1,xi,pars)
        return _logpdf
    
    def set_submodel_logpdf(self, i, f):
        """Replace the logpdf function for the ith submodel"""
        self.submodel_logpdf_replacements[i] = f

    def set_logpdf(self, listf):
        """Replace the logpdf for all submodels (use 'None' for elements where
        you want to keep the original pdf"""
        self.submodel_logpdf_replacements = f

    def rvs(self, size, submodel_parameters=None):
        """Output will be a list of length N, where N is the number
        of random variables in the joint PDF. Each element will be an array of shape
        'size', possibly with extra dimensions if submodel is multivariate. That is, each variable is drawn
        with 'size', and the results are joined into a list."""
        #print("in rvs:", submodel_parameters)
        if self.frozen:
            submodel_parameters = [{} for i in range(len(self.submodels))]
        else:   
            submodel_parameters = self._check_parameters(submodel_parameters)
        submodel_samples = [submodel.rvs(size=size,**pars) for submodel, pars in zip(self.submodels,submodel_parameters)]
        return submodel_samples
