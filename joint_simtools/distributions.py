import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as sps
from scipy.stats import rv_continuous
import scipy.optimize as spo
from scipy.integrate import quad

# Handy class for sampling from mixture models in scipy.stats
class MixtureModel(rv_continuous):
    def __init__(self, submodels, weights, *args, **kwargs):
        super(MixtureModel,self).__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = weights

    def pdf(self, x):
        try:
            _pdf = self.weights[0] * np.exp(self.submodels[0].logpdf(x))
        except AttributeError:
            _pdf = self.weights[0] * np.exp(self.submodels[0].logpmf(x))
        for w,submodel in zip(self.weights[1:],self.submodels[1:]):
            try:
                _pdf += w*np.exp(submodel.logpdf(x))
            except AttributeError:
                _pdf += w*np.exp(submodel.logpmf(x))
        return _pdf
    
    def logpdf(self, x):
        return np.log(self.pdf(x)) # No better way to do this for mixture model

    def rvs(self, size):
        submodel_choices = np.random.choice(range(len(self.submodels)), p=self.weights, size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        _rvs = np.choose(submodel_choices, submodel_samples)
        return _rvs
    
class PowerMixtureModel(rv_continuous):
    def __init__(self, submodels, weights, domain, *args, **kwargs):
        super(PowerMixtureModel,self).__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = weights
        self.norm = 1
        # Need to compute normalisation factor
        # Can do this numerically, but need user to specify the domain to use
        # No discrete distributions allowed, for now.
        # Also only mixtures of 1D models are allowed for now.
        res = quad(self.pdf,*domain)
        self.norm = res[0]
        #print "self.norm = ",self.norm
        
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    
    def logpdf(self, x):
        _logpdf = self.weights[0] * self.submodels[0].logpdf(x)
        for w,submodel in zip(self.weights[1:],self.submodels[1:]):
            _logpdf += w * submodel.logpdf(x)
        return _logpdf - np.log(self.norm)
        
    def rvs(self, size):
        raise ValueError("Sorry, random samples cannot be drawn from this distribution, it is too freaky")
        return None
    
class JointModel(rv_continuous):
    """Class for constructing a joint pdf from independent pdfs, which can be sampled from.
       Has a feature for overriding the pdf of submodels so that, for example, portions of
       the joint pdf may be profiled or marginalised analytically to speed up fitting routines.
    """
    def __init__(self, submodels, parameters=None, frozen=None, submodel_logpdf_replacements=None, *args, **kwargs):
        super(JointModel,self).__init__(*args, **kwargs)
        self.submodels = submodels
        self.N_submodels = len(self.submodels)
 
        if submodel_logpdf_replacements==None:
            self.submodel_logpdf_replacements = [None for i in range(len(self.submodels))]
        else:
            self.submodel_logpdf_replacements = submodel_logpdf_replacements

        if parameters==None:
            self.parameters = [{} for i in range(len(self.submodels))]
        else:
            self.parameters = parameters

        if frozen==None and parameters==None:
            self.frozen = False
        elif frozen==None and parameters!=None:
            self.frozen = True
        elif frozen!=None and parameters==None:
            self.frozen = frozen
        elif frozen==False and parameters!=None:
            raise ValueError("This distribution has been manually set as 'not frozen', however \
you also supplied parameters to its constructor! This doesn't make sense. If frozen=True you \
may supply parameters (so that overridden pdfs can be called with these parameters), but if \
frozen=False then you should supply parameters later, when you call the pdf functions.")
 
 
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
 
    def __call__(self, parameters=None):
        """Construct a 'frozen' version of the distribution
           Need to fix all parameters of all submodels if doing this.
        """
        if self.frozen:
            raise ValueError("This distribution is already frozen! You cannot re-freeze it with different parameters")
        else:
            parameters = self.check_parameters(parameters)
            out_submodels = []
            for submodel, pars in zip(self.submodels,parameters):
                out_submodels += [submodel(**pars)]
        return JointModel(out_submodels, parameters) # Copy of this object, but frozen

    def check_parameters(self, parameters=None):
        """Validate and return parameters"""
        if self.frozen and parameters!=None:
           raise ValueError("This distribution is frozen! You are not permitted to alter the parameters used to compute the pdf of a frozen distribution object.")
        elif not self.frozen and parameters==None:
           raise ValueError("This distribution is not frozen, but no parameters were supplied to compute the pdf! Please provide some.")
        elif self.frozen and parameters==None:
            parameters = self.parameters # Return "inbuilt" parameters
        elif not self.frozen and parameters!=None:
            pass # just use what was passed in 
        return parameters

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
        parameters = self.check_parameters(parameters)
        print("parameters:",parameters)
        # Use first submodel to determine pdf array output shape
        if self.submodel_logpdf_replacements[0]!=None:
            _pdf = np.exp(self.submodel_logpdf_replacements[0](x[0],**parameters[0]))
        else:
            _pdf = self.submodel_pdf(0,x[0],parameters[0])
        # Loop over rest of the submodels
        for i,(xi,submodel,alt_logpdf,pars) in enumerate(zip(x[1:],self.submodels[1:],self.submodel_logpdf_replacements[1:],parameters[1:])):
            print(pars)
            if alt_logpdf!=None:
                _pdf *= np.exp(alt_logpdf(xi,**pars))
            else:
                _pdf *= self.submodel_pdf(i+1,xi,pars)
        return _pdf
    
    def logpdf(self, x, parameters=None):
        """As above but for logpdf
        """
        parameters = self.check_parameters(parameters)

        # Use first submodel to determine pdf array output shape
        if self.submodel_logpdf_replacements[0]!=None:
            _logpdf = self.submodel_logpdf_replacements[0](x[0],**parameters[0])
        else:
            _logpdf = self.submodel_logpdf(0,x[0],parameters[0])
        # Loop over rest of the submodels
        for i,(xi,submodel,alt_logpdf,pars) in enumerate(zip(x[1:],self.submodels[1:],self.submodel_logpdf_replacements[1:],parameters[1:])):
            print(pars)
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

    def rvs(self, size, parameters=None):
        """Output will be a list of length N, where N is the number
        of random variables in the joint PDF. Each element will be an array of shape
        'size', possibly with extra dimensions if submodel is multivariate. That is, each variable is drawn
        with 'size', and the results are joined into a list."""
        if self.frozen:
            parameters = [{} for i in range(len(self.submodels))]
        else:   
            parameters = self.check_parameters(parameters)
        submodel_samples = [submodel.rvs(size=size,**pars) for submodel, pars in zip(self.submodels,parameters)]
        return submodel_samples
