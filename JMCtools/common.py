"""Common helper tools"""

import numpy as np
import scipy.interpolate as spi
import inspect

# def apply_f(f,a):
#     """Apply some function to 'bottom level' objects in a nested structure of lists,
#        return the result in the same nested listed structure.
#        Credit: https://stackoverflow.com/a/43357135/1447953
#     """
#     if isinstance(a,list):
#         return map(lambda u: apply_f(f,u), a)
#     else:
#         return f(a)
# 
# def apply_f_binary(f,a,b):
#     """Apply some binary function to matching 'bottom level' objects 
#        in mirrored nested structure of lists,
#        return the result in the same nested listed structure.
#     """
#     # We have to descend both list structures in lock-step!
#     if isinstance(a,list) and isinstance(b,list):
#         return map(lambda u,v: apply_f_binary(f,u,v), a, b)
#     else:
#         return f(a,b)

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    In Python >3.5 can just do {**a,**b,etc}, but we need backwards
    compatibility sadly...
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# Generalisation of the above to any number of arguments
def apply_f(f,*iters):
    """Apply some function to matching 'bottom level' objects 
       in mirrored nested structure of lists,
       return the result in the same nested listed structure.
       'iters' should be 
    """
    # We have to descend both list structures in lock-step!
    if all(isinstance(item,list) for item in iters):
        return list(map(lambda *items: apply_f(f,*items), *iters))
    elif any(isinstance(item,list) for item in iters):
        raise ValueError("Inconsistency in nested list structure of arguments detected! Nested structures must be identical in order to apply functions over them")
    else:
        return f(*iters)

def almost_flatten(A):
    """Flatten array in all except last dimension"""
    return A.reshape((-1,A.shape[-1]))

def get_data_slice(x,i,j=None):
    """Extract a single data realisation from 'x', or a numpy 'slice' of realisations
       This harder than it sounds because we don't know what object structure
       we are dealing with. For example the JointModel to which we interface
       could be built from a bunch of different MixtureModels which are themselves 
       made out of JointModels, so that the actual data arrays are buried in
       a complicated list of lists of structure. We need to descend through these
       lists, pluck a data realisation out of every "bottom level" array, and
       put everything back together in the same list structure. And we need to
       do it pretty efficiently since we're going to have to iterate through these 
       slices.

       Edit: Modified to deal with differently shapped arrays.
             Slicing indices always assumed to apply to first dimension.
             This now means that data realisations should only iterate over the
             first dimension, they should not be some bizarre shape.
             If they are a weird shape they need to be reshaped before this
             function can be applied.
    """
    data, size = x
    if j==None:
       data_slice = list(apply_f(lambda A: A[i,...],data))
       slice_length = 1
    else:
       data_slice = list(apply_f(lambda A: A[i:j,...],data))
       slice_length = j-i
    return data_slice, tuple([slice_length] + list(size[1:]))

def get_data_structure(x):
    """Report the nested structure of a list of lists of numpy arrays"""
    return list(apply_f(lambda A: A.shape, x))

def split_data(samples,dims):
    """Split a numpy array of data into a list of sub-arrays to be passed to independent
    submodel objects.
    Components must be indexed by last dimension of 'samples' array
    'dims' specifies the number of elements in last dimension to slice out, e.g.
    dims = [1,1,2]
    would assume that samples had a last dimensions of size 4, and it would be split
    into 3 parts with sizes 1, 1, and 2 respectively.
    """
    out = []
    i = 0 # Next index to be sliced
    #print("samples shape:",samples.shape)
    #print("dims:", dims)
    # Check dimensions
    if samples.shape[-1] != np.sum(dims):
        raise ValueError("Dimension mismatch between supplied \
arguments! 'samples' has last dimension of size {0}, however the sum \
of the requested slice sizes is {1}! These need to match.".format(samples.shape[-1],np.sum(dims)))
    for d in dims:
        if d==1:
           #print("slicing: {0}".format(i))
           out += [samples[...,i]]
        else:
           #print("slicing: {0}:{1}".format(i,i+d))
           out += [samples[...,i:i+d]]
        i = i+d
    #print("split samples shapes:",[o.shape for o in out])
    return out

def eCDF(x):
    """Get empirical CDF of some samples"""
    return np.arange(1, len(x)+1)/float(len(x))

def e_pval(samples,obs):
    """Compute an empirical p-value based on simulated test-statistic values"""
    # Better sort the samples into ascending order first!
    # Note that sorting is along *last* axis by default! Probably only want
    # 1D data, but just remember this!
 
    # One problem: we have to decide how to treat NaNs.
    # I think it is best if we just remove them and pretend they
    # don't exist.
    s = np.sort(samples[np.isfinite(samples)])
    CDF = spi.interp1d([0]+list(s)+[1e99],[0]+list(eCDF(s))+[1])
    #for si in s:
    #   print(si, CDF(si))
    print("obs:",obs)
    print("CDF(obs):",CDF(obs))
    print("pval(obs):",1-CDF(obs))
    return 1 - CDF(obs)

def get_func_args(func):
    """Inspect a function for its named arguments
       (only returns ones that have no default value
        set)"""
    try:
       argo = inspect.getfullargspec(func)
       fargs = argo.args 
       for arg in argo.kwonlyargs:
          # Only add the kwonly arg if it doesn't have a default value provided
          if arg not in argo.kwonlydefaults.keys():
             fargs += [arg]
    except AttributeError:
       # Might be in Python 2. This will fail on some function signatures,
       # but we can give it a whirl:
       try:
          fargs = inspect.getargspec(func)[0]
       except TypeError:
          print("Could not parse the arguments of the supplied function! You\
 can usually instead supply them as an explicit list to whatever object has\
 called this routine.")
          raise
    
    if len(fargs)==0:
       raise ValueError("Failed to find any arguments for the supplied\
 function! It may not have explicit arguments (i.e. it may use *args, \
 **kwargs instead, or just have no arguments). If you have received this\
 error while using an object such as TransDist then you may need to\
 supply an explict argument list to be used (please check where\
 this function was called from; the calling object should provide a way to\
 supply such a list, for example the 'func_args' argument can be used in the\
 TransDist constructor.")
    return fargs 

def get_dist_args(dist):
    """Probe a scipy.stats distribution (or JMCtools distribution) for its
       required arguments"""
    has_args = False
    try:
        fargs = dist.args
        has_args = True
    except AttributeError:
        has_args = False
    if not has_args:
        try:
            func_args = get_func_args(dist.logpdf)
        except AttributeError:
            func_args = get_func_args(dist.logpmf)
        # Remove 'x' from this list, we only want the parameter names
        reject_list = ['x','self']
        fargs = [item for item in func_args if item not in reject_list]
        if len(fargs)==0:
            raise ValueError("Failed to find any arguments for the supplied\
 distribution after inspecting its functions! They may not have explicit\
 arguments (i.e. they may use *args, **kwargs). If that is the case then you\
 will need to supply an explict argument list to be used (please check where\
 this function was called from; the calling object should provide a way to\
 supply such a list, for example the 'func_args' argument can be used in the\
 TransDist constructor.")
    return fargs

