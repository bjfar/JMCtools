from __future__ import print_function
import JMCtools as jt
import JMCtools.distributions as jtd
import numpy as np
import JMCtools.common as c
import concurrent.futures
import itertools
import time
import sys
import JMCtools.six as six # Python 2 & 3 compatibility tools

# Hacky thing to get a decent traceback from concurrent processing in Python 2.x
# Credit: https://stackoverflow.com/a/29357032/1447953
import functools
import traceback
def reraise_with_stack(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback_str = traceback.format_exc(e)
            raise StandardError("Error occurred. Original traceback "
                                "is\n%s\n" % traceback_str)

    return wrapped

class Objective:
    """Constructs the objective function to be fitted my iminuit in the 
       'minuit' fitting method of ParameterModel.getMLEs.
       Inspired by example here:
       http://iminuit.readthedocs.io/en/latest/api.html#function-signature-extraction-ordering
    """
    def __init__(self, block, x):
        import iminuit
        self.block = block
        self.x = x 
        args = list(self.block.deps) # parameters required by the block, and thus this objective function
        #print('Objective function constructed with args:',args)
        #print('Data is:',self.x)
        self.func_code = iminuit.Struct(
                co_varnames = args,
                co_argcount = len(args)
                )
        self.run_yet = False

    def __call__(self, *args):
        parameters = {}
        #if not self.run_yet: print("Calling Objective function...")
        #print("Calling Objective function...")
        #print("Data is:", self.x)
        #print("block.submodels:",self.block.submodels)
        #print("block.submodel_deps:",self.block.submodel_deps)
        #if not np.all(np.isfinite(args)):
        #    raise ValueError("iminuit tried to call objective function with nan arguments! It has probably freaked out because the problem was too hard or something. Try improving your starting guesses and step sizes. (args were: {0})".format(args))
        #print("Args are:")
        for name,val in zip(self.block.deps,args):
            #if not self.run_yet:
            #    print("Starting value for {0} is {1}".format(name,val))
            #V = np.atleast_1d(val)
            #print("   {0}: {1}".format(name,val))
            parameters[name] = val #V.reshape(V.shape + tuple([1]*len(self.x.shape[:-1]))) # Add new singleton axes to broadcast against data 
            #print("parameters[{0}].shape:".format(name),parameters[name].shape)
        self.run_yet = True
        block_logpdf_all = self.block.logpdf(self.x, parameters)
        #print("block_logpdf_all.shape",block_logpdf_all.shape)
        #print("block_logpdf_all:",block_logpdf_all)
        block_logpdf = np.sum(block_logpdf_all,axis=-1) 
        #print("-2*block_logpdf:",-2*block_logpdf)
        return -2*block_logpdf
 
class Block:
    """Record of dependency structure of a block of submodels in ParameterModel"""
    def __init__(self,deps,submodels,jointmodel=None,submodel_deps=None):
       self.deps = frozenset(deps) # parameter dependencies
       self.submodels = frozenset(submodels) # indices of submodels associated with this block 
       self.jointmodel = jointmodel
       self.submodel_deps = submodel_deps # parameter dependencies of each submodel in jointmodel

    @classmethod  
    def fromBlock(cls,block,jointmodel=None,submodel_deps=None):
       deps = block.deps
       submodels = block.submodels
       if (block.jointmodel!=None) and (jointmodel!=None):
          raise ValueError("Tried to set 'jointmodel' for a copy of a Block that already has a 'jointmodel' set!")
       if block.jointmodel!=None:
          jointmodel = block.jointmodel
       return cls(deps,submodels,jointmodel,submodel_deps)

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

        # We don't worry about transforming anymore, but we still
        # need to provide the parameters as a list for JointModel.
        args = []
        for deps in self.submodel_deps:
            args += [{p:parameters[p] for p in deps}] # extract just the parameters that this submodel needs
        return args

    def logpdf(self,x,parameters):
        return self.jointmodel.logpdf(x,self.get_pdf_args(parameters))

# Functions to be run by worker processes in ParameterModel.find_MLE_parallel
# Slightly different approach used depending on the minimisation
# method chosen.
@reraise_with_stack
def loop_func_grid(par_object,options,i,data_slice):
   #data_slice = c.get_data_slice(data,i*chunksize,i*chunksize+slicesize)  
   Lmax_chunk, pmax_chunk = par_object.find_MLE(options,method='grid',x=data_slice)
   return i, Lmax_chunk, pmax_chunk

@reraise_with_stack
def loop_func_minuit(par_object,options,i,data_slice):
    # Here we farm off to worker processes by chunk
    # Minuit can't do the whole chunk at once, but
    # we still reduce overhead related to launching
    # the parallel processes.
    ## slicesize = chunksize
    ## if i==Nchunks and Nremainder!=0:
    ##     slicesize = Nremainder
    ## Lmax_chunk = np.zeros(slicesize)
    ## pmax_chunk = {p: np.zeros(slicesize) for p in self.parameters}
    ## for j in range(slicesize):
    ##    ia = i*chunksize+j # absolute dataset index
    ##    datai = c.get_data_slice(data,ia)  
    ##    # Get maximum likelihood estimators for all parameters
    ##    Lmax_chunk[j], pmax = parmix.find_MLE(m_options,method='minuit',x=datai)
    ##    for p,val in pmax_chunk.items():
    ##       val[j] = pmax[p]
    #print('Running find_MLE with options:', options)
    #Lmax_chunk, pmax_chunk = par_object.find_MLE(options,method='minuit',x=data_slice)

    # No more chunking, can be taken care of elsewhere.
    Lmax, pmax = par_object.find_MLE(options,method='minuit',x=data_slice)
    return i, Lmax, pmax

class ChunkedData:
    """A wrapper for ParameterModel data which turns it into a
       generator, so that we can efficiently iterate through the
       data
    """
    def __init__(self,x,chunksize):
        self.x = x
        self.Ntrials, self.Ndraws, self.Ncomponents = x.shape
        self.chunksize = chunksize
        self.Nchunks = self.Ntrials // chunksize
        self.Nremainder = self.Ntrials % chunksize
        self.i = 0 # Index of next data chunk to be returned

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.Nchunks:
            raise StopIteration  # Done iterating.
        slicesize = self.chunksize
        if self.i==self.Nchunks and self.Nremainder!=0:
            slicesize = self.Nremainder
        # Extract the data slice
        #print("Extracting slice {0}".format(self.i))
        #dataslice = self.x[self.i] #c.get_data_slice(self.x,self.i*self.chunksize,self.i*self.chunksize+slicesize)
        dataslice = self.x[self.i*self.chunksize:self.i*self.chunksize+slicesize] #c.get_data_slice(self.x,self.i*self.chunksize,self.i*self.chunksize+slicesize)
        #print("   data is:", dataslice)
        self.i += 1 # Move iterator variable
        return dataslice 

    def next(self):
        """For Python 2.x compatibility"""
        return self.__next__()

class ParameterModel:     
    """An object for managing the mapping between a parameter space and
       a set of distribution functions, i.e. a JointModel object.
       Also has methods for computing common statistics, such as
       maximum likelihood estimators"""

    #def __init__(self,jointmodel,parameter_funcs,parameter_names=None,x=None,fix={}):        
    #    """Note: once this object is constructed, it is best if you
    #       don't mess around with the internals of the stored JointModels
    #       and so on, or else you might screw up the internal
    #       consistency of the member routines in this object"""

    # Changing this up. Should FIRST convert all submodels into the appropriate
    # parameterisation using TransDist. Then just supply them all directly
    # to this class. Or I guess can wrap them in a jointmodel first, why not. We'll
    # stick with that for now.
  
    # Why not both?
 
    @classmethod  
    def fromList(cls,submodels_and_parameters,x=None,fix={}):
       try:
           submodels = []
           parameters = []
           for submodel, pars in submodels_and_parameters:
             submodels += [submodel]
             parameters += [pars]
       except ValueError:
          # No parameters provided (probably), will have to infer them 
          submodels = submodels_and_parameters
          parameters = None
       joint = jtd.JointDist(submodels)
       return cls(joint,parameters,x,fix)  
    
    def __init__(self,jointmodel,parameters=None,x=None,fix={}):
        iterable = True
        try:
            #print("Testing if iterable")
            # We can allow a list of submodels to be supplied rather than a pre-constructed jointmodel
            # Both can work.
            iter(jointmodel)
        except TypeError:
            iterable = False

        if iterable:
            # If 'jointmodel' is a list of (submodel,transform_function) pairs then we
            # can also deal with that automatically:
            transmodels = []
            try:
               #print("Testing if list of pairs")
               for dist, func in jointmodel:
                   transmodels += [jtd.TransDist(dist,func)]
               self.model = jtd.JointDist(transmodels)
            except ValueError:
               try:
                   #print("Testing if list of triples")
                   for dist, func, remap in jointmodel:
                       transmodels += [jtd.TransDist(dist,func,remap)]
                   self.model = jtd.JointDist(transmodels)
               except ValueError:
                   print("Wrong number of iterables?")
                   raise
            except TypeError:
                # Ok items in list are not iterable: assume it is already a good argument for the JointModel constructor
               self.model = jtd.JointDist(jointmodel)
        else:
            #print("Assigning directly to self.model") 
            # Ok not a list or something, assume it is already JointModel object    
            self.model = jointmodel

        #print('model:',self.model)
        #print('submodels:', self.model.submodels)
        #print('parameters:', parameters)
        # If manual parameter names are specified, uses those. Otherwise attempt to
        # infer them from the function signatures of the underlying distribution.
        if parameters is None:
            parameters = [None for submodel in self.model.submodels]
        
        self.submodel_deps = []
        for submodel,pars in zip(self.model.submodels,parameters):
            if pars is None:
                func_args = c.get_dist_args(submodel)
            else:
                func_args = pars
            self.submodel_deps += [func_args] 
        #print('self.submodel_deps:', self.submodel_deps)
 
        if x==None:
           self.x = [None for i in range(self.model.N_submodels)]
        else:
           self.validate_data(x)
           self.x = x

        # Get complete list of all parameters
        self.parameters = set([])
        for deps in self.submodel_deps:
           self.parameters.update(deps)

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
        block_list = set([Block(deps,[i]) for i,deps in enumerate(self.submodel_deps)])
        no_merge_occurred = False
        while not no_merge_occurred:
           new_block_list = set([])
           for block in block_list:
              block_deps = set([])
              block_submodels = set([])
              for parameter in block.deps:
                 # We need to skip parameters that are supposed to be considered as fixed
                 if parameter not in fix.keys():
                    matches = self.find_submodels_which_depend_on(parameter)
                    if len(matches)>0:
                       block_submodels.update(matches)
                       # Add all the parameters on which the newly added submodels depend
                       for i in matches:
                          block_deps.update(self.submodel_deps[i])
              #new_block_list.add((frozenset(block_deps),frozenset(block_submodels)))
              new_block_list.add(Block(block_deps,block_submodels))
           if(block_list==new_block_list):
              no_merge_occurred = True # Nothing changed, we are finished.   
           block_list = new_block_list

        # Break down full JointModel into individual JointModels for each block
        # and add these to the Blocks       
        final_block_list = set([])
        for block in block_list:
           smlist = block.submodels # It is important that we know the order of submodels in the new JointModel,
                                    # so that we can split up input data arrays correctly. The frozenset should
                                    # have a fixed iteration order I think, since no elements of it will change.
           block_jointmodel = self.model.split(smlist)
           block_submodel_deps = [self.submodel_deps[i] for i in smlist]
           final_block_list.add(Block.fromBlock(block,block_jointmodel,block_submodel_deps))

        self.blocks = final_block_list
        #print(self.blocks)

    def get_pdf_args(self,parameters):
        """Obtain the argument list required to evaluate the joint
           pdf of a block, from the more abstract parameters known
           to this object"""
       
        # For each submodel, we need to run its 'parf' function
        # But to do that we need to first know what parameters that
        # function needs.
        # Fortunately, this is stored in 'submodel_deps'
        args = []
        #print("In JointModel.get_pdf_args: parameters = ",parameters)
        for deps in self.submodel_deps:
            try:
                pars = {p:parameters[p] for p in deps} # extract just the parameters that a given submodel needs
            except Exception:
                for p in deps:
                    try:
                        parameters[p]
                    except KeyError:
                        raise KeyError("In JointModel: Parameter '{0}' missing! A submodel expected parameters {1}, but we have only received parameters {2}".format(p,deps,parameters.keys()))  
            args += [pars] # List of dictionaries of arguments to be passed to each submodel
        return args

    def logpdf(self,parameters,x=None):
        if x is None:
           x = self.x # Use pre-generated data if none provided
        else:
           self.validate_data(x)
        #print("pdf_args: ", self.get_pdf_args(parameters))
        return self.model.logpdf(x,self.get_pdf_args(parameters))

    def logpdf_list(self,parameters,x=None):
        """Gets logpdfs of all submodels of self.model as a list
           Mostly for debugging purposes."""
        if x is None:
           x = self.x # Use pre-generated data if none provided
        else:
           self.validate_data(x)
        #print("pdf_args: ", self.get_pdf_args(parameters))
        return self.model.logpdf_list(x,self.get_pdf_args(parameters))


    def find_submodels_which_depend_on(self,parameter):
        """Return the indices of the submodels which depend on 'parameter'"""
        matches = set([])
        for i,d in enumerate(self.submodel_deps):
            if parameter in d:
                matches.add(i)
        return matches

    def validate_data(self,x,verbose=False):
        pass
        # Data now supplied as big array; need to change validation
        #if verbose: print("x:",x)
        #data,size = x
        #if len(data)!=self.model.N_submodels:
        #   raise ValueError("The length of the supplied list of data values does not match the number of submodels in the wrapped JointModel! This means that this data cannot possibly be generated by (or used to compute the pdf of) this model and so is invalid.")

    def set_data(self,x):
        """Set internally-stored data realisations to use for parameter 
           estimation etc."""
        self.validate_data(x)
        self.x = x

    def simulate(self,Ntrials,null_parameters=None,Ndraws=1):
        """Generate simulated datasets to use for parameter estimation etc.
           null_parameters - parameters to use for the simulation.
           Ntrials - Number of pseudoexperiments to simulate
           Ndraws - Number of draws per pseudoexperiment
           These two parameters affect the shape of the output! The output
           will have dimensions shaped as follows:
           (Ntrials,Ndraws,Ncomponents)
           where Ncomponents is the total number of random variables
           simulated (at least one per submodel; more if submodels are
           multivariate). This is the shape required by other routines
           in the ParameterModel class.
        """
        #print(null_parameters)
        if null_parameters!=None:
           args = self.get_pdf_args(null_parameters)
        else:
           args = {}
        #print("in simulate: args = ",args)
        self.x = self.model.rvs((Ntrials,Ndraws),args)
        return self.x     

    def find_MLE(self,options,x=None,method='grid'):
        """Find the global maximum likelihood estimate of the model parameters,
           using data as x
           If x contains a (list of) arrays, then returns MLEs for each element of that
           array. That is, this function is vectorised with respect to data
           realisations.
           Note, x should match the output of 'simulate', so it should be a tuple
           x = (self.model.rvs(N), N)
           options["ranges"] - dictionary giving ranges to scan in each parameter direction.
           options["N"] - number of parameter points to use in each dimension of grid search
           (TODO: choose algorithm to use for search)

           We search for the MLE by breaking up the problem into independent
           pieces, since we know which submodels of the joint pdf depend
           on which parameters, so we can do the maximisation in several steps.

           method - Type of optimisation routine to run on each "block" of parameters
              'grid' - Just uses np.max over a big array of parameter values.
                       Good and very fast for low dimensions and many data realisations,
                       but will quickly become very slow (and use a lot of RAM) once 
                       dimensions exceed say 3. Also prone to discretisation errors
                       if the grid is too coarse relative to the region around the 
                       likelihood function maximum.
              'minuit' - Uses gradient-descent algorithms in Minuit (via iminuit)
                         Fast for simple likelihood surfaces with decent initial guesses.
                         Can't do all data realisations at once like 'grid' can, so
                         can get slow if there are lots of data realisations to loop over.
              'MCMC' - (Not implemented, just a plan) Good for moderate to high dimension
                       functions, though not so good if likelhood function is multi-modal.
              'ScannerBit' - (Not implemented) would be cool to do this!
        """
        if x is None:
           x = self.x # Use pre-generated data if none provided
        else:
           self.validate_data(x)
        
        #print('in find_mle:, data=',x)
        Lmax_tot = 0 # output maximum log-likelihood
        MLE_pars_full = {}
        for i,block in enumerate(self.blocks):
           block_data = self.repack_data(x,block.submodels) # Select data relevant to this submodel
           #print('in find_mle:, block.submodels: {0}'.format(block.submodels))
           #print('in find_mle:, block_data ({0})={1}'.format(i,block_data))
           Lmax, MLE_pars = self.find_MLE_for_block(block,options,block_data,method)
           Lmax_tot += Lmax
           MLE_pars_full.update(MLE_pars)
        return Lmax_tot, MLE_pars_full

    def repack_data(self,data,submodels):
        """Split data array into sub-arrays appropriate for evaluating the PDF of
           each block; i.e. for feeding to the JointModel belonging to each block"""
        # datalist = []
        # i = 0 # Next index to be sliced
        # for d in self.model.dims:
        #     datalist += [data[...,i:i+d]]
        #     i = i+d
        datalist = self.model.split_data(data)
        # Data is now in list form. Need to select the submodels we want
        # (in the order we want!) and return the re-stacked array
        size = data.shape[:-1]
        newsize = tuple(list(size) + [-1]) 
        out = np.concatenate([datalist[j].reshape(*newsize) for j in submodels],axis=-1) #Makes sure number of dimensions is correct
        return out 

    def find_MLE_parallel(self,options,x=None,method='grid',chunksize=100,Nprocesses=None,seeds=None):
        """Find MLEs for a large number of sample realisations, by breaking the samples into
           chunks and evaluating them in parallel.
           Note: This currently only works if the data has underlying shape (Ntrials,Ndraws), where we
           will chunk over Ntrials, and Ndraws is treated as independent draws from our pdf
           which should be automatically combined into a joint pdf.
        """
        if x is None:
           x = self.x # Use pre-generated data if none provided
        else:
           self.validate_data(x)

        Ntrials, Ndraws, Ncomponents = x.shape
        pmax_all = {p: np.zeros(Ntrials) for p in self.parameters} # This is a bit wasteful if there are lots of fixed parameters, but oh well.
        Lmax_all = np.zeros(Ntrials)

        starttime = time.time()
 
        print("Using method {0}".format(method))

        # Evaluate the chunks in parallel
        if method=='grid':
            loopfunc = loop_func_grid
            Nchunks = Ntrials // chunksize
            Nremainder = Ntrials % chunksize
            with concurrent.futures.ProcessPoolExecutor(Nprocesses) as executor:
                for i, Lmax, pmax in executor.map(loopfunc, itertools.repeat(self), 
                                                            itertools.repeat(options), 
                                                            range(Nchunks), 
                                                            ChunkedData(x,chunksize) 
                                                  ):
                    # Collect results
                    print("\r","Getting {0} MLEs for chunk {1} of {2}           ".format(method,i,Nchunks), end="")
                    #stdout.write("\rGetting {0} MLEs for chunk {1} of {2}           ".format(method,i,Nchunks))
                    #stdout.flush()
                    slicesize = chunksize
                    if i==Nchunks and Nremainder!=0:
                        slicesize = Nremainder
                    start, end = i*chunksize, i*chunksize + slicesize
                    Lmax_all[start:end] = Lmax
                    for p,val in pmax_all.items():
                       val[start:end] = pmax[p]
        elif method=='minuit':
            loopfunc = loop_func_minuit
            # Set up seeds for initial points if desired

            # Figure out which parameters are fixed
            fixed = []
            for key,val in options.items():
                words = key.split("_")
                if words[0]=="fix" and val==True:
                    fixed += [words[1]]

            if seeds!=None:
                def seed(i):
                    out = {}
                    for p,val in seeds.items():
                        if p not in fixed:
                            # Check that seed is a scalar
                            if not np.size(val[i])==1:
                                raise ValueError("{0}th seed for parameter {1} was not a scalar! Minuit requires all variables to be scalar, so non-scalars make no sense as starting guesses. (seed provided was: {2})".format(i,p,val[i]))
                            out[p] = val[i]
                    return out
                seed_opts = map(seed,range(Ntrials))
                opts = []
                for sopts in seed_opts:
                   opts += [c.merge_dicts(options,sopts)]
            else:
                opts = itertools.repeat(options)
            # Check that iterator construction worked
            #for i,o in enumerate(opts):
            #    print("options {0}: {1}\n".format(i,o))
            did_we_run=False
            with concurrent.futures.ProcessPoolExecutor(Nprocesses) as executor:
                for i, Lmax, pmax in executor.map(loopfunc, itertools.repeat(self), 
                                                           opts, 
                                                           range(Ntrials), 
                                                           ChunkedData(x,1), 
                                                 chunksize=chunksize):
                   did_we_run=True # Need to check if this doesn't run for some reason
                   if i % chunksize==0: 
                       print("\r","Getting {0} MLEs for chunk {1} of {2}              ".format(method,i//chunksize,Ntrials//chunksize), end="")
                       #stdout.write("\rGetting {0} MLEs for chunk {1} of {2}           ".format(method,i,Nchunks))
                       #stdout.flush()
                   Lmax_all[i] = Lmax
                   for p,val in pmax_all.items():
                      #if(p=='theta_1'): print("found theta_1 MLE as {0}".format(pmax[p])) 
                      val[i] = pmax[p]
            if not did_we_run:
                raise ValueError("Failed to perform any minimisations! Supplied data vector might be null, or this might be a bug.")
        else:
            raise ValueError("Unrecognised minimisation method selected!")

        endtime = time.time()
        print("\nTook {0:.0f} seconds".format(endtime-starttime))

        return Lmax_all, pmax_all
 
    def find_MLE_for_block(self,block,options,block_x,method):
        """Find MLE for a single block of parameters
           Mostly for internal use
        """
        if method=='grid':
            Lmax, pmax = self.find_MLE_for_block_with_grid(block,options,block_x)         
        elif method=='minuit':
            Lmax, pmax = self.find_MLE_for_block_with_Minuit(block,options,block_x)         
        else:
            raise ValueError('Unknown MLE-finding method specified! Please choose from: "grid", "minuit"')
        return Lmax, pmax
 
    def find_MLE_for_block_with_grid(self,block,options,block_x):
        """Find MLE for a single block of parameters using 'grid'
           method. Mostly for internal use.
        """
        # Construct some ND cube of parameters and maximise over it?
        # This is the simplest thing that is vectorisable, but will run
        # out of RAM if more than a couple of parameter dimensions needed at once
        # Oh well just try it for now. We can make it fancier later.
        
        # We are going to need to infer the shape of the simulated data arrays.
        #

        # block.deps contains a list of parameters on which this
        # block of submodels depends, e.g. ["p1","p2","p3"]
        # block.submodels contains a list of indices of the submodels
        # which belong to this block
        # The first task, then, is to compute the parameter cube that
        # we want to scan
        N = options["N"]
        ranges = options["ranges"]
        p1d = []
        fixed = {par: len(np.atleast_1d(ranges[par]))==1 for par in block.deps} # See if any parameters are fixed.
        for par in block.deps:
            if fixed[par]:
                #print('par {0} is fixed!'.format(par))
                p1d += [ np.array([ranges[par]]) ] # interpret single range entry as a fixed parameter value
            else:
                p1d += [ np.linspace(*ranges[par], num=N, endpoint=True) ]
        PARS = np.meshgrid(*p1d,indexing='ij')
        #print('block_x.shape:',block_x.shape)
        #print('PARS[0].shape:',PARS[0].shape)
        parameters = {}
        for i,par in enumerate(block.deps):
           parameters[par] = PARS[i].reshape(PARS[i].shape + tuple([1]*(len(block_x.shape)-1))) # Add new singleton axes to broadcast against data
           #print('parameters[{0}].shape: {1}'.format(par,parameters[par].shape))
 
        block_logpdf_all = block.logpdf(block_x,parameters)
        #print("block_logpdf_all.shape:", block_logpdf_all.shape)
 
        # We now want to interpret the last dimension of data as 'events per trial',
        # So we compute the joint logpdf as the sum over the last dimension of this output
        block_logpdf = np.sum(block_logpdf_all,axis=-1) 
        #print("block_logpdf.shape:", block_logpdf.shape)
 
        # Maximise over all dimensions except the data dimension

        # Use this to check result of reshape/indexing manipulations
        #print("flatten axes:",tuple(range(len(block.deps))))
        Lmax_simple = np.max(block_logpdf,axis=tuple(range(len(block.deps))))

        # During reshape; keep size of last dimension, 'flatten' the rest, then maximise over the flattened part
        # No idea if the ordering is correctly presevered here...
        #max_idx = np.swapaxes(block_logpdf,0,-1).reshape((block_logpdf.shape[-1],-1)).argmax(-1)
        flatview = c.almost_flatten(block_logpdf)
        max_idx = flatview.argmax(axis=0)
        #print("max_idx.shape:", max_idx.shape)        
        #print("max_idx:", max_idx)

        # Get unravel indices corresponding to original shape of A
        #maxpos_vect = np.column_stack(np.unravel_index(max_idx, np.swapaxes(block_logpdf,0,-1).shape[1:]))
        maxpos_vect = np.column_stack(np.unravel_index(max_idx, block_logpdf.shape[:-1]))
        #print("maxpos_vect.shape:",maxpos_vect.shape)
        #print("maxpos_vect:",maxpos_vect)

        Lmax = flatview[max_idx,range(len(max_idx))]
        #print("out shapes:",Lmax_simple.shape, flatview.shape, max_idx.shape, Lmax.shape)
        #print("agreement?", np.array_equal(Lmax_simple,Lmax))

        #H1_chi2_min = -2*Lmax

        # Get parameter values of the maximum
        pmax = {}
        for i,par in enumerate(block.deps):
            #print maxpos
            #print('par:',par,i)
            #print(maxpos_vect[:,i])
            pmax[par] = p1d[i][maxpos_vect[:,i]]
            #print("parmax {0} shape: {1}".format(par,pmax[par].shape))

        return Lmax, pmax

    def find_MLE_for_block_with_Minuit(self,block,options,block_x):
        """Find MLE for a single block of parameters using 'minuit'
           method. Mostly for internal use.
           Requires iminuit package: https://github.com/iminuit/iminuit
        """
        import iminuit
        #print("block.deps",block.deps)
        #print("options",options)
  
        # We need to filter out the options related to any parameters
        # that don't feature in this block.
        all_parameters = set(self.parameters)
        block_parameters = set(block.deps)
        non_block_parameters = all_parameters - block_parameters
        # Custom default options for minuit to prevent printing output to terminal
        # Will only use these if the user didn't provide them
        block_options = {'pedantic': False, 'print_level': -1}
        for key,val in options.items():
            words = key.split("_")
            if words[0] == 'error' or words[0] == 'fix':
               varname = '_'.join(words[1:])
               if varname in block_parameters:
                  block_options[key] = val # copy the option if it is the step size or 'fix' option for a block parameter
            elif key not in non_block_parameters:
                block_options[key] = val # copy the option if it is not a non-block parameter
        #print("block_parameters:", block_parameters)
        #print("non_block_parameters:", non_block_parameters)
        #print("block_options:", block_options)   

        maxpars = {}
        # Construct objective function
        ofunc = Objective(block, block_x)
        # Minimise it!
        try:
            m=iminuit.Minuit(ofunc,**block_options)
        except RuntimeError:
            six.reraise(RuntimeError,RuntimeError("Failed to run Minuit minimisation for {0} using options {1}".format(ofunc, block_options)))

        min_struct, par_struct_list = m.migrad(precision = 1e-5)
        if not min_struct.is_valid:
            msg = ""
            for key,val in min_struct.__dict__.items():
                msg += "  {0} : {1}\n".format(key,val)
            #raise ValueError("Minuit migrad failed to converge! Information from the run is as follows:\n{0}".format(msg))
        #print("m:",m)
        maxpars = m.values
        #print("m.values:", m.values)
        parvals = [maxpars[p] for p in block.deps]
        #print("Finished minimisation: evaluating function at best fit:",parvals)

        Lmax = -0.5 * ofunc(*parvals) # Objective is -2 * log(likelihood), but here we want to return just the log-likelihood

        return Lmax, maxpars
