import JMCtools as jt
import JMCtools.distributions as jtd
import numpy as np
import inspect
import JMCtools.common as c
import concurrent.futures
import itertools
import time
   
class Objective:
    """Constructs the objective function to be fitted my iminuit in the 
       'minuit' fitting method of ParameterModel.getMLEs.
       Inspired by example here:
       http://iminuit.readthedocs.io/en/latest/api.html#function-signature-extraction-ordering
    """
    def __init__(self, block, x):
        import iminuit
        self.block = block
        self.data, self.datasize = x 
        args = list(self.block.deps) # parameters required by the block, and thus this objective function
        #print('Objective function constructed with args:',args)
        #print('Data is:',self.data)
        self.func_code = iminuit.Struct(
                co_varnames = args,
                co_argcount = len(args)
                )

    def __call__(self, *arg):
        parameters = {}
        for name,val in zip(self.block.deps,arg):
            V = np.atleast_1d(val)
            parameters[name] = V.reshape(V.shape + tuple([1]*len(self.datasize))) # Add new singleton axes to broadcast against data 
        block_logpdf_all = self.block.logpdf(self.data, parameters)
        block_logpdf = np.sum(block_logpdf_all,axis=-1) 
        return -2*block_logpdf
 
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
        return self.jointmodel.logpdf(x,self.get_pdf_args(parameters))

# Functions to be run by worker processes in ParameterModel.find_MLE_parallel
# Slightly different approach used depending on the minimisation
# method chosen.
def loop_func_grid(par_object,options,i,data_slice):
   #data_slice = c.get_data_slice(data,i*chunksize,i*chunksize+slicesize)  
   Lmax_chunk, pmax_chunk = par_object.find_MLE(options,method='grid',x=data_slice)
   return i, Lmax_chunk, pmax_chunk

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
    Lmax_chunk, pmax_chunk = par_object.find_MLE(options,method='minuit',x=data_slice)
    return i, Lmax_chunk, pmax_chunk

class ChunkedData:
    """A wrapper for ParameterModel data which turns it into a
       generator, so that we can efficiently iterate through the
       data
    """
    def __init__(self,x,chunksize):
        self.x = x
        self.data, self.size = self.x
        self.chunksize = chunksize
        self.Ntrials, self.Ndraws = self.size
        self.Nchunks = self.Ntrials // chunksize
        self.Nremainder = self.Ntrials % chunksize
        self.i = 0 # Index of next data chunk to be returned
    def __iter__(self):
        return self
    def __next__(self):
        self.i += 1
        if self.i == self.Nchunks:
            raise StopIteration  # Done iterating.
        slicesize = self.chunksize
        if self.i==self.Nchunks and self.Nremainder!=0:
            slicesize = self.Nremainder
        # Extract the data slice
        return c.get_data_slice(self.x,self.i*self.chunksize,self.i*self.chunksize+slicesize)

class ParameterModel:     
    """An object for managing the mapping between a parameter space and
       a set of distribution function, i.e. a JointModel object.
       Also has methods for computing common statistics, such as
       maximum likelihood estimators"""

    def __init__(self,jointmodel,parameter_funcs,x=None):
        """Note: once this object is constructed, it is best if you
           don't mess around with the internals of the stored JointModels
           and so on, or else you might screw up the internal
           consistency of the member routines in this object"""
        self.model = jointmodel
        self.parfs = parameter_funcs
        # Get parameter dependencies of each block
        #self.submodel_deps = [f.__code__.co_varnames for f in self.parfs] # seems to get other variables too
        self.submodel_deps = [inspect.getargspec(f)[0] for f in self.parfs] 
        #print(self.submodel_deps)
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
        #print(self.parfs)
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
        for deps,parf in zip(self.submodel_deps,self.parfs):
            pdict = {p: parameters[p] for p in deps} # extract just the parameters that this parf needs
            args += [parf(**pdict)]
        return args

    def logpdf(self,parameters,x=None):
        if x==None:
           x = self.x # Use pre-generated data if none provided
        else:
           self.validate_data(x)
        return self.model.logpdf(x[0],self.get_pdf_args(parameters))

    def find_submodels_which_depend_on(self,parameter):
        """Return the indices of the submodels which depend on 'parameter'"""
        matches = set([])
        for i,d in enumerate(self.submodel_deps):
            if parameter in d:
                matches.add(i)
        return matches

    def validate_data(self,x):
        data,size = x
        if len(data)!=self.model.N_submodels:
           raise ValueError("The length of the supplied list of data values does not match the number of submodels in the wrapped JointModel! This means that this data cannot possibly be generated by (or used to compute the pdf of) this model and so is invalid.")

    def set_data(self,x):
        """Set internally-stored data realisations to use for parameter 
           estimation etc."""
        self.validate_data(x)
        self.x = x

    def simulate(self,size,null_parameters=None):
        """Generate 'size' simulated datasets to use for parameter estimation etc.
           null_parameters - parameters to use for the simulation.
             If None, hopes that 'model' is frozen with some pre-set parameters.
           Note: shape of N has importance! Last dimension will be treated as 
           the number of events *per trial*. Other dimensions will be the number
           of trials (and may be any shape that is convenient). So, for example,
              size = (1000,10)
           would mean 'Do 1000 trials, with 10 independent draws per trial.'
              size = (100, 100, 10)
           would mean 'Do (100,100) trials, with 10 independent draws per trial'
           Actually, the full (10,100,100) events would be returned from this
           function, it is more a matter of how the find_MLE function will
           interpret this array of events.
        """
        if null_parameters!=None:
           args = self.get_pdf_args(null_parameters)
        else:
           args = {}
        #print(args)
        self.x = self.model.rvs(size,args), size # keep record of the data shape, makes life easier
        # Might as well return the data as well
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
        if x==None:
           x = self.x # Use pre-generated data if none provided
        else:
           self.validate_data(x)
        data,size = x
        
        #print('in find_mle:, data=',data)
        Lmax_tot = 0 # output maximum log-likelihood
        MLE_pars_full = {}
        for block in self.blocks:
           block_data = [data[i] for i in block.submodels] # Select data relevant to this submodel
           #print('in find_mle:, block_data=',block_data)
           Lmax, MLE_pars = self.find_MLE_for_block(block,options,(block_data,size),method)
           Lmax_tot += Lmax
           MLE_pars_full.update(MLE_pars)

        return Lmax_tot, MLE_pars_full

    def find_MLE_parallel(self,options,x=None,method='grid',chunksize=100,Nprocesses=None,seeds=None):
        """Find MLEs for a large number of sample realisations, by breaking the samples into
           chunks and evaluating them in parallel.
           Note: This currently only works if the data has underlying shape (Ntrials,Ndraws), where we
           will chunk over Ntrials, and Ndraws is treated as independent draws from our pdf
           which should be automatically combined into a joint pdf.
        """
        if x==None:
           x = self.x # Use pre-generated data if none provided
        else:
           self.validate_data(x)
        data,size = x

        Ntrials, Ndraws = size
        pmax_all = {p: np.zeros(Ntrials) for p in self.parameters} # This is a bit wasteful if there are lots of fixed parameters, but oh well.
        Lmax_all = np.zeros(Ntrials)

        starttime = time.time()
 
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
                    print("Getting {0} MLEs for chunk {1} of {2}           ".format(method,i,Nchunks), end="\r")
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
                def seed(options,i):
                    for p,val in seeds.items():
                        if p not in fixed:
                            options[p] = val[i]
                    return options
                opts = map(seed,itertools.repeat(options),range(Ntrials))
            else:
                opts = itertools.repeat(options)

            with concurrent.futures.ProcessPoolExecutor(Nprocesses) as executor:
                for i, Lmax, pmax in executor.map(loopfunc, itertools.repeat(self), 
                                                           opts, 
                                                           range(Ntrials), 
                                                           ChunkedData(x,1), 
                                                 chunksize=chunksize):
                   if i % chunksize==0: print("Getting {0} MLEs for chunk {1} of {2}              ".format(method,i//chunksize,Ntrials//chunksize), end="\r")
                   Lmax_all[i] = Lmax
                   for p,val in pmax_all.items():
                      val[i] = pmax[p]
        else:
            ValueError("Unrecognised minimisation method selected!")

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
        data,size = block_x
        #print("block_x:", block_x)

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
        #print('PARS[0].shape:',PARS[0].shape)
        parameters = {}
        for i,par in enumerate(block.deps):
           parameters[par] = PARS[i].reshape(PARS[i].shape + tuple([1]*len(size))) # Add new singleton axes to broadcast against data
           #print('parameters[{0}].shape: {1}'.format(par,parameters[par].shape))
 
        block_logpdf_all = block.logpdf(data,parameters)
        #print("block_logpdf_all.shape:", block_logpdf_all.shape)
 
        # We now want to interpret the last dimension of data as 'events per trial',
        # So we compute the joint logpdf as the sum over the last dimension of this output
        block_logpdf = np.sum(block_logpdf_all,axis=-1) 

        # Maximise over all dimensions except the data dimension

        # Use this to check result of reshape/indexing manipulations
        #print("flatten axes:",tuple(range(len(block.deps))))
        Lmax_simple = np.max(block_logpdf,axis=tuple(range(len(block.deps))))

        # During reshape; keep size of last dimension, 'flatten' the rest, then maximise over the flattened part
        # No idea if the ordering is correctly presevered here...
        #max_idx = np.swapaxes(block_logpdf,0,-1).reshape((block_logpdf.shape[-1],-1)).argmax(-1)
        flatview = c.almost_flatten(block_logpdf)
        max_idx = flatview.argmax(axis=0)
        #print("PARS[0].shape, data[0].shape:", PARS[0].shape, data[0].shape)
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
        block_options = {}
        for key,val in options.items():
            words = key.split("_")
            if all(w not in non_block_parameters for w in words):
                block_options[key] = val # copy the option if unrelated to any non-block parameters

        data,size = block_x
        Ntrials = np.sum(size[:-1]) # all data dimensions aside from the last are considered as independent trials.
        Lmax = -1e99*np.ones(Ntrials)
        maxpars = {p: np.zeros(Ntrials) for p in block.deps}
        for i in range(Ntrials):
            # Extract ith trial from the data
            #print("block_x:",block_x)
            block_x_i = c.get_data_slice(block_x,i)
            #print("block_x_i:",block_x_i)
            # Construct objective function
            ofunc = Objective(block, block_x_i)
            # Minimise it!
            m=iminuit.Minuit(ofunc,**block_options)
            m.migrad(precision = 1e-5)
            #print(m)
            pars = m.values
            parvals = [pars[p] for p in block.deps]
            for p in block.deps:
                maxpars[p][i] = pars[p]
            Lmax[i] = -0.5 * ofunc(*parvals) # Objective is -2 * log(likelihood), but here we want to return just the log-likelihood

        return Lmax, maxpars
