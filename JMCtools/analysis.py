"""Classes related to defining an 'analysis', that is, a set of statistical
   tests to perform on a set of experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import scipy.interpolate as spi
from JMCtools.experiment import Experiment
from JMCtools.plotting import plot_teststat, power_plot
import JMCtools.common as c
import pandas as pd
import sys

# Don't let pandas do its weird line-wrapping when printing; just want one big table for output.
pd.set_option('display.expand_frame_repr', False)

def genNone():
    """Generator of infinite Nones"""
    while True:
        yield None

def repeat(thing):
    """Generator of infinite repeats"""
    while True:
        yield thing

def vflat(a):
    """Turns arrays into floats if they have only one element,
       and turns Nones into NaNs"""
    if a is None:
       r = np.nan
    else:
       b = np.atleast_1d(a)
       if b.shape==(1):
          r = np.float64(b[0])
       if np.squeeze(b).shape==():
          r = np.float64(np.squeeze(b))
       else:
          r = a
    return r

class Results:
    """Object to manage and display the results of statistical tests
       The central storage object is a Pandas dataframe.
    """
    def __init__(self,fields):
        self.results_df = pd.DataFrame(columns=fields)

    @classmethod  
    def fromDataFrame(cls,df):
        """Construct object from existing dataframe"""
        r = Results(list(df))
        r.add(df)
        return r

    def add(self,new_records):
        """Add the results from the analysis of an Experiment"""
        #print("Before adding new records:")
        #print(self)
        new_df = pd.DataFrame(new_records,columns=self.fields())
        self.results_df = pd.concat([self.results_df,new_df])
        #print("After adding new records:")
        #print(self)

    def add_column(self,name,data):
        """Add an entire new 'column' (i.e. field) to the dataset.
           Should have the same ordering as existing data, i.e.
           generally this should be information computed using
           the other fields in the dataset"""
        self.results_df[name] = pd.Series(data, index=self.results_df.index)

    def fields(self):
        """Return list of fields/column names"""
        return list(self.results_df)

    def __getitem__(self, key):
        """Extract a column from the dataset"""
        return self.results_df[key]

    def __str__(self):
        return self.results_df.__str__()

    def query(self,qstring):
        """Apply a boolean search filter to the results"""
        return Results.fromDataFrame(self.results_df.query(qstring))

    def to_latex(self):
        """Output results as LaTeX code for a table"""
        # Pandas dataframes already have a function for this, makes life easier
        return self.results_df.to_latex()

class Analysis:

    def __init__(self,experiments,tag,run_diagnostics=False,make_plots=True,Nproc=3):
        self.experiments = experiments
        for e in self.experiments:
            print(e)
            e.Nproc = Nproc # set number of processes to use for parallelisation of fits
        self.monster = Experiment.fromExperimentList(self.experiments)
        self.tag = tag # For naming output
  
        # Prepare object to store summary of results
        self._results = Results(["experiment","test","a_pval","e_pval","DOF"])
        #print("Results:")
        #print(self.results)

        # Disable diagnostics functions if desired
        if run_diagnostics is False or make_plots is False: # No point running diagnostics if we aren't emitting the plots
            for e in self.experiments:
                for t in e.tests.values():
                    t.diagnostics = None
        self.make_plots = make_plots

    def results(self,qstring=None):
        """Get results object, filtered using a query string (pandas format) if desired"""
        if qstring is None:
            r = self._results
        else:
            r = self._results.query(qstring)
        return r

    def simulate(self,Nsamples,test_type,true_parameters):
        """Perform pseudoexperiments, to generate pseudodata to be used in analysis
           functions for chosen statistical test.
           The test needs to be specified because not all experiments necessarily
           have every type of test defined, and so some will be automatically
           exclude from a certain kind of test."""
        pseudodata = []
        for e in self.experiments_for_test(test_type):
            pars = true_parameters[e.name]
            print("Generating {0} samples for experiment {1}, using parameters {2}".format(Nsamples,e.name,pars))
            pseudodata += [e.general_model.simulate(Nsamples,pars)]
        return pseudodata

    def experiments_for_test(self,test_type):
        """Filter the full experiment list for experiments with a certain
           kind of statistical test defined"""
        filtered_expts = []
        for e in self.experiments:
           test_defined = False
           try: 
              e.tests[test_type]
              test_defined = True
           except KeyError:
              # Doesn't have this test
              pass
           if test_defined:
              filtered_expts += [e]
        return filtered_expts

    def gof_analysis(self,test_parameters,pseudodata=None):
        """Perform goodness-of-fit tests on all experiments individually
           and jointly"""

        LLR_obs_monster_gof = 0
        LLR_monster_gof = 0
        monster_gofDOF = 0
        test_results = []
        if pseudodata is None:
            pseudodata = genNone() # generates Nones when iterated
        for j,(e,samples) in enumerate(zip(self.experiments_for_test('gof'),pseudodata)):
            # Inspect experiment (debugging)
            #print("Experiment {0} block structure: {1}".format(e.name, e.general_model.blocks))

            # Do fit!
            e_test_pars = test_parameters[e.name] # replace this with e.g. prediction from MSSM best fit
            print("Performing 'gof' test for experiment {0}, using null hypothesis {1}".format(e.name,e_test_pars),file=sys.stderr)
            model, LLR, LLR_obs, apval, epval, gofDOF = e.do_gof_test(e_test_pars,samples)
            # Save LLR for combining (only works if experiments have no common parameters)
            #print("e.name:{0}, LLR_obs:{1}, gofDOF: {2}".format(e.name,LLR_obs,gofDOF))
            if LLR is not None:
               LLR_monster_gof += LLR
            else:
               LLR_monster_gof = None
            monster_gofDOF += gofDOF
            LLR_obs_monster_gof += LLR_obs

            test_results += [ [e.name, "gof", vflat(apval), vflat(epval), gofDOF] ]

            # Plot! (only the first simulated 'observed' value, if more than one) 
            if self.make_plots:
                if apval is None:
                    print("p-value was None; test may be degenerate (e.g. if zero signal predicted), or just buggy. Skipping plot.",file=sys.stderr)
                    quit()
                else:
                    fig= plt.figure(figsize=(6,4))
                    ax = fig.add_subplot(111)
                    # Range for test statistic axis. Draw as far as is equivalent to 5 sigma
                    qran = [0, sps.chi2.ppf(sps.chi2.cdf(25,df=1),df=gofDOF)]  
                    plot_teststat(ax, LLR, lambda q: sps.chi2.pdf(q, gofDOF), log=True, 
                            label='free s', c='g', obs=LLR_obs, pval=apval[0], qran=qran, 
                             title=e.name+" (Nbins={0})".format(gofDOF))
                    ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                    fig.savefig('auto_experiment_{0}_{1}.png'.format(e.name,self.tag))
                    plt.close(fig)

        # Compute joint test results
        m_apval, m_epval = Experiment.chi2_pval(LLR_monster_gof,LLR_obs_monster_gof,monster_gofDOF)
        test_results += [ ["Monster", "gof", vflat(m_apval), vflat(m_epval), monster_gofDOF] ]

        # Save results
        self._results.add(test_results)

        # Plot! (only the first simulated 'observed' value, if more than one) 
        if self.make_plots:
            if m_apval is None:
                print("p-value was None; test may be degenerate (e.g. if zero signal predicted), or just buggy. Skipping plot.")
            else:
                fig= plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                # Range for test statistic axis. Draw as far as is equivalent to 5 sigma
                qran = [0, sps.chi2.ppf(sps.chi2.cdf(25,df=1),df=monster_gofDOF)]  
                plot_teststat(ax, LLR_monster_gof, lambda q: sps.chi2.pdf(q, monster_gofDOF), log=True, 
                        label='free s', c='g', obs=LLR_obs_monster_gof, pval=m_apval[0], qran=qran, 
                         title="Monster (Nbins={0})".format(monster_gofDOF))
                ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                fig.savefig('auto_experiment_{0}_{1}.png'.format("Monster",self.tag))
                plt.close(fig)

    def gof_analysis_dual(self,test_parameters,sb_pseudodata=None,b_pseudodata=None):
        """Perform goodness-of-fit tests on all experiments individually
           and jointly. This version MCs the distribution of the test statistics
           under both the background-only hypothesis AND the signal hypothesis.
           This allows us to also compute the power of the test to discover a
           particular signal. It also computes a meta-analysis combination
           of the p-values obtained from every experiment (using Fisher's method)
           and computes the power of that as well (this method may be more
           powerful than the likelihood-based combination, depending on the relative 
           numbers of degrees of freedom in the various likelihood components)."""

        LLR_obs_monster_gof = 0
        LLR_monster_gof = 0
        LLR_monster_gof_s = 0 # LLR samples under signal pseudodata
        monster_gofDOF = 0
        test_results = []
        if sb_pseudodata is None:
            sb_pseudodata = genNone() # generates Nones when iterated
        if b_pseudodata is None:
            b_pseudodata = genNone() # generates Nones when iterated
        # Storage for simulated p-values, for computing meta-analysis distribution and power
        N = len(b_pseudodata) # Number of experiments 
        M = b_pseudodata[0].shape[0] # Number of pseudodata trials
        all_epvals_b = np.ones((N,M))
        all_epvals_b_obs = [] # Observed p-value for each experiment
        N = len(sb_pseudodata)
        M = b_pseudodata[0].shape[0] 
        all_epvals_sb = np.ones((N,M))
        for j,(e,b_samples,s_samples) in enumerate(zip(self.experiments_for_test('gof'),b_pseudodata,sb_pseudodata)):
            # Inspect experiment (debugging)
            #print("Experiment {0} block structure: {1}".format(e.name, e.general_model.blocks))

            # Do fit!
            e_test_pars = test_parameters[e.name] # replace this with e.g. prediction from MSSM best fit
            print("Performing 'gof' test for experiment {0}, using null hypothesis {1}".format(e.name,e_test_pars),file=sys.stderr)
            model, LLR, LLR_obs, apval, epval, gofDOF = e.do_gof_test(e_test_pars,b_samples)
            # Save LLR for combining (only works if experiments have no common parameters)
            #print("e.name:{0}, LLR_obs:{1}, gofDOF: {2}".format(e.name,LLR_obs,gofDOF))
            if LLR is not None:
               LLR_monster_gof += LLR
            else:
               LLR_monster_gof = None
            monster_gofDOF += gofDOF
            LLR_obs_monster_gof += LLR_obs
            a = np.argsort(LLR)
            pvals = c.eCDF(LLR[a][::-1])[::-1] # do integral from right and then switch order back again
            all_epvals_b_obs += [epval]

            rCDF = spi.interp1d([-1e99]+list(LLR[a])+[1e99],[pvals[0]]+list(pvals)+[pvals[-1]]) # rather than 0/1, assign min/max observed pvalue to out-of-bounds
            all_epvals_b[j] = rCDF(LLR) 

            test_results += [ [e.name, "gof", vflat(apval), vflat(epval), gofDOF] ]

            print("Performing 'gof' test for experiment {0} with signal pseudodata".format(e.name),file=sys.stderr)
            model, s_LLR, s_LLR_obs, s_apval, s_epval, s_gofDOF = e.do_gof_test(e_test_pars,s_samples)
            # Save LLR for combining (only works if experiments have no common parameters)
            #print("e.name:{0}, LLR_obs:{1}, gofDOF: {2}".format(e.name,LLR_obs,gofDOF))
            if s_LLR is not None:
               LLR_monster_gof_s += s_LLR
            else:
               LLR_monster_gof_s = None

            # Ahh crap, I see my mistake! We don't want to compute these p-values based on
            # the *signal* simulated distribution! They are supposed to be p-values to
            # reject the *background* hypothesis! So we need them computed as if the
            # *background* hypothesis is true!
            #a = np.argsort(s_LLR)
            #all_epvals_sb[j,a] = 1 - c.eCDF(s_LLR[a])
            
            all_epvals_sb[j] = rCDF(s_LLR)
 
            # Plot! (only the first simulated 'observed' value, if more than one) 
            if self.make_plots:
                if apval is None:
                    print("p-value was None; test may be degenerate (e.g. if zero signal predicted), or just buggy. Skipping plot.",file=sys.stderr)
                    quit()
                else:
                    fig= plt.figure(figsize=(6,4))
                    ax = fig.add_subplot(111)
                    # Range for test statistic axis. Draw as far as is equivalent to 5 sigma
                    qran = [0, sps.chi2.ppf(sps.chi2.cdf(25,df=1),df=gofDOF)]  
                    if s_LLR is not None:
                        # Plot distribution under signal hypothesis
                        plot_teststat(ax, s_LLR, None, log=True, 
                            label='signal', c='r', obs=None, qran=qran)
                    # Plot distribution under background-only hypothesis
                    plot_teststat(ax, LLR, lambda q: sps.chi2.pdf(q, gofDOF), log=True, 
                            label='background-only', c='g', obs=LLR_obs, pval=apval[0], qran=qran, 
                            title=e.name+" (Nbins={0})".format(gofDOF),reverse_fill=True)

                    ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                    fig.savefig('auto_experiment_{0}_{1}_GOFdual.png'.format(e.name,self.tag))
                    plt.close(fig)

                    if LLR is not None and s_LLR is not None:
                         # Plot power of test to discover this signal hypothesis, vs CL level
                         fig = plt.figure(figsize=(6,4))
                         ax = fig.add_subplot(111)
                         power_plot(ax, LLR, s_LLR)
                         fig.savefig('auto_experiment_{0}_{1}_power.png'.format(e.name,self.tag))
                         plt.close(fig)

        # Compute joint test results
        m_apval, m_epval = Experiment.chi2_pval(LLR_monster_gof,LLR_obs_monster_gof,monster_gofDOF)
        test_results += [ ["Monster", "gof", vflat(m_apval), vflat(m_epval), monster_gofDOF] ]

        # Compute Fisher's method combination of results
        x = -2*np.sum(np.log(all_epvals_b),axis=0) # Sum over experiments
        DOF_fisher = 2*len(all_epvals_b)
        p_comb = 1 - sps.chi2.cdf(x,df=DOF_fisher)
        #sig_comb = -sps.norm.ppf(p_comb)

        # Observed:
        x_obs = -2*np.sum(np.log(all_epvals_b_obs)) # Sum over experiments
        p_obs = 1 - sps.chi2.cdf(x_obs,df=DOF_fisher)

        # Under signal hypothesis:
        x_s = -2*np.sum(np.log(all_epvals_sb),axis=0) # Sum over experiments
        p_comb_s = 1 - sps.chi2.cdf(x_s,df=DOF_fisher)
        #sig_comb_s = -sps.norm.ppf(p_comb_s)

        # Save results
        self._results.add(test_results)

        # Plot! (only the first simulated 'observed' value, if more than one) 
        if self.make_plots:
            if m_apval is None:
                print("p-value was None; test may be degenerate (e.g. if zero signal predicted), or just buggy. Skipping plot.")
            else:
                fig= plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                # Range for test statistic axis. Draw as far as is equivalent to 5 sigma
                qran = [0, sps.chi2.ppf(sps.chi2.cdf(25,df=1),df=monster_gofDOF)]  
                if s_LLR is not None:
                    # Plot distribution under signal hypothesis
                    plot_teststat(ax, LLR_monster_gof_s, None, log=True, 
                        label='signal', c='r', obs=None, qran=qran)
                plot_teststat(ax, LLR_monster_gof, lambda q: sps.chi2.pdf(q, monster_gofDOF), log=True, 
                        label='background-only', c='g', obs=LLR_obs_monster_gof, pval=m_apval[0], qran=qran, 
                         title="Monster (Nbins={0})".format(monster_gofDOF),reverse_fill=True)
                ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                fig.savefig('auto_experiment_{0}_{1}.png'.format("Monster",self.tag))
                plt.close(fig)
                if LLR_monster_gof is not None and LLR_monster_gof_s is not None:
                    # Plot distribution of meta-analysis test statistic
                    fig= plt.figure(figsize=(6,4))
                    ax = fig.add_subplot(111)
                    # Range for test statistic axis. Draw as far as is equivalent to 5 sigma
                    qran = [0, sps.chi2.ppf(sps.chi2.cdf(25,df=1),df=DOF_fisher)]  
                    if s_LLR is not None:
                        # Plot distribution under signal hypothesis
                        plot_teststat(ax, x_s, None, log=True, 
                            label='signal', c='r', obs=x_obs, qran=qran)
                    plot_teststat(ax, x, lambda q: sps.chi2.pdf(q, DOF_fisher), log=True, 
                            label='background-only', c='g', obs=x_obs, pval=p_obs, qran=qran, 
                             title="Monster (Fisher's method; DOF={0})".format(DOF_fisher),reverse_fill=True)
                    ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                    fig.savefig('auto_experiment_{0}_{1}_FisherComb.png'.format("Monster",self.tag))
                    plt.close(fig)
                     # Plot power of test to discover this signal hypothesis, vs CL level
                    fig = plt.figure(figsize=(6,4))
                    ax = fig.add_subplot(111)
                    power_plot(ax, LLR_monster_gof, LLR_monster_gof_s, label="Likelihood",c='g')
                    # Also plot power of meta-analysis combination to discover this signal hypothesis
                    power_plot(ax, x, x_s, label="Meta-analysis",c='m')
                    ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                    fig.savefig('auto_experiment_{0}_{1}_power.png'.format("Monster",self.tag))
                    plt.close(fig)
 

    def musb_analysis(self,test_parameters,pseudodata=None,nullmu=0,observed=None):
        """Perform mu=0 VS mu=1 tests on all experiments individually
           and jointly.
           'nullmu' parameter sets which value of mu is to be treated
           as the null hypothesis."""

        LLR_obs_monster_mmusb = 0 # Combined musb test (for Monster). Components are independent, I think.
        LLR_monster_mmusb = 0
        LLRA_monster = 0
        Eq_monster = 0
        Varq_monster = 0
        test_results = []
        if nullmu==0:
           reverse_fill = False
           c = 'b'
        else:
           reverse_fill = True
           c = 'r'
        if pseudodata is None:
            pseudodata = genNone() # generates Nones when iterated
        if observed is None:
            observed = genNone()
        for j,(e,samples,obs) in enumerate(zip(self.experiments_for_test('musb'),pseudodata,observed)):
            e_test_pars = test_parameters[e.name] # replace this with e.g. prediction from MSSM best fit
            print("Performing 'musb' test for experiment {0}, using 'signal shape' {1}".format(e.name, e_test_pars),file=sys.stderr) 
            model, musb_LLR, musb_LLR_obs, musb_apval, musb_epval, LLRA, Eq, Varq = e.do_musb_test(e_test_pars,samples,nullmu,observed=obs)
            if musb_LLR is not None:
               LLR_monster_mmusb += musb_LLR
            else:
               LLR_monster_mmusb = None
            LLR_obs_monster_mmusb += musb_LLR_obs
            LLRA_monster += LLRA
            Eq_monster += Eq
            Varq_monster += Varq

            test_results += [ [e.name, "musb_mu={0}".format(nullmu), vflat(musb_apval), vflat(musb_epval), 0] ]

            # Plot! (only the first simulated 'observed' value, if more than one)
            if self.make_plots:
                if musb_apval is None:
                    print("p-value was None; test may be degenerate (e.g. if zero signal predicted), or just buggy. Skipping plot.",file=sys.stderr)
                else:
                    fig= plt.figure(figsize=(6,4))
                    ax = fig.add_subplot(111)
                    qran = (Eq - 5*np.sqrt(Varq), Eq + 5*np.sqrt(Varq)) # asymptotic 5 sigma-ish range
                    plot_teststat(ax, musb_LLR, lambda q: sps.norm.pdf(q, loc=Eq, scale=np.sqrt(Varq)), log=True,
                            label='mu', c=c, obs=musb_LLR_obs[0], pval=musb_apval[0], title=e.name, qran=qran, reverse_fill=reverse_fill)
                    ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                    fig.savefig('auto_experiment_musb_mu={0}_{1}_{2}.png'.format(nullmu,e.name,self.tag))
                    plt.close(fig)

        # Compute joint test results
        m_apval, m_epval, m_Eq, m_Varq = Experiment.sb_pval(LLR_monster_mmusb,
                                                            LLR_obs_monster_mmusb,
                                                            LLRA_monster,nullmu=nullmu)
        test_results += [ ["Monster", "musb_mu={0}".format(nullmu), vflat(m_apval), vflat(m_epval), 0] ]

        # Plot Monster results
        if self.make_plots:
            if m_apval is None:
                print("p-value was None; test may be degenerate (e.g. if zero signal predicted), or just buggy. Skipping plot.")
            else:
                fig= plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                qran = (m_Eq - 5*np.sqrt(m_Varq), m_Eq + 5*np.sqrt(m_Varq)) # asymptotic 5 sigma-ish range
                plot_teststat(ax, LLR_monster_mmusb, lambda q: sps.norm.pdf(q, loc=m_Eq, scale=np.sqrt(m_Varq)), log=True,
                        label='mu', c=c, obs=LLR_obs_monster_mmusb[0], pval=m_apval[0], title="Monster", qran=qran, reverse_fill=reverse_fill)
                ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                fig.savefig('auto_experiment_musb_mu={0}_{1}_{2}.png'.format(nullmu,"Monster",self.tag))
                plt.close(fig)

        # Save results
        self._results.add(test_results)

    def musb_analysis_dual(self,test_parameters,sb_pseudodata=None,b_pseudodata=None):
        """Perform mu=0 VS mu=1 tests on all experiments individually
           and jointly.
           Tests both mu=0 and mu=1 null hypotheses, plots both distributions together,
           and also computes CL_s values""" 

        LLR_obs_monster_mmusb = 0 # Combined musb test (for Monster). Components are independent, I think.

        LLRsb_monster_mmusb = 0
        LLRAsb_monster = 0
        Eqsb_monster = 0
        Varqsb_monster = 0

        LLRb_monster_mmusb = 0
        LLRAb_monster = 0
        Eqb_monster = 0
        Varqb_monster = 0

        test_results = []

        if sb_pseudodata is None and b_pseudodata is None:
            sb_pseudodata = genNone() # generates Nones when iterated
            b_pseudodata  = genNone()

        for j,(e,b_samples,sb_samples) in enumerate(zip(self.experiments_for_test('musb'),b_pseudodata,sb_pseudodata)):
            e_test_pars = test_parameters[e.name] # replace this with e.g. prediction from MSSM best fit

            print("Performing 'musb' test (mu=1) for experiment {0}, using 'signal shape' {1}".format(e.name, e_test_pars),file=sys.stderr) 
            model, musb_LLRsb, musb_LLR_obs, musb_apvalsb, musb_epvalsb, LLRAsb, Eqsb, Varqsb = e.do_musb_test(e_test_pars,sb_samples,nullmu=1)
            if musb_LLRsb is not None:
               LLRsb_monster_mmusb += musb_LLRsb
            else:
               LLRsb_monster_mmusb = None
            LLR_obs_monster_mmusb += musb_LLR_obs
            LLRAsb_monster += LLRAsb
            Eqsb_monster += Eqsb
            Varqsb_monster += Varqsb

            test_results += [ [e.name, "musb_mu=1", vflat(musb_apvalsb), vflat(musb_epvalsb), 0] ]

            print("Performing 'musb' test (mu=0) for experiment {0}, using 'signal shape' {1}".format(e.name, e_test_pars),file=sys.stderr) 
            model, musb_LLRb, musb_LLR_obs, musb_apvalb, musb_epvalb, LLRAb, Eqb, Varqb = e.do_musb_test(e_test_pars,b_samples,nullmu=0)
            if musb_LLRb is not None:
               LLRb_monster_mmusb += musb_LLRb
            else:
               LLRb_monster_mmusb = None
            #LLR_obs_monster_mmusb += musb_LLR_obs # already did this, observed LLR is same in both tests
            LLRAb_monster += LLRAb
            Eqb_monster += Eqb
            Varqb_monster += Varqb

            if musb_apvalb is not None and len(musb_apvalb)==1: musb_apvalb=musb_apvalb[0]
            if musb_epvalb is not None and len(musb_epvalb)==1: musb_epvalb=musb_epvalb[0]
            test_results += [ [e.name, "musb_mu=0", vflat(musb_apvalb), vflat(musb_epvalb), 0] ]

            # CL_s (Tevatron style)
            if musb_apvalsb is not None and musb_apvalb is not None:
               a_CLs = musb_apvalsb / (1 - musb_apvalb)
            else:
               a_CLs = None
            if musb_epvalsb is not None and musb_epvalb is not None:
               e_CLs = musb_epvalsb / (1 - musb_epvalb)
            else:
               e_CLs = None
            test_results += [ [e.name, "musb_CLs", vflat(a_CLs), vflat(e_CLs), 0] ]

            # Extract single value for pvalue (in case of multiple "observed" data realisations)
            # Just use first one. TODO: probably better to make a different kind of plot if multiple
            # p-values computed at once.
            apvalsb = np.atleast_1d(musb_apvalsb)[0]
            apvalb = np.atleast_1d(musb_apvalb)[0]

            # Plot!
            if self.make_plots:
                fig= plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                qran = (Eqsb - 5*np.sqrt(Varqsb), Eqb + 5*np.sqrt(Varqb)) # cover asymptotic 5 sigma-ish range of both distributions 
                # Plot s+b distribution
                plot_teststat(ax, musb_LLRsb, lambda q: sps.norm.pdf(q, loc=Eqsb, scale=np.sqrt(Varqsb)), log=True,
                        label='mu=1', c='r', obs=musb_LLR_obs, pval=apvalsb, title=e.name, qran=qran, reverse_fill=True)
                # Plot b distribution
                plot_teststat(ax, musb_LLRb, lambda q: sps.norm.pdf(q, loc=Eqb, scale=np.sqrt(Varqb)), log=True,
                        label='mu=0', c='b', obs=musb_LLR_obs, pval=apvalb, title=e.name, qran=qran, reverse_fill=False)
                ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                fig.savefig('auto_experiment_musb_dual_{0}_{1}.png'.format(e.name,self.tag))
                plt.close(fig)

                # Non-log axis
                fig= plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                plot_teststat(ax, musb_LLRsb, lambda q: sps.norm.pdf(q, loc=Eqsb, scale=np.sqrt(Varqsb)), log=False,
                        label='mu=1', c='r', obs=musb_LLR_obs, pval=apvalsb, title=e.name, qran=qran, reverse_fill=True)
                plot_teststat(ax, musb_LLRb, lambda q: sps.norm.pdf(q, loc=Eqb, scale=np.sqrt(Varqb)), log=False,
                        label='mu=0', c='b', obs=musb_LLR_obs, pval=apvalb, title=e.name, qran=qran, reverse_fill=False)
                ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
                fig.savefig('auto_experiment_musb_dual_{0}_{1}_nonlog.png'.format(e.name,self.tag))
                plt.close(fig)

                # TODO: fix
                #if musb_LLRb is not None and musb_LLRsb is not None:
                #    # Plot power of test to discover this signal hypothesis, vs CL level
                #    fig = plt.figure(figsize=(6,4))
                #    ax = fig.add_subplot(111)
                #    power_plot(ax, musb_LLRb, musb_LLRsb,left_tail=True)
                #    fig.savefig('auto_experiment_musb_dual_{0}_{1}_power.png'.format(e.name,self.tag))
                #    plt.close(fig)

        # Compute joint test results
        m_apvalsb, m_epvalsb, m_Eqsb, m_Varqsb = Experiment.sb_pval(LLRsb_monster_mmusb,
                                                            LLR_obs_monster_mmusb,
                                                            LLRAsb_monster,nullmu=1)
        test_results += [ ["Monster", "musb_mu=1", vflat(m_apvalsb), vflat(m_epvalsb), 0] ]

        m_apvalb, m_epvalb, m_Eqb, m_Varqb = Experiment.sb_pval(LLRb_monster_mmusb,
                                                            LLR_obs_monster_mmusb,
                                                            LLRAb_monster,nullmu=0)
        test_results += [ ["Monster", "musb_mu=0", vflat(m_apvalb), vflat(m_epvalb), 0] ]

        # CL_s (Tevatron style)
        if m_apvalsb is not None and m_apvalb is not None:
            a_CLs = m_apvalsb / (1 - m_apvalb)
        else:
            a_CLs = None
        if m_epvalsb is not None and m_epvalb is not None:
            e_CLs = m_epvalsb / (1 - m_epvalb)
        else:
            e_CLs = None
        test_results += [ ["Monster", "musb_CLs", vflat(a_CLs), vflat(e_CLs), 0] ]

        apvalsb = np.atleast_1d(m_apvalsb)[0]
        apvalb = np.atleast_1d(m_apvalb)[0]

        # Plot Monster results 
        if self.make_plots:
            fig= plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            qran = (m_Eqsb - 5*np.sqrt(m_Varqsb), m_Eqb + 5*np.sqrt(m_Varqb)) # asymptotic 5 sigma-ish range
            plot_teststat(ax, LLRsb_monster_mmusb, lambda q: sps.norm.pdf(q, loc=m_Eqsb, scale=np.sqrt(m_Varqsb)), log=True,
                    label='mu=1', c='r', obs=LLR_obs_monster_mmusb, pval=apvalsb, title="Monster", qran=qran, reverse_fill=True)
            plot_teststat(ax, LLRb_monster_mmusb, lambda q: sps.norm.pdf(q, loc=m_Eqb, scale=np.sqrt(m_Varqb)), log=True,
                    label='mu=0', c='b', obs=LLR_obs_monster_mmusb, pval=apvalb, title="Monster", qran=qran, reverse_fill=False)
            ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
            fig.savefig('auto_experiment_musb_dual_{0}_{1}.png'.format("Monster",self.tag))
            plt.close(fig)

            # Non-log axis
            fig= plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            plot_teststat(ax, LLRsb_monster_mmusb, lambda q: sps.norm.pdf(q, loc=m_Eqsb, scale=np.sqrt(m_Varqsb)), log=False,
                    label='mu=1', c='r', obs=LLR_obs_monster_mmusb, pval=apvalsb, title="Monster", qran=qran, reverse_fill=True)
            plot_teststat(ax, LLRb_monster_mmusb, lambda q: sps.norm.pdf(q, loc=m_Eqb, scale=np.sqrt(m_Varqb)), log=False,
                    label='mu=0', c='b', obs=LLR_obs_monster_mmusb, pval=apvalb, title="Monster", qran=qran, reverse_fill=False)
            ax.legend(loc=1, frameon=False, framealpha=0,prop={'size':10})
            fig.savefig('auto_experiment_musb_dual_{0}_{1}_nonlog.png'.format("Monster",self.tag))
            plt.close(fig)

        # Save results
        self._results.add(test_results)

