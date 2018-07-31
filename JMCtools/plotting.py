"""Just some helpful functions for doing common plots"""

import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps
import JMCtools.common as common

def plot_teststat(ax, tobin, theoryf, log=True, label="", c='r', obs=None, pval=None,
             qran=None, title=None, reverse_fill=False):
    """Bin up a test statistic and plot it against a theoretical distribution"""
    print("Generating test statistic plot {0}".format(label))
    if qran is None:
        ran = (0,25)
    else:
        ran = qran
    yran = (1e-4,0.5)
    if tobin is not None:
       m = np.isfinite(tobin)
       nsamp = np.sum((tobin[m]>ran[0]) & (tobin[m]<ran[1]))
       #print("Number of samples in selected range: {0}".format(nsamp))
       if nsamp == 0:
           # No LLR samples in chosen binning range, so extend it
           print("Warning: no LLR samples found in chosen binning range for plot; extending plot domain.")
           ran = [None,None]
           ran[0] = np.min([0,np.min(tobin[m])])
           ran[1] = np.max(tobin[m])
       #print("tobin:",tobin)
       n, bins = np.histogram(tobin, bins=50, normed=True, range=ran)
       #print("n:",n)
       ax.plot(bins[:-1],n,drawstyle='steps-post',label=label,c=c)
       if log:
           minupy = 0.5
       else:
           minupy = 1e-4
       yran = (1e-4,np.max([minupy,1.2*np.max(n[np.isfinite(n)])]))
       #print("Histogram y range:", yran)
    q = np.arange(ran[0],ran[1],0.01)
    if theoryf is not None:
        ax.plot(q, theoryf(q),c='k')
    ax.set_xlabel("LLR")
    ax.set_ylabel("pdf(LLR)")
    if log:
        #ax.set_ylim(np.min(n[n!=0]),10*np.max(n))
        ax.set_yscale("log")     
    if obs is not None:
        # Draw line for observed value, and show p-value region shaded
        if theoryf!=None:
           if reverse_fill:
              # Sometimes p-value is computed using the other tail of the distribution.
              qfill = np.arange(obs,ran[1],0.01)
           else:
              qfill = np.arange(ran[0],obs,0.01)
           ax.fill_between(qfill, 0, theoryf(qfill), lw=0, facecolor=c, alpha=0.2)
        pval_str = None
        if pval is not None:
           #print("pval:", pval)
           pval_str = "Observed (p={0:.2g})".format(pval)
        ax.axvline(x=obs,lw=2,c=c,label=pval_str)
    ax.set_xlim(ran[0],ran[1])
    ax.set_ylim(yran[0],yran[1])
    if title is not None:
        ax.set_title(title)
 
def power_plot(ax, bq, sq, label=None, c='r', left_tail=False):
    """Plot of the statistical power of a test vs exclusion significance
    That is, it is a plot of alpha (probability to exclude null hypothesis
    when null hypothesis is true)
    against 1-beta (probability to accept alternate hypothesis when alternate
    is true)
    
    Assumes bq is chi-squared distributed (i.e. alpha is tail integral to the right)
    """

    # So, first we need to know the test-statistic thresholds that correspond
    # to certain significance levels (alphas)
    # We'll just do it directly in significance levels I guess, physicists
    # will understand that more easily.
    if left_tail:
        # This order for L_sb/L_b (left tail integral)
        bq_sort = np.sort(bq)
        sq_sort = np.sort(sq)
        bq_range = [-1e99]+list(bq_sort)+[1e99]
        # CDF from left
        eCDF_alpha = common.eCDF(bq_sort) 
        # input alpha, get back corresponding q
        q_alpha = spi.interp1d([0]+list(eCDF_alpha)+[1],bq_range)
        # Now we need beta (right tail integral; power is 1-beta)
        eCDF_beta = 1 - common.eCDF(sq_sort)
        # input q, get back corresponding beta
        beta = spi.interp1d(([-1e99]+list(sq_sort)+[1e99]),([0]+list(eCDF_beta)+[1]))
    else:
        # This order for chi-square (right tail integral)
        bq_sort = np.sort(bq)[::-1]
        sq_sort = np.sort(sq)
        bq_range = [1e99]+list(bq_sort)+[-1e99]
        # CDF from right
        eCDF_alpha = common.eCDF(bq_sort) 
        # input alpha, get back corresponding q
        q_alpha = spi.interp1d([0]+list(eCDF_alpha)+[1],bq_range)
        # Now we need beta (left tail integral; power is 1-beta)
        eCDF_beta = common.eCDF(sq_sort)
        # input q, get back corresponding beta
        beta = spi.interp1d(([-1e99]+list(sq_sort)+[1e99]),([0]+list(eCDF_beta)+[1]))
    # Interpolate
    ai = np.logspace(-5,0,1000)
    qi = q_alpha(ai)
    power = 1-beta(qi)
    sig = -sps.norm.ppf(ai)
    # plot probability to accept alternate when null is true (Type I error (false positive) rate)
    ax.plot(sig,ai,drawstyle='steps-post',c='k') 
    # power
    ax.plot(sig,power,drawstyle='steps-post',c=c,label=label)
    ax.set_xlabel("Significance threshold (sigma)")
    ax.set_ylabel("Power (1-beta)")
    ax.set_xlim(0,5)
    #ax.set_xscale("log")     
 
