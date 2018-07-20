"""Just some helpful functions for doing common plots"""

import numpy as np
 
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
       yran = (1e-4,np.max([minupy,np.max(n[np.isfinite(n)])]))
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
        pval_str = ""
        if pval is not None:
           #print("pval:", pval)
           pval_str = " (p={0:.2g})".format(pval)
        ax.axvline(x=obs,lw=2,c=c,label="Observed ({0}){1}".format(label,pval_str))
    ax.set_xlim(ran[0],ran[1])
    ax.set_ylim(yran[0],yran[1])
    if title is not None:
        ax.set_title(title)
 
