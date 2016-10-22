from scipy.stats import probplot as qq_plot
from scipy.stats import ttest_1samp, shapiro, chi2_contingency, wilcoxon, mannwhitneyu
from scipy.stats import ttest_ind, ttest_rel, fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import samplesize_confint_proportion
from scipy.stats import pearsonr, spearmanr, kstest, ks_2samp, chisquare
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats
import itertools
import numpy as np
import statsmodels.stats.api as sms

def get_z(mu,mu_0,sigma,n):
    z = (mu-mu_0)/(sigma/np.sqrt(n))
    return z

def mean_diff_confint_ind(sample1,sample2):
    cm = CompareMeans(DescrStatsW(sample1), DescrStatsW(sample2))
    return cm.tconfint_diff()

def mean_diff_confint_rel(sample1,sample2):
    return DescrStatsW(sample1 - sample2).tconfint_mean()

def prop_test(sample,p_0,alternative='two-sided'):
    p = sample.mean()
    n = len(sample)
    se = np.sqrt(p*(1-p)/n)
    z = (p - p_0)/se
    return get_norm_p(z,alternative=alternative)

def prop_confint(sample,method='normal'):
    return proportion_confint(sum(sample),len(sample),method=method)

def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

def bootstrap_conf_int(data,stat_func,alpha=0.05,n_samples=1000):
    ''' 
    a = np.random.normal(size=1000)
    conf_int(a,np.median)
    '''
    scores = [stat_func(sample) for sample in get_bootstrap_samples(data,n_samples)]
    return stat_intervals(scores, alpha)

def bootstrap_test(sample, param, stat_func, n_samples = 1000, alternative = 'two-sided'):
    
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
      
    param_d = [stat_func(i) for i in get_bootstrap_samples(sample,n_samples)]
    
    mean_p = np.mean(param_d)
    t_stat = stat_func(sample) - param
    
    zero_dist = [(mm - mean_p) for mm in param_d]
       
    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_dist]) / len(zero_dist)
    
    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_dist]) / len(zero_dist)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_dist]) / len(zero_dist)

def bootstrap_diff_conf_int(a,b,stat_func,alpha=0.05,n_samples=1000):
    '''
    a = np.random.normal(size=1000)
    b = np.random.normal(loc=2,size=1000)
    diff_conf_int(b,a,np.median)
    '''
    scores_a = [stat_func(sample) for sample in get_bootstrap_samples(a,n_samples)]
    scores_b = [stat_func(sample) for sample in get_bootstrap_samples(b,n_samples)]
    delta_scores = [x[0] - x[1] for x in zip(scores_a,scores_b)]
    return stat_intervals(delta_scores, alpha)

def vcramer(table):
    chi, p, _, _ = stats.chi2_contingency(table,correction=False)
    n = table.sum()
    r,c = table.shape
    return np.sqrt(chi/(n*(min(r,c)-1.))), p

def mcc(a,b,c,d):
    '''
    Matthews correlation from contigency table
    '''
    return (a*d - b*c) / np.sqrt((a+b)*(a+c)*(b+d)*(c+d))

def get_norm_p(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    if alternative == 'less':
        return stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - stats.norm.cdf(z_stat)

def get_t_p(z_stat, n, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - stats.t.cdf(np.abs(z_stat),df=(n-1)))
        
    if alternative == 'less':
        return stats.t.cdf(np.abs(z_stat),df=(n-1))

    if alternative == 'greater':
        return 1 - stats.t.cdf(np.abs(z_stat),df=(n-1))
    
def proportions_diff_ind(p1,n1,p2,n2,alternative = 'two-sided'):
    '''
    AB test
    '''
    P = float(p1*n1+p2*n2)/(n1+n2)
    z = (p1-p2)/np.sqrt(P*(1-P)*(1./n1+1./n2))
    return get_norm_p(z,alternative)

def proportions_diff_ind_table(table,alternative = 'two-sided'):
    '''
    AB test from contigency table
    a, b, c, d = tables.values.ravel()
    '''
    a,b,c,d = table.ravel()
    n1, n2 = a+c, b+d
    p1, p2 = float(a)/n1, float(b)/n2
    return proportions_diff_ind(p1,n1,p2,n2,alternative)

def proportions_diff_ind_samples(sample1,sample2,alternative = 'two-sided'):
    '''
    AB test from samples
    '''
    n1 = len(sample1)
    n2 = len(sample2)
    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2 
    return proportions_diff_ind(p1,n1,p2,n2,alternative)

def proportions_confint_diff_ind(p1,n1,p2,n2, alpha = 0.05):
    '''
    confidence interval for proportion difference from ps and ns
    '''
    z = stats.norm.ppf(1 - alpha / 2.)   
    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)
    return (left_boundary, right_boundary)

def proportions_confint_diff_ind_table(table,alpha = 0.05):
    '''
    confidence interval for proportion difference from contigency table
    '''
    a,b,c,d = table.ravel()
    n1, n2 = a+c, b+d
    p1, p2 = float(a)/n1, float(b)/n2
    return proportions_confint_diff_ind(p1,n1,p2,n2, alpha)

def proportions_confint_diff_ind_samples(sample1,sample2, alpha = 0.05):
    '''
    confidence interval for proportion difference from samples
    '''
    n1 = len(sample1)
    n2 = len(sample2)
    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2 
    return proportions_confint_diff_ind(p1,n1,p2,n2, alpha)

def get_props_and_lens(table,invertion=True):
    a,b,c,d = table.values[::-1,:].ravel()
    n1, n2 = a+c, b+d
    p1, p2 = float(a)/n1, float(b)/n2
    return p1, n1, p2, n2

#----------------------------------------------------------------

def proportions_confint_diff_rel(sample1, sample2, alpha = 0.05):
    z = stats.norm.ppf(1 - alpha / 2.)
    sample = zip(sample1, sample2)
    n = len(sample)
        
    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])
    
    left_boundary = float(f - g) / n  - z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    right_boundary = float(f - g) / n  + z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    return (left_boundary, right_boundary)


def proportions_diff_rel(sample1, sample2, alternative = 'two-sided'):
    sample = zip(sample1, sample2)
    n = len(sample)
    
    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])
    
    z = float(f - g) / np.sqrt(f + g - float((f - g)**2) / n )
    
    return get_norm_p(z,alternative)

def two_proportions_sample_size(p1,p2,alpha=0.05,power=0.8,frac=0.5):
    ratio = frac/(1.-frac)
    es = sms.proportion_effectsize(p1, p2)
    n = np.floor(sms.NormalIndPower().solve_power(es, power=power, alpha=alpha, ratio=ratio))
    n1,n2 = n*ratio, n
    return n1,n2