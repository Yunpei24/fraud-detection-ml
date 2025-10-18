from scipy.stats import ks_2samp

def ks_p_value(sample_a, sample_b):
    stat, p = ks_2samp(sample_a, sample_b)
    return p
