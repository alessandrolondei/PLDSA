import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# DFA
def DFA(seq, L, verbose=True, fit=False, fr=None, to=None):
    # Detrended Fluctuation Analysis
    ind_most_common = list(zip(*Counter(seq).most_common()))[0]
    ind_to_most_common = dict()
    for k, imc in enumerate(ind_most_common):
        ind_to_most_common[imc] = k
    sortedseq = np.array([ind_to_most_common[i] for i in seq])
    mean = sortedseq.mean()
    X = (sortedseq - mean).cumsum()
    FL = []
    for l in tqdm(L, disable=(not verbose)):
        FL0 = []
        for k in range(0, len(seq), l):
            subX = X[k:k+l]
            xx = np.arange(k,k+len(subX)).reshape(-1,1)
            reg = LinearRegression().fit(xx, subX)
            pred = reg.predict(xx)
            FL0 += list(np.power(subX - pred, 2.0))
        FL0 = np.array(FL0)
        FL.append(np.sqrt(FL0.sum())/len(seq))
    FL= np.array(FL)
    if fit:
        popt, pcov, y_fit =_loglog_linear_fit(FL, xx=L, fr=fr, to=to)
        return FL, popt, y_fit
    else:
        return FL, None, None

    

def heaps(seq, verbose=True,
                fit=False, fr=None, to=None):
    items = set()
    heaps = []
    for c in tqdm(seq, disable=(not verbose)):
        items.add(c)
        heaps.append(len(items))
    if fit:
        popt, pcov, y_fit =_loglog_linear_fit(heaps, fr=fr, to=to)
        return heaps, popt, y_fit
    else:
        return heaps, None, None


def zipf(seq, verbose=True,
                fit=False, fr=None, to=None):
    seq_list = list(seq)
    if verbose:
        zipf = []
        for t in tqdm(Counter(seq_list).most_common(), disable=False):
            zipf.append(t[1])
    else:
        zipf = [t[1] for t in Counter(seq_list).most_common()]
    if fit:
        popt, pcov, y_fit =_loglog_linear_fit(zipf, fr=fr, to=to)
        return zipf, popt, y_fit
    else:
        return zipf, None, None


def entropy(distr, norm=False):
    entr = 0.0
    for k, x in enumerate(distr):
        if x > 0.0:
            entr -= x * np.log(x)
    if norm:
        if len(distr) > 1:
            entr /= np.log(len(distr))
    return entr


def entropy_seq(seq, norm=False, verbose=True,
                fit=False, fr=None, to=None):
    ind = 0
    freq = []
    c2ind = dict()
    entr_seq = []
    for c in tqdm(seq, disable=(not verbose)):
        if c not in c2ind.keys():
            c2ind[c] = ind
            freq.append(1)
            ind += 1
        else:
            freq[c2ind[c]] += 1
        distr = np.array(freq) / sum(freq)
        entr_seq.append(entropy(distr, norm=norm))
    entr_seq = np.array(entr_seq[1:])
    if fit:
        popt, pcov, y_fit = _loglog_linear_fit(entr_seq, fr=fr, to=to)
        return entr_seq, popt, y_fit
    else:
        return entr_seq, None, None
    

def _loglog_linear_fit(data, xx=None, fr=None, to=None):
    def func(x, m, q):
        return np.log(x) * m + q
    if fr is None:
        fr = 0
    if to is None:
        to = len(data)
    ydata = np.log(data[fr:to])
    if xx is None:
        xdata = np.arange(fr+1, to+1)
        x_fit = np.arange(1, len(data)+1)
    else:
        xdata = xx[fr:to]
        x_fit = xx
    popt, pcov = curve_fit(func, xdata, ydata, p0=(-1,0))
    m = popt[0]
    q = popt[1]
    
    y_fit = np.exp(np.log(x_fit) * m + q)
    return popt, pcov, y_fit