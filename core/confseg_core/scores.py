"""Pure NumPy segmentation conformity and efficiency reductions."""
import numpy as np

def _probs(p):
    a=np.asarray(p,dtype=np.float64)
    if a.ndim < 2 or not np.isfinite(a).all(): raise ValueError('probabilities must be finite Kx...')
    if np.any(a < 0) or np.any(a > 1) or not np.allclose(a.sum(axis=0),1.,atol=1e-3,rtol=0): raise ValueError('probabilities must sum to one')
    return a

def pixel_scores(probs, labels, ignore_index=255):
    p=_probs(probs); y=np.asarray(labels)
    if p.shape[1:] != y.shape: raise ValueError('label shape mismatch')
    if not np.isfinite(y).all() or np.any(y != np.floor(y)): raise ValueError('label IDs must be finite integers')
    m=y != ignore_index
    if not m.any(): return np.empty(0,dtype=np.float64)
    yy=y[m].astype(np.int64)
    if np.any((yy<0)|(yy>=p.shape[0])): raise ValueError('label out of range')
    flat=p.reshape(p.shape[0],-1)[:,m.ravel()]; idx=np.arange(yy.size)
    s=1-flat[yy,idx]; return np.where(flat.argmax(axis=0)==yy,0.,s)

def quantile_score(scores,tau):
    s=np.asarray(scores,dtype=np.float64)
    if s.ndim!=1 or not np.isfinite(s).all() or np.any((s<0)|(s>1)): raise ValueError('scores must be finite in [0,1]')
    if not 0<=tau<=1: raise ValueError('tau outside [0,1]')
    n=s.size
    if n==0: raise ValueError('all-void image excluded')
    k=int(np.floor(tau*n+1e-9)); return 0. if k>=n else float(np.partition(s,n-k-1)[n-k-1])

def miss_fraction(scores,lam):
    s=np.asarray(scores); return float(np.mean(s>lam)) if s.size else float('nan')

def crc_threshold(calibration_scores,alpha):
    s=np.asarray(calibration_scores,dtype=np.float64)
    if s.ndim!=1 or not s.size or not np.isfinite(s).all() or np.any((s<0)|(s>1)): raise ValueError('invalid calibration scores')
    if not 0<alpha<1: raise ValueError('alpha must be in (0,1)')
    k=int(np.ceil((len(s)+1)*(1-alpha)-1e-12))
    return 1. if k>len(s) else float(np.partition(s,k-1)[k-1])

def ar_apf_curves(probs, labels, grid, ignore_index=255):
    """Exact curves using O(KN log(KN)) sorted activations, never an N x K x G tensor."""
    p = _probs(probs)
    y = np.asarray(labels)
    g = np.asarray(grid, dtype=np.float64)
    if (p.shape[1:] != y.shape or g.ndim != 1 or not np.isfinite(g).all()
            or np.any(np.diff(g) < 0) or np.any((g < 0.) | (g > 1.))):
        raise ValueError('shape/grid invalid')
    if not np.isfinite(y).all() or np.any(y != np.floor(y)):
        raise ValueError('label IDs must be finite integers')
    valid = y != ignore_index
    n = int(valid.sum())
    if not n:
        raise ValueError('all-void image excluded')
    flat = p.reshape(p.shape[0], -1)[:, valid.ravel()]
    top = flat.argmax(axis=0)  # NumPy argmax gives lowest-index ties.
    activation = 1. - flat
    activation[top, np.arange(n)] = np.inf
    sorted_class = np.sort(activation[np.isfinite(activation)])
    ar = 1. + np.searchsorted(sorted_class, g, side='right') / n
    pixel_min = np.min(activation, axis=0)
    sorted_pixel = np.sort(pixel_min)
    apf = np.searchsorted(sorted_pixel, g, side='right') / n
    return ar.astype(np.float64), apf.astype(np.float64)

def paired_calibration_draw(pool,seed,n=250):
    """Sorted, fit-only sampling without replacement; reuse the returned vector across arms."""
    values=np.asarray(pool,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all() or not 0<n<=len(values): raise ValueError('invalid calibration pool or draw size')
    idx=np.random.default_rng(int(seed)).choice(len(values),size=int(n),replace=False)
    return values[np.sort(idx)]

def score_image(probs,labels,tau=.05,ignore_index=255):
    """Return scores plus an explicit auditable exclusion reason for all-void images."""
    scores=pixel_scores(probs,labels,ignore_index)
    if scores.size==0: return {'scores':scores,'n_valid':0,'excluded':True,'exclusion_reason':'all_void'}
    return {'scores':scores,'n_valid':int(scores.size),'excluded':False,'exclusion_reason':None,'score_tau':quantile_score(scores,tau)}
