import numpy as np


# TODO: write unit tests
def rdc(x, y, k=20, s=1/6., f=np.sin):
    """
    Compute the randomized dependence coefficient

    This algorithm is able to detect linear and non-linear correlations in the
    data vectors x and y.
    This is based on the paper titled "The Randomized Dependence Coefficient"
    located here https://arxiv.org/abs/1304.7717.

    Parameters
    ----------
    x : 1D numpy array with shape (N,)
        data coordinates
    y : 1D numpy array with shape (N,)
        data coordinates
    k,s : float
          tuning parameters - do not alter unless you really know what you're
          doing
    f : non-linear basis function

    Returns
    -------
    randomized dependence coefficient
    """
    import scipy.stats as stat
    from sklearn.cross_decomposition import CCA

    # the original was written in R (just 5 lines!), this is my translation
    # to numpy/scipy/scikit-learn (the original code is in the comments)

    # x <- cbind(apply(as.matrix(x),2,function(u)rank(u)/length(u)),1)
    # y <- cbind(apply(as.matrix(y),2,function(u)rank(u)/length(u)),1)
    x = stat.rankdata(x)/x.size
    y = stat.rankdata(y)/y.size
    x = np.insert(x[:, np.newaxis], 1, 1, axis=1)
    y = np.insert(y[:, np.newaxis], 1, 1, axis=1)
    # x <- s/ncol(x)*x%*%matrix(rnorm(ncol(x)*k),ncol(x))
    # y <- s/ncol(y)*y%*%matrix(rnorm(ncol(y)*k),ncol(y))
    x = np.dot(s/x.shape[1]*x,
               np.random.normal(size=x.shape[1]*k).reshape((x.shape[1], -1)))
    y = np.dot(s/y.shape[1]*y,
               np.random.normal(size=y.shape[1]*k).reshape((y.shape[1], -1)))
    # cancor(cbind(f(x),1),cbind(f(y),1))$cor[1]
    x = np.insert(f(x), x.shape[1], 1, axis=1)
    y = np.insert(f(y), y.shape[1], 1, axis=1)
    # the following is taken from:
    # http://stackoverflow.com/questions/37398856/
    # how-to-get-the-first-canonical-correlation-from-sklearns-cca-module
    cca = CCA(n_components=1)
    x_c, y_c = cca.fit_transform(x, y)
    return np.corrcoef(x_c.T, y_c.T)[0, 1]
