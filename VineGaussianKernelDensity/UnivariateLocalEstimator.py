import numpy as np
from sklearn.base import BaseEstimator
from linear_binning import linear_binning
# from UnivariateLocalEstimator_helper import eval_tree


# TODO: unit test
class UnivariateLocalEstimator (BaseEstimator):

    def __init__(self, axis_trafo=None,
                 nbins=101,
                 local_event_frac=0.05,
                 bandwidth_scale=1.0):
        """Store all values of parameters and nothing else

        Keyword arguments
        -----------------
        axis_trafo : dict
            dictionary of if and how to transform the axis

        nbins : integer
            number of bins (linear rebinning) to downsample to

        local_event_frac : float
            fraction of total event weight to account for when
            computing the local variance (the local variances
            are used as the pointwise bandwidths)

        bandwidth_scale : float
            scale factor that multiplies the bandwidth(s)
        """
        self.axis_trafo_ = axis_trafo
        self.nbins_ = nbins
        self.local_event_frac_ = local_event_frac
        self.bandwidth_scale_ = bandwidth_scale
        # import inspect
        #
        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # values.pop("self")
        #
        # for arg, val in values.items():
        #     setattr(self, arg, val)

    def fit(self, X, sample_weight=None):
        """
        This function takes in a univariate dataset, performs a linear
        binning of the data, and constructs an adaptive Gaussian kernel
        estimator
        X has dimensions (n_samples,)
        sample_weight has dimensions (n_samples,) or None for the case
        of equally weighted data
        """
        from sklearn.utils import check_array

        X = check_array(X, order='C')

        if sample_weight is None:
            sample_weight = np.ones((X.size, 1))/float(X.size)

        self._axis_info = self._make_axis_info(X)
        self._scaler = self._create_scaler(X)

        self._X_scaled = self._preprocess_data(X)
        self._X_weights = sample_weight

        self._X_grid, self._weights = \
            self._make_1D_linear_binning(self._X_scaled,
                                         sample_weight,
                                         size=self.nbins_)
        self._sum_weights = self._weights.sum()
        if not np.isclose(self._X_weights.sum(), self._sum_weights):
            print self._X_weights.sum(), self._sum_weights
            raise ValueError('sum of weights after linear binning is ' +
                             'not conserved - something went wrong')

        if self.local_event_frac_ < 1.0 and self.local_event_frac_ > 0.0:
            _covs = self._compute_local_covariances()
        else:
            _covs = self._compute_covariance()
        if self.bandwidth_scale_ != 1.0 and self.bandwidth_scale_ > 0.0:
            _covs *= self.bandwidth_scale_

        self._pdf_denom = (self._weights / np.sqrt(2.*np.pi*_covs))
        self._inv_covs = np.reciprocal(_covs)
        self._cdf_exp_denom = np.reciprocal(np.sqrt(2*_covs))

    def score_samples(self, X):
        from sklearn.utils import check_array

        #X = self._check_array(X, 'X')

        X = check_array(X, order='C')

        scaled_X = self._preprocess_data(X)

        # compute exponent
        delta = np.squeeze(scaled_X[:, None, :] - self._X_grid)
        ex = delta*self._inv_covs*delta

        est = np.sum(np.exp(-.5 * ex) * self._pdf_denom,
                     axis=1)

        return np.log(self._post_process_estimate(est, X)) - np.log(self._sum_weights)

    def score(self, X):
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.
        random_state : RandomState or an int seed (0 by default)
            A random number generator instance.
        Returns
        -------
        X : array_like, shape (n_samples, 1)
            List of samples.
        """
        from sklearn.utils import check_random_state

        #data = np.asarray(self.tree_.data)

        rng = check_random_state(random_state)
        #i = rng.randint(data.shape[0], size=n_samples)

        #if self.kernel == 'gaussian':
        #    return np.atleast_2d(rng.normal(data[i], self.bandwidth))

        #elif self.kernel == 'tophat':
        #    # we first draw points from a d-dimensional normal distribution,
        #    # then use an incomplete gamma function to map them to a uniform
        #    # d-dimensional tophat distribution.
        #    dim = data.shape[1]
        #    X = rng.normal(size=(n_samples, dim))
        #    s_sq = row_norms(X, squared=True)
        #    correction = (gammainc(0.5 * dim, 0.5 * s_sq) ** (1. / dim)
        #                  * self.bandwidth / np.sqrt(s_sq))
        #    return data[i] + X * correction[:, np.newaxis]

    def pdf(self, X):
        return np.exp(self.score_samples(X))

    def cdf(self, X):
        from scipy.special import erf
        from sklearn.utils import check_array

        X = check_array(X, order='C')

        scaled_X = self._preprocess_data(X)

        ex = np.squeeze(scaled_X[:, None, :] - self._X_grid)*self._cdf_exp_denom

        c = .5*np.sum((erf(ex) + 1.) * self._weights, axis=1)

        if self._axis_info['flip']:
            return 1. - c
        return c

    def _make_axis_info(self, X):
        from scipy.special import logit
        # check if axis_trafo is None, has only range, has only trafo func,
        #  then has range + trafo func, then has all info

        # default is identity transform
        result = {'range': None,
                  'trafo': lambda x: x,
                  'inv_trafo': np.reciprocal,
                  'denom': lambda x: np.ones(x.size),
                  'flip': False
                 }
        if self.axis_trafo_ is None:
            return result

        if 'range' in self.axis_trafo_:
            result['range'] = self.axis_trafo_['range']
        if 'flip' in self.axis_trafo_:
            result['flip'] = self.axis_trafo_['flip']

        if 'trafo' in self.axis_trafo_ and\
           'inv_trafo' in self.axis_trafo_ and\
           'denom' in self.axis_trafo_:
            result['trafo'] = self.axis_trafo_['trafo']
            result['inv_trafo'] = self.axis_trafo_['inv_trafo']
            result['denom'] = self.axis_trafo_['denom']
            return result
        if 'range' in self.axis_trafo_ and\
           'trafo' not in self.axis_trafo_:
            result['trafo'] = np.tan
            result['inv_trafo'] = np.arctan
            result['denom'] = lambda x: np.power(np.cos(x), 2)
            return result
        if 'trafo' in self.axis_trafo_:
            if self.axis_trafo_['trafo'] == np.tan:
                result['trafo'] = np.tan
                result['inv_trafo'] = np.arctan
                result['denom'] = lambda x: np.power(np.cos(x), 2)
            elif self.axis_trafo_['trafo'] == np.log:
                result['trafo'] = np.log
                result['inv_trafo'] = np.exp
                result['denom'] = lambda x: x
            elif self.axis_trafo_['trafo'] == logit:
                result['trafo'] = logit
                result['inv_trafo'] = ss.logistic
                result['denom'] = lambda x: x*(1. - x)
            return result


    def _create_scaler(self, X):
        from scipy.special import logit
        from sklearn.preprocessing import MinMaxScaler
        # shift and scale dataset
        X_min, X_max = X.min(), X.max()
        X_range = np.abs(X_min - X_max)
        scale_range = None
        if self._axis_info['range'] is not None:
            scale_range = np.abs(self._axis_info['range'][1] - self._axis_info['range'][0])
        if self._axis_info['flip']:
            X_max, X_min = -X_min, -X_max
        if self._axis_info['trafo'] == logit:
            r = [np.nextafter(0., 1.), np.nextafter(1., 0.)]
            if scale_range is not None:
                r[0] = .5 - X_range/scale_range*(.5 - np.nextafter(0., 1.))
                r[1] = .5 + X_range/scale_range*(np.nextafter(1., 0.) - .5)
            return MinMaxScaler(feature_range=(r[0], r[1]))
        elif self._axis_info['trafo'] == np.tan:
            r = [np.nextafter(-0.5*np.pi, 0.), np.nextafter(0.5*np.pi, 0.)]
            if scale_range is not None:
                r[0] = 0. - X_range/scale_range*(0. - np.nextafter(-0.5*np.pi, 0.))
                r[1] = 0. + X_range/scale_range*(np.nextafter(0.5*np.pi, 0.) - 0.)
            return MinMaxScaler(feature_range=(r[0], r[1]))
        elif self._axis_info['trafo'] == np.log:
            return MinMaxScaler(feature_range=(np.nextafter(0., 1.),
                                               X.max()))
        else:
            return None

    def _preprocess_data(self, X):
        if self._axis_info['flip']:
            X = np.negate(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._axis_info['trafo'](X)

    def _post_process_estimate(self, f, X):
        # X is assumed to not be preprocessed
        if self._axis_info['flip']:
            X = np.negate(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        f /= self._axis_info['denom'](X)
        if self._scaler is not None:
            # NOTE: in the sklearn transformations, the scale_ attribute is the inverse of m_S in the above discussion
            f /= self._scaler.scale_
        return f

    def _compute_covariance(self):
        return np.cov(self._X_scaled,
                      rowvar=False,
                      aweights=self._X_weights.reshape(self._X_weights.shape[0]))

    # TODO: possibly jit this or implement in cython
    def _compute_local_covariances(self):
        from scipy.spatial import cKDTree
        tree = cKDTree(self._X_scaled)
        result = np.empty(self._X_grid.size)

        temp = np.empty((1, 1))
        total_weight_frac = self.local_event_frac_*self._sum_weights
        for i in range(self._X_grid.size):
            sum_weight_found = False
            k = 1
            weight_sum = 0.0
            last_ii = 0
            good_ii = None
            min_ii = None
            temp[0, 0] = self._X_grid[i]
            while not sum_weight_found:
                dists, indices = tree.query(temp, k)
                dists = dists.flatten()
                indices = indices.flatten()
                for ii in range(last_ii, indices.size):
                    last_ii = ii
                    weight_sum += self._X_weights[indices[ii]]
                    #print weight_sum, self.local_event_frac_*self._sum_weights
                    if min_ii is None and dists[ii] >= 0.0:
                        min_ii = ii
                    if weight_sum >= total_weight_frac and min_ii is not None:
                        good_ii = ii
                        break
                if good_ii is not None:
                    good_indices = indices[:good_ii + 1]
                    result[i] = np.cov(tree.data[good_indices],
                                       rowvar=False,
                                       aweights=self._X_weights.reshape(self._X_weights.shape[0])[good_indices])
                    sum_weight_found = True
                elif k == tree.n:
                    result[i] = self._compute_covariance()
                    sum_weight_found = True
                k = min(2*k, tree.n)
        return result

    def _make_1D_linear_binning(self, X, weights, size=51, extent=None):
        """
        Linear binning is used for down-sampling data while retaining much
        higher fidelity (in terms of asymptotic behavior) than nearest-neighbor
        binning (the usual type of binning).
        A--------P--------------------------B
        For a 1D point P with weight wP:
        Assign a weight to point A of the proportion of length (times wP)
            between P and B
        Assign a weight to point B of the proportion of length (times wP)
            between P and A
        """
        D = 1
        n_bins = np.full(D, size, dtype=np.dtype('uint64'))
        if extent is None:
            extent = np.array([X.min(), X.max()])
        extent = extent.reshape((1, 2))
        return linear_binning(X, np.squeeze(weights), extent, n_bins)
