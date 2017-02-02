from sklearn.base import BaseEstimator

from UnivariateLocalEstimator import UnivariateLocalEstimator


class VineKernelDensity (BaseEstimator):

    def __init__(self, tree_depth=None):
        """Store all values of parameters and nothing else

        Keyword arguments:
        """
        self.tree_depth_ = tree_depth
        # import inspect
        #
        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # values.pop("self")
        #
        # for arg, val in values.items():
        #     setattr(self, arg, val)

    def fit(self, X, sample_weight=None):
        """
        This function determines the structure of the copulae and creates the
        respective pdfs for the various marginal and copula densities
        X has dimensions (n_samples, n_features)
        """
        from sklearn.utils import check_array

        X = check_array(X, order='C')

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        # from scipy.sparse.csgraph import minimum_spanning_tree
        # import itertools
        # Z_ = self._gaussian_coord_transform(X)
        # self._first_tree_reduced_Z = Z_[reduced_indices]
        # self._first_tree_kdes = []
        # graph = np.zeros((X.shape[1], X.shape[1]))
        # for pair in itertools.combinations(np.arange(X.shape[1]), 2):
        #     graph[pair] = rdc(self._first_tree_reduced_U[:, pair[0]],
        #                       self._first_tree_reduced_U[:, pair[1]])
        # for indices in minimum_spanning_tree(-1*graph).nonzero():
        #     Z_pair = self._first_tree_reduced_Z[:, indices]
        #     # denom = np.product(stat.norm.pdf(Z_pair), axis=1)
        #
        #     self._first_tree_kdes.append(
        #         [indices,
        #          lambda u,
        #          kde=KernelDensity(kernel='gaussian',
        #                            bandwidth=0.1).fit(Z_pair):
        #          np.exp(kde.score_samples(u))])

    def score_samples(self, X):
        from sklearn.utils import check_array

        X = check_array(X, order='C')
