import numpy as np
# from ..UnivariateLocalEstimator import UnivariateLocalEstimator
import UnivariateLocalEstimator
import logging
from timeit import default_timer as timer

logging.basicConfig(level=logging.INFO)

def generate_gaussian_data(n_samples=1000, **kw_args):
    return np.random.normal(size=n_samples, **kw_args)
    # mean = [0, 0]
    # cov = [[10, -25], [-25, 100]]
    # x, y = np.random.multivariate_normal(mean, cov, n_samples).T
    # cov = [[10, 25], [25, 100]]
    # x2, y2 = np.random.multivariate_normal(mean, cov, n_samples).T
    # x = np.hstack((x, x2))
    # y = np.hstack((y, y2))
    # return np.vstack((x, y)).T


def test_UnivariateLocalEstimator():
    n_samples = 5000

    raw_data = generate_gaussian_data(n_samples)
    ULE = UnivariateLocalEstimator.UnivariateLocalEstimator

    # cross validate local_event_frac through a
    # holdout method (split dataset in two)
    shuffled = raw_data.copy()
    np.random.shuffle(shuffled)
    test_raw_data, train_raw_data = \
        shuffled[:n_samples/2], shuffled[n_samples/2:]
    best_bw = None
    best_nll = None
    best_nlbins = 50
    fit_timings = []
    score_timings = []
    for bw in np.linspace(0.05, 0.8, 16, endpoint=True):  # np.array([1.0]):
        kde = ULE(nbins=best_nlbins, local_event_frac=bw)
        start = timer()
        kde.fit(train_raw_data.reshape(-1, 1))
        end = timer()
        fit_timings.append(end - start)
        start = timer()
        nll = -kde.score(test_raw_data.reshape(-1, 1))
        end = timer()
        score_timings.append(end - start)
        if best_nll is None:
            best_nll = nll
            best_bw = bw
        elif best_nll > nll:
            best_nll = nll
            best_bw = bw

    kde = ULE(nbins=best_nlbins, local_event_frac=best_bw)
    kde.fit(raw_data.reshape(-1, 1))

    logging.info('Best bandwidth: {}'.format(best_bw))
    logging.info('Average fit time ({} data points): {}'.format(
        train_raw_data.size, np.array(fit_timings).mean()))
    logging.info('Average score time ({} data points): {}'.format(
        test_raw_data.size, np.array(score_timings).mean()))

test_UnivariateLocalEstimator()
