#ifndef WeightedGaussKDE_KD_Tree_HPP
#define WeightedGaussKDE_KD_Tree_HPP

#include <cmath>
#include <limits>
#include "nanoflann.hpp"

// #include <iostream>

using namespace nanoflann;

template <size_t D>
struct WeightedNumpyPointCloud
{
	const double *pts;
	const double *weights;
  size_t n_points;

  // NOTE: ignore, just temporaries
  mutable double dist, diff;
  mutable size_t d;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return n_points; }

	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline double kdtree_distance(const double *p1, const size_t &idx_p2, size_t /*size*/) const
	{
    const size_t i_p2 = D*idx_p2;

    dist = 0.0;
    for (d = 0; d < D; ++d) {
      diff = p1[d] - pts[i_p2 + d];
      dist += diff*diff;
    }

    return dist + weights[idx_p2];
	}

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline double kdtree_get_pt(const size_t &idx, const size_t &dim) const
	{
    if (dim < D) return pts[D*idx + dim];
    return weights[idx];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

};

template <size_t D>
class WeightedGaussKDE_KD_Tree {
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, WeightedNumpyPointCloud<D> > ,
		WeightedNumpyPointCloud<D>,
		D + 1 /* tree dim (point dimensions plus weight) */
		> kd_tree_t;

  const double   pi;
  double         weight_shift,
                 tolerance;
  double        *sorting_weights;
  WeightedNumpyPointCloud<D> pc;
	kd_tree_t      index;

  // NOTE: CPU performance for both function bodies is identical
  inline double compute_si (const double &max_var, const double &weight) const
  {return max_var*(D*std::log(2*pi*max_var) + 2.*std::log(tolerance/weight));}
  // {return max_var*(std::log(std::pow(2*pi*max_var, D)*tolerance*tolerance/(weight*weight)));}

public:
  WeightedGaussKDE_KD_Tree () :
    pi(3.14159265358979323846264338327950288),
    sorting_weights(NULL),
    index(D + 1, pc, KDTreeSingleIndexAdaptorParams(10))
  {}

  WeightedGaussKDE_KD_Tree (const unsigned &max_leaf = 10) :
    pi(3.14159265358979323846264338327950288),
    sorting_weights(NULL),
    index(D + 1, pc, KDTreeSingleIndexAdaptorParams(max_leaf))
  {}

  ~WeightedGaussKDE_KD_Tree ()
  {
    if (sorting_weights != NULL) delete [] sorting_weights;
  }

  // NOTE: for pointwise adaptive bandwidths
  void SetPointsAndWeightsAdaptive (const double *points,
                                    const double *weights,
                                    const double *max_var,
                                    const size_t &n_points,
                                    const double &tol)
  {
    weight_shift = std::numeric_limits<double>::max();
    // NOTE: reduce tolerance due to numerical instabilities
    tolerance = 0.5*tol;

    if (sorting_weights != NULL) delete [] sorting_weights;
    sorting_weights = new double[n_points];
    for (size_t i = 0; i < n_points; ++i) {
      sorting_weights[i] = compute_si(max_var[i], weights[i]);
      if (weight_shift > sorting_weights[i]) weight_shift = sorting_weights[i];
    }
    if (weight_shift < 0.0) weight_shift *= -1.0;
    // NOTE: shift all sorting weights to be positive
    for (size_t i = 0; i < n_points; ++i) {
      sorting_weights[i] += weight_shift;
    }

    pc.pts = points;
    pc.weights = sorting_weights;
    pc.n_points = n_points;

    index.buildIndex();
  }

  // NOTE: for pointwise adaptive bandwidths
  // NOTE: https://groups.google.com/forum/#!topic/cython-users/4ecKM-p8dPA
  void SetPointsAndWeightsAdaptive (const double *points,
                                    const double *weights,
                                    const double *max_var,
                                    const size_t &n_points)
  {
    this->SetPointsAndWeightsAdaptive(points, weights, max_var, n_points, 1e-8);
  }

  // NOTE: for global bandwidth
  void SetPointsAndWeightsGlobal (const double *points,
                                  const double *weights,
                                  const double &max_var,
                                  const size_t &n_points,
                                  const double &tol = 1e-8)
  {
    weight_shift = std::numeric_limits<double>::max();
    // NOTE: reduce tolerance due to numerical instabilities
    tolerance = 0.5*tol;

    if (sorting_weights != NULL) delete [] sorting_weights;
    sorting_weights = new double[n_points];
    for (size_t i = 0; i < n_points; ++i) {
      sorting_weights[i] = compute_si(max_var, weights[i]);
      if (weight_shift > sorting_weights[i]) weight_shift = sorting_weights[i];
    }
    if (weight_shift < 0.0) weight_shift *= -1.0;
    // NOTE: shift all sorting weights to be positive
    for (size_t i = 0; i < n_points; ++i) {
      sorting_weights[i] += weight_shift;
    }

    pc.pts = points;
    pc.weights = sorting_weights;
    pc.n_points = n_points;

    index.buildIndex();
  }

  // NOTE: for global bandwidth
  // NOTE: https://groups.google.com/forum/#!topic/cython-users/4ecKM-p8dPA
  void SetPointsAndWeightsGlobal (const double *points,
                                  const double *weights,
                                  const double &max_var,
                                  const size_t &n_points)
  {
    this->SetPointsAndWeightsGlobal(points, weights, max_var, n_points, 1e-8);
  }

  size_t* RadiusSearch (size_t &match_size, const double *sc, const double &eps)
  {
		std::vector<std::pair<size_t, double> > ret_matches;
		nanoflann::SearchParams                 params;

    params.eps = eps;
		// params.sorted = true;

    // std::cout << "Search radius: " << weight_shift << std::endl;

		match_size = index.radiusSearch(sc,// search_center,
                                    weight_shift /*search radius*/,
                                    ret_matches,
                                    params);
    size_t *matched_indices = new size_t[match_size];
    for (size_t i = 0; i < match_size; ++i) {
      // std::cout << "matched radius: " << ret_matches[i].first << " " << ret_matches[i].second << std::endl;
      matched_indices[i] = ret_matches[i].first;
    }

    return matched_indices;
  }

  // NOTE: https://groups.google.com/forum/#!topic/cython-users/4ecKM-p8dPA
  size_t* RadiusSearch (size_t &match_size, const double *sc)
  {
    return this->RadiusSearch(match_size, sc, 1e-12);
  }
};

typedef WeightedGaussKDE_KD_Tree<1>  WeightedGaussKDE_KD_Tree_1D;
typedef WeightedGaussKDE_KD_Tree<2>  WeightedGaussKDE_KD_Tree_2D;

#endif /* WeightedGaussKDE_KD_Tree_HPP */
