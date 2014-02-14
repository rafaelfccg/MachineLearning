#pragma once

/*
   Copyright Microsoft Corporation 2012
   
   Author: Sebastian Nowozin (senowozi)
   
   Date: 25 April 2012
*/

#include <vector>
#include <limits>
#include <unordered_map>
#include <cassert>
#include <cmath>

namespace dtf {

/* Gelman and Rubin convergence diagnosis for MCMC inference.
 *
 * The PSRF is a per-variable diagnosis and when diagnosing convergence for
 * many variables the recommended procedure is to take the maximum of all
 * variable's PSRF.  If a variable has never been observed in one chain, the
 * corresponding PSRF will be +infinity.
 *
 * We also compute running mean and variance estimates of all variables
 * tracked.
 *
 * References
 * [Brooks1998], Stephen P. Brooks, Andrew Gelman,
 *     "General Methods for Monitoring Convergence of Iterative Simulations",
 *     Journal of Computational and Graphical Statistics,
 *     Vol. 7, No. 4, pages 434--455, December 1998.
 *
 * [Gelman1992], Andrew Gelman, Donald B. Rubin,
 *     "Inference from Iterative Simulation using Multiple Sequences",
 *     Statistical Science,
 *     Vol. 7, pages 457--511, 1992.
 *
 * [Ruppert2010], David Ruppert,
 *     "Statistics and Data Analysis for Financial Engineering",
 *     pages 553--556, Springer, 2010.
 */
template<typename TFloat>
class psrf_stat {
private:
	size_t number_of_chains;

	typedef std::unordered_map<size_t, size_t> chain_count_t;
	typedef std::vector<chain_count_t> multichain_count_t;

	typedef std::unordered_map<size_t, TFloat> chain_stat_t;
	typedef std::vector<chain_stat_t> multichain_stat_t;

	multichain_count_t mchain_count;
	multichain_stat_t mchain_mu;
	multichain_stat_t mchain_s2;

	size_t chain_count(size_t chain_id, size_t var_id) const {
		chain_count_t::const_iterator ci = mchain_count[chain_id].find(var_id);
		if (ci == mchain_count[chain_id].end())
			return 0;
		return ci->second;
	}

	size_t min_count(size_t var_id) const {
		size_t count(std::numeric_limits<size_t>::max());
		for (size_t ci(0); ci < number_of_chains; ++ci)
			count = std::min(chain_count(ci, var_id), count);

		return count;
	}

	// Sample mean of a variable in a chain
	TFloat chain_mean(size_t chain_id, size_t var_id) const {
		typename chain_stat_t::const_iterator mi =
			mchain_mu[chain_id].find(var_id);
		if (mi == mchain_mu[chain_id].end())
			return std::numeric_limits<TFloat>::quiet_NaN();

		return mi->second;
	}

	// Sample variance of var_id in chain chain_id.
	TFloat chain_var(size_t chain_id, size_t var_id) const {
		typename chain_stat_t::const_iterator mi =
			mchain_s2[chain_id].find(var_id);
		if (mi == mchain_s2[chain_id].end())
			return std::numeric_limits<TFloat>::quiet_NaN();

		typename chain_count_t::const_iterator mci =
			mchain_count[chain_id].find(var_id);
		assert(mci != mchain_count[chain_id].end());

		return (mi->second / static_cast<TFloat>(mci->second-1));
	}

	// Compute within-chain variance W
	TFloat compute_within_chain_variance(size_t var_id) const {
		TFloat vsum(0);
		for (size_t ci(0); ci < number_of_chains; ++ci)
			vsum += chain_var(ci, var_id);

		return vsum / static_cast<TFloat>(number_of_chains);
	}

	TFloat compute_mean(size_t var_id) const {
		// Compute mean over chains
		TFloat mu(0);
		for (size_t ci(0); ci < number_of_chains; ++ci)
			mu += chain_mean(ci, var_id);
		return mu / static_cast<TFloat>(number_of_chains);
	}

	// Compute between-chain variance B/T
	TFloat compute_between_chain_variance(size_t var_id) const {
		TFloat mu(compute_mean(var_id));
		TFloat var(0);
		for (size_t ci(0); ci < number_of_chains; ++ci)
			var += std::pow(mu - chain_mean(ci, var_id), 2.0);

		return var / static_cast<TFloat>(number_of_chains-1);
	}

public:
	// Reset to clear state without any observations
	void reset(void) {
		mchain_count.clear();
		mchain_mu.clear();
		mchain_s2.clear();
		mchain_count.resize(number_of_chains);
		mchain_mu.resize(number_of_chains);
		mchain_s2.resize(number_of_chains);
	}

	psrf_stat(size_t number_of_chains)
		: number_of_chains(number_of_chains) {
		reset();
	}

	void observe(size_t chain_id, size_t var_id, TFloat value) {
		assert(chain_id < number_of_chains);
		chain_count_t& count = mchain_count[chain_id];
		chain_stat_t& mu = mchain_mu[chain_id];
		chain_stat_t& s2 = mchain_s2[chain_id];

		// Increase number of observations for this variable in this chain
		size_t n(count[var_id]);
		if (n == 0) {
			mu[var_id] = TFloat(0);
			s2[var_id] = TFloat(0);
		}

		// Online mean/variance update (Knuth and Welford)
		n += 1;
		TFloat d(value - mu[var_id]);
		mu[var_id] += d / static_cast<TFloat>(n);
		if (n > 1)
			s2[var_id] += d*(value - mu[var_id]);

		count[var_id] = n;
	}

	// Compute the simplified potential scale reduction factor (PSRF).
	// If PSRF is <= (1.0 + epsilon) then convergence of the chains can be
	// diagnosed.
	TFloat compute_psrf(size_t var_id) const {
		// We take T to be the minimum number of observations for this
		// variable over all chains, which is the most conservative estimate
		// as only the slowest chain counts.
		size_t T(min_count(var_id));
		if (T <= 1)
			return std::numeric_limits<TFloat>::infinity();

		// Gelman and Rubin, 1992, (1) to (4)
		TFloat B_T(compute_between_chain_variance(var_id));
		TFloat W_T(compute_within_chain_variance(var_id));
		TFloat sigma_2((static_cast<TFloat>(T-1)/static_cast<TFloat>(T))*W_T + B_T);

		return std::sqrt(sigma_2 / W_T);
	}

	// Compute approximate effective sample size, by (20.42) in [Ruppert2010].
	TFloat compute_ess(size_t var_id) const {
		size_t T(min_count(var_id));
		TFloat B_T(compute_between_chain_variance(var_id));
		TFloat W_T(compute_within_chain_variance(var_id));
		TFloat sigma_2((static_cast<TFloat>(T-1)/static_cast<TFloat>(T))*W_T + B_T);

		return static_cast<TFloat>(number_of_chains)* (sigma_2 / B_T);
	}

	// Estimate of the mean
	TFloat mean(size_t var_id) const {
		return compute_mean(var_id);
	}

	// Estimate of the variance
	TFloat variance(size_t var_id) const {
		size_t T(min_count(var_id));
		if (T <= 1)
			return std::numeric_limits<TFloat>::infinity();

		TFloat B_T(compute_between_chain_variance(var_id));
		TFloat W_T(compute_within_chain_variance(var_id));

		return (static_cast<TFloat>(T-1)/static_cast<TFloat>(T))*W_T
			+ B_T;
	}
};

}

