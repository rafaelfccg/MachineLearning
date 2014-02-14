/*
   Copyright Microsoft Corporation 2011
   
   Author: Sebastian Nowozin (senowozi)
   
   Date: 22 November 2011
   
*/

#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <limits>
#include <cmath>
#include <cassert>

#include "dtf.h"

namespace dtf {

// Gibbs sampler (Geman & Geman, 1984).
template <typename _DataTraits>
class gibbs_chain {
public:
	gibbs_chain(const dtf::factor_graph<_DataTraits>& structure,
		const _DataTraits::input_type& input)
		: inv_temp(1.0), m_structure(structure), m_input(input) {
	}

	// Set the temperature of the Gibbs distribution
	void set_inverse_temperature(double inv_temp) {
		this->inv_temp = inv_temp;
	}

	const ts::image<label_t>& get_state(void) const {
		return (state);
	}

	// Update state by applying the Gibbs transition operator
	void perform_sweep(int sweep_count = 1) {
		assert(sweep_count > 0);
		// TODO: perform a temperized sweep, in parallel
	}

private:
	typedef _DataTraits::label_t label_t;
	static const label_t label_count = _DataTraits::label_count;

	double inv_temp;

	// Current state of the Gibbs chain: FIXME: where is label_t defined?
	ts::image<label_t> state;

	const dtf::factor_graph<_DataTraits>& m_structure;
	const _DataTraits::input_type& m_input;

	// TODO: Toby, this is the function I will need to call at every variable in the model,
	// batched such that they do not interfere with each other.
	//
	// Resample a label given only its current energies and a random number generator.
	// This needs to be called for every pixel variable in the field.
	//
	// energies: Will be modified in place but can be discarded thereafter.
	label_t resample_label(double* energies,
		std::mt19937& mt, std::uniform_distribution<double>& drandu) const {
		double max_e = *std::max_element(energies, energies + label_count);
		double logZ = 0.0;
		for (label_t li = 0; li < label_count; ++li) {
			energies[li] = std::exp(inv_temp * (-energies[li] - max_e));
			logZ += energies[li];
		}

		// Sample within [0; logZ]
		double pv = logZ * drandu(mt);
		double cumsum = 0.0;
		for (label_t li = 0; li < label_count; ++li) {
			cumsum += energies[li];
			if (pv <= cumsum)	// in cdf interval?
				return (li);
		}
		assert(0);	// should never happen
	}
};

// Simulated annealing approximate MAP inference
template <typename _DataTraits>
class simulated_annealing_MAP_inference {
public:
	simulated_annealing_MAP_inference(const dtf::factor_graph<_DataTraits>& structure,
		const _DataTraits::input_type& input)
		: chain(structure, input) {
	}

	const ts::image<label_t>& get_state(void) const {
		return (chain.get_state());
	}

	void perform_inference(int sweep_count = 5000, double Tstart = 15.0, double Tend = 0.05) {
		assert(sweep_count > 0);
		assert(Tstart >= Tend);

		// Compute log-linear temperature schedule
		double alpha = std::exp(std::log(Tend/Tstart) / static_cast<double>(sweep_count));
		double T = Tstart;
		for (int sweep = 0; sweep < sweep_count; ++sweep) {
			chain.set_inverse_temperature(1.0 / T);
			chain.perform_sweep();

			T *= alpha;
		}
	}

private:
	gibbs_chain<_DataTraits> chain;
};


template <typename _DataTraits>
class multichain_gibbs_inference {
public:
	// Run multiple Gibbs chains in parallel until a convergence diagnostic
	// suggests that all the chains have reached the equilibrium distribution.
	multichain_gibbs_inference(const dtf::factor_graph<_DataTraits>& structure,
		const _DataTraits::input_type& input, int chain_count = 8)
		: var_count(input.width() * input.height()), width(input.width()), height(input.height());
	{
		assert(chain_count > 0);

		chains.reserve(chain_count);
		for (int ci = 0; ci < chain_count; ++ci)
			chains.push_back(gibbs_chain(structure, input));
	}

	// Perform multi-chain inference using PSRF as convergence monitor statistic.
	//
	// [Gelman1992], Andrew Gelman, Donald B. Rubin,
	//    "Inference from Iterative Simulation using Multiple Sequences",
	//    Statistical Science, Vol. 7, pages 457--511, 1992.
	//
	// max_sweeps: Maximum number of sweeps to perform before aborting.
	// accept_psrf: >1.0, potential scale reduction factor acceptance threshold.
	//    If max_i PSRF_c <= accept_psrf, we assume the chains have converged.
	//
	// Return true is PSRF suggests a good estimate; false if we did not converge.
	bool perform_inference(int max_sweeps = 0, double accept_psrf = 1.01) {
		setup_chains();
		int total_sample_count = 0;
		int number_of_chains = static_cast<int>(chains.size());

		// TODO: randomly initialize chains' states

		bool converged = false;
		while (true) {
			total_sample_count += 1;

			// Update all chains in parallel by a single sweep
			#pragma omp parallel for schedule(dynamic)
			for (int ci = 0; ci < number_of_chains; ++ci) {
				sweep(ci);
				update_mean_variance(total_sample_count, ci);
			}

			// Compute convergence statistic
			if (total_sample_count >= 2) {
				double current_psrf = compute_psrf(total_sample_count);
				if (current_psrf <= accept_psrf) {
					converged = true;
					break;
				}
			}

			// If we exhausted the maximum number of sweeps, give up
			if (max_sweeps != 0 && total_sample_count >= max_sweeps) {
				converged = false;
				break;
			}
		}
		// Compute pixel marginals as the average over all chains
		compute_pixel_marginals();

		return (converged);
	}

	// Return pixel posterior marginals
	const ts::image<std::vector<double>>& get_pixel_marginals(void) const {
		return (pixel_marginals);
	}

	// Return maximum posterior marginal decisions
	const ts::image<label_t>& get_pixel_mpm(void) const {
		return (pixel_mpm);
	}

private:
	// Number of labels
	typedef _DataTraits::label_t label_t;
	static const label_t label_count = _DataTraits::label_count;

	std::vector<gibbs_chain<_DataTraits>> chains;

	size_t var_count;
	unsigned int width;
	unsigned int height;

	// Marginal mean and variances
	typedef std::vector<std::vector<double>> chain_marginal_t;
	std::vector<chain_marginal_t> chain_mean;	// [ci][vi][ei]
	std::vector<chain_marginal_t> chain_varm;

	// Inference results
	ts::image<std::vector<double>> pixel_marginals;
	ts::image<label_t> pixel_mpm;

	void setup_chains(void) {
		setup_chains_stat(chain_mean);
		setup_chains_stat(chain_varm);
	}
	void setup_chains_stat(std::vector<chain_marginal_t>& cstat) {
		// Setup convergence statistic array
		cstat.resize(chains.size());
		for (size_t ci = 0; ci < chains.size(); ++ci) {
			cstat[ci].resize(var_count);
			for (size_t vi = 0; vi < var_count; ++vi) {
				cstat[ci][vi].resize(label_count);
				std::fill(cstat[ci][vi].begin(), cstat[ci][vi].end(), 0.0);
			}
		}
	}

	void sweep(int ci, int sweep_count = 1) {
		assert(sweep_count > 0);
		chains[ci].perform_sweep(sweep_count);
	}

	// Update the running mean/variance estimates in a stable way
	void update_mean_variance(int total_sample_count, int ci) {
		assert(total_sample_count >= 1);
		// Obtain current state of the chain
		const ts::image<label_t>& state = chains[ci].get_state();

		int vi = 0;
		for (int y = 0; y < state.height(); ++y) {
			for (int x = 0; x < state.width(); ++x) {
				for (size_t ei = 0; ei < chain_mean[ci][vi].size(); ++ei) {
					double x = (ei == state(x, y)) ? 1.0 : 0.0;
					double diff = x - chain_mean[ci][vi][ei];
					chain_mean[ci][vi][ei] += diff / static_cast<double>(total_sample_count);
					chain_varm[ci][vi][ei] += diff*(x - chain_mean[ci][vi][ei]);
				}
				vi += 1;
			}
		}
	}

	// This function should only be called after perform_inference has returned
	void compute_pixel_marginals(void) {
		// Average all chains
		assert(width*height == var_count);
		pixel_marginals.resize(width, height);
		double nfac = 1.0 / static_cast<double>(chains.size());

		size_t vi = 0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				// Create pixel marginal vector
				std::vector<double>& marg_vi = pixel_marginals(x, y);
				marg_vi.resize(label_count);

				// Take average
				for (size_t ci = 0; ci < chains.size(); ++ci) {
					std::transform(chain_mean[ci][vi].begin(), chain_mean[ci][vi].end(),
						marg_vi.begin(), marg_vi.begin(),
						[nfac](double v1, double v2) -> double { return (v1 + nfac*v2); });
				}
			}
		}

		// Infer MPM decision
		pixel_mpm.resize(width, height);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const std::vector<double>& marg = pixel_marginals(x, y);
				label_t max_label = static_cast<label_t>(
					std::max_element(marg.begin(), marg.end()) - marg.begin());

				pixel_mpm(x,y) = max_label;
			}
		}
	}

	// Compute the current potential scale reduction factor, given the individual
	// marginal distributions of each chain.
	double compute_psrf(int total_sample_count) {
		assert(total_sample_count >= 2);
		double max_psrf = -std::numeric_limits<double>>::infinity();

		// For each variable
		for (size_t vi = 0; vi < chain_mean[0].size(); ++vi) {
			// For each marginal distribution element
			for (size_t ei = 0; ei < chain_mean[0][vi].size(); ++ei) {
				// Compute chain mean
				double bar_t = 0.0;
				for (size_t ci = 0; ci < chains.size(); ++ci)
					bar_t += chain_mean[ci][vi][ei];
				bar_t /= static_cast<double>(chains.size());

				// Estimate between-chain variance B and within-chain variance W
				double B = 0.0;
				double W = 0.0;
				for (size_t ci = 0; ci < chains.size(); ++ci) {
					B += std::pow(chain_mean[ci][vi][ei] - bar_t, 2.0);
					W += chain_varm[ci][vi][ei] / static_cast<double>(total_sample_count-1);
				}

				B *= static_cast<double>(total_sample_count);
				B /= static_cast<double>(chains.size() - 1);
				W /= static_cast<double>(chains.size());

				// Posterior marginal variance estimate
				double V_hat = (static_cast<double>(total_sample_count-1)/static_cast<double>(total_sample_count)) * W
					+ (static_cast<double>(chains.size() + 1) / static_cast<double>(total_sample_count * chains.size())) * B;
				assert(V_hat >= 0.0);

				double psrf = 0.0;
				if (W >= 1.0e-12)
					psrf = std::sqrt(V_hat / W);

				max_psrf = std::max(psrf, max_psrf);
			}
		}

		return (max_psrf);
	}
};

}