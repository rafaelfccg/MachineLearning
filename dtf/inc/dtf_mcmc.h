#pragma once

/*
   Copyright Microsoft Corporation 2012
   
   Author: Sebastian Nowozin (senowozi)
   
   Date: 25 April 2012
*/

#include <iostream>
#include <algorithm>
#include <functional>
#include <random>
#include <vector>
#include <cassert>

#include "convdiag.h"
#include "dtf.h"
#include "dtf_inference.h"

namespace dtf {

// Multiple-chain MCMC inference
//
// Implementing Replica-Exchange MCMC (parallel tempering) and Gelman-Rubin
// convergence monitoring.
//
// References
// [Geyer1991] Geyer, "Parallel Tempering",
//    Computing Science and Statistics, 1991.
// [Hukushima1996] Hukushima, Nemoto, "Exchange Monte Carlo method and
//    application to spin glass simulations",
//    Journal of the Physical Society of Japan, Vol. 65, No. 4,
//    pages 1604-1608, 1996.
// [Liu2004] Jun S. Liu, "Monte Carlo Strategies in Scientific Computing",
//    Springer, 2004.

template<typename DataTraits>
class multmcmc {
public:
	typedef std::array<size_t, DataTraits::label_count> marginal_type;
	typedef ts::image<marginal_type> inference_result_type;

private:
	size_t number_of_chains;
	size_t number_of_levels;	// number of temperature levels
	std::vector<std::vector<std::shared_ptr<inference::gibbs_chain<DataTraits>>>> ctxs;	// [ci][li]
	std::vector<std::vector<double>> ctxs_energies;

	std::vector<double> temperatures;	// Temperature ladder, [0]=1.0

	std::mt19937 rgen;

	// Convergence monitoring is performed on a random subsample of variables
	// only because updating the required statistics can be expensive
	psrf_stat<double> psrf_monitor;
	sparse_sampling pixels_to_monitor;

	// Marginal pixel distributions
	inference_result_type posterior;

	// Update convergence monitoring statistics for ladder ci
	void observe_variables(size_t ci) {
		auto state = ctxs[ci][0]->state();	// temp=1 chain label field

		size_t var_idx = 0;
		for (auto pi = pixels_to_monitor.begin();
			pi != pixels_to_monitor.end(); ++pi)
		{
			offset_t pos = *pi;
			label_t vstate = state(pos.x, pos.y);
			for (label_t label = 0; label < DataTraits::label_count; ++label) {
				psrf_monitor.observe(ci, var_idx, vstate == label ? 1.0 : 0.0);
				var_idx += 1;
			}
		}
	}

	// Standard temperized Gibbs sweep within a single chain.
	//
	// ci: chain index.
	// li: ladder index.
	// temp: chain temperature.
	void chain_sweep(size_t ci, size_t li) {
		ctxs_energies[ci][li] = ctxs[ci][li]->sweep();
	}

	void ladder_sweep(size_t ci) {
		std::uniform_real_distribution<double> randu;

		// Attempt between chain swap
		if (number_of_levels > 1) {
			std::uniform_int_distribution<size_t> rlevel(0, number_of_levels-2);
			size_t li = rlevel(rgen);
			assert(li < number_of_levels-1);

			// Swap candidate: li <-> li+1
			double E_xi0 = ctxs_energies[ci][li];
			double E_xi1 = ctxs_energies[ci][li+1];
			if (E_xi0 != E_xi0 || E_xi1 != E_xi1) {	// portable isnan (MSVC does not support C99 isnan)
				// In the first iteration we do not know the state energies
			} else {
				double T_0 = temperatures[li];
				double T_1 = temperatures[li+1];
				double log_accept_prob = (1.0/T_0 - 1.0/T_1)*(E_xi0 - E_xi1);

				if (log_accept_prob >= 0.0 || std::log(randu(rgen)) <= log_accept_prob) {
					// Accept temperature transition by swapping Gibbs chains
					std::swap(ctxs[ci][li], ctxs[ci][li+1]);
					std::swap(ctxs_energies[ci][li], ctxs_energies[ci][li+1]);

					// Set new temperatures of chains
					ctxs[ci][li]->temperature(temperatures[li]);
					ctxs[ci][li+1]->temperature(temperatures[li+1]);
				}
			}
		}

		// In-chain step
		for (size_t li(0); li < number_of_levels; ++li)
			chain_sweep(ci, li);
	}

	double mean_energy(void) const {
		double me(0);
		for (size_t ci(0); ci < number_of_chains; ++ci)
			me += ctxs_energies[ci][0];

		return me / static_cast<double>(number_of_chains);
	}

public:
	// number_of_chains: Number of MCMC ladders to create, >= 2.
	// number_of_levels: Number of temperature levels in each ladder, >= 1.
	// high_temp: Temperature of the highest-temperature chain, >= 1.0.
	// watch_fraction: Fraction of random pixels to monitor convergence for.
	multmcmc(const dtf::factor_graph<DataTraits>& graph,
		const typename DataTraits::input_type& image,
		size_t number_of_chains, size_t number_of_levels,
		double high_temp, double watch_fraction = 0.05, unsigned long seed = 31289u)
		: number_of_chains(number_of_chains),
			number_of_levels(number_of_levels), psrf_monitor(number_of_chains),
			pixels_to_monitor(sparse_subset_pixels(image, watch_fraction, seed)),
			posterior(image.width(), image.height())
	{
		assert(number_of_levels > 0);
		assert(high_temp >= 1.0);

		// Calculate geometric temperature ladder from high_temp to 1.0.
		double alpha = 1.0;
		if (number_of_levels > 1) {
			alpha = std::exp(std::log(1.0 / high_temp) /
				static_cast<double>(number_of_levels - 1));
		}
		double temp = high_temp;
		temperatures.resize(number_of_levels);
		for (int li = static_cast<int>(number_of_levels-1); li >= 0; --li) {
			temperatures[li] = temp;
			temp *= alpha;
		}

		rgen.seed(seed);

		// Generate chains
		ctxs.resize(number_of_chains);
		ctxs_energies.resize(number_of_chains);
		for (size_t ci(0); ci < number_of_chains; ++ci) {
			// Energies of state are still undefined
			ctxs_energies[ci].resize(number_of_levels);
			std::fill(ctxs_energies[ci].begin(), ctxs_energies[ci].end(),
				std::numeric_limits<double>::signaling_NaN());

			for (size_t li(0); li < number_of_levels; ++li) {
				// Initialize each chain with an individual seed
				ctxs[ci].push_back(std::make_shared<inference::gibbs_chain<DataTraits>>(graph, image,
					static_cast<unsigned long>(seed+8192*ci*number_of_levels+512*li+1)));
				ctxs[ci][li]->temperature(temperatures[li]);
			}
		}
	}

	// Compute max_i psrf_i
	double global_psrf(void) const {
		size_t var_idx = 0;
		double psrf = 0.0;

		for (auto pi = pixels_to_monitor.begin();
			pi != pixels_to_monitor.end(); ++pi) {
			for (label_t label = 0; label < DataTraits::label_count; ++label) {
				psrf = std::max(psrf, psrf_monitor.compute_psrf(var_idx));
				var_idx += 1;
			}
		}
		return psrf;
	}

	// Perform burn-in phase for all variables in the monitor.
	// Return true if burn-in was likely successful according to the PSRF
	// statistic, return false if it failed.
	bool perform_burnin(size_t maximum_sweeps, size_t minimum_sweeps = 1000,
		double psrf_thresh = 1.01)
	{
		assert(minimum_sweeps <= maximum_sweeps);
		assert(psrf_thresh >= 1.0);
		assert(pixels_to_monitor.begin() != pixels_to_monitor.end());

		for (size_t mi = 0; mi < maximum_sweeps; ++mi) {
			// Update all chains
			for (size_t ci = 0; ci < number_of_chains; ++ci) {
				ladder_sweep(ci);
				observe_variables(ci);
			}

			// Check global PSRF
			double psrf(global_psrf());
			if (mi % 100 == 0 || (mi >= minimum_sweeps && psrf <= psrf_thresh)) {
				double me(mean_energy());
				std::cout << "MultiMCMC burn-in " << (mi+1) << "  max_psrf " << psrf
					<< "  mean_E " << me << std::endl;
			}
			if (mi >= minimum_sweeps && psrf <= psrf_thresh)
				return true;
		}
		return false;
	}

	// Perform inference (after burn-in), creating the given number of
	// dependent samples per Markov chain.
	const inference_result_type& perform_inference(size_t samples_per_chain) {
		psrf_monitor.reset();

		// Reset inference result
		ts::parallel_for_each_pixel(posterior, [](marginal_type& pixel) -> void {
			std::fill(pixel.begin(), pixel.end(), label_t(0));
		});

		// Perform sampling, updating marginals
		assert(samples_per_chain > 0);
		for (size_t mi = 0; mi < samples_per_chain; ++mi) {
			for (size_t ci = 0; ci < number_of_chains; ++ci) {
				ladder_sweep(ci);
				observe_variables(ci);

				// Increase pixel marginal counts
				ts::parallel_pointwise_inplace(posterior, ctxs[ci][0]->state(),
					[](marginal_type& post_vi, const label_t& chain_vi) -> void {
						post_vi[chain_vi] += 1;
					});
			}
			if (mi % 100 == 0) {
				std::cout << "MultiMCMC sampling " << (mi+1) << " of "
					<< samples_per_chain << "  max_psrf " << global_psrf()
					<< std::endl;
			}
		}
		return posterior;
	}

	const inference_result_type& get_inference_result(void) {
		return posterior;
	}
};

// MPM, Maximum Posterior Marginal, classifier
// Given beliefs, the MPM decision is the one that minimizes the expected Hamming loss (i.e. per-pixel accuracy).
template <typename DataTraits>
class mpm_gibbs_classifier {
public:
	mpm_gibbs_classifier(const factor_graph<DataTraits>& graph,
		size_t number_of_chains = 3, size_t number_of_levels = 4, double high_temp = 10.0,
		size_t burnin_max_sweeps = 8192, double burnin_psrf_thresh = 1.01,
		size_t sample_count = 1024)
		: m_graph(graph), m_number_of_chains(number_of_chains), m_number_of_levels(number_of_levels),
			m_high_temp(high_temp), m_burnin_max_sweeps(burnin_max_sweeps),
			m_burnin_psrf_thresh(burnin_psrf_thresh), m_sample_count(sample_count)
	{
	}

	typename dtf::multmcmc<DataTraits>::inference_result_type infer_marginals(
		const typename DataTraits::input_type& input) const {
		// Inference
		dtf::multmcmc<DataTraits> mcmc(m_graph, input, m_number_of_chains, m_number_of_levels, m_high_temp, 0.1);
		mcmc.perform_burnin(m_burnin_max_sweeps, 100, m_burnin_psrf_thresh);
		return mcmc.perform_inference(m_sample_count);
	}

	ts::image<label_t> operator()(const typename DataTraits::input_type& input) const {
		auto posterior = infer_marginals(input);

		// MPM decisions
		ts::image<label_t> mpm_pred(posterior.width(), posterior.height());
		ts::parallel_pointwise_inplace(mpm_pred, posterior,
			[](label_t& mpm, const multmcmc<DataTraits>::marginal_type& post) -> void {
				size_t max_count = 0;
				for (label_t li = 0; li < post.size(); ++li) {
					if (post[li] > max_count) {
						mpm = li;
						max_count = post[li];
					}
				}
			});
		return mpm_pred;
	}

protected:
	const factor_graph<DataTraits>& m_graph;
	size_t m_number_of_chains;
	size_t m_number_of_levels;
	double m_high_temp;
	size_t m_burnin_max_sweeps;
	double m_burnin_psrf_thresh;
	size_t m_sample_count;
};

}

