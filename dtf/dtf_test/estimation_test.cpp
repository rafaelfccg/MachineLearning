#include <algorithm>
#include <iostream>
#include <cmath>

#include "dtf.h"
#include "dtf_inference.h"
#include "dtf_mcmc.h"
#include "image.h"
#include "imageio.h"
#include "timing.h"

namespace EstimationTest
{

// Simple dummy class that can take a set of labels
template <unsigned char _LabelCount, typename _Pixel>
class dataset {
private:
   std::vector<ts::image<const unsigned char>> m_gt;
   ts::image<_Pixel> dummy_image;

public:
   static const dtf::label_t label_count = _LabelCount;

   dataset(unsigned int image_size, const std::vector<ts::image<const unsigned char>>& gt) :
      m_gt(gt), dummy_image(image_size, image_size)
   {
   }

   size_t size() const { return m_gt.size(); }

   ts::image<const _Pixel> get_training(size_t i) const
   {
      assert(i < m_gt.size());
      return dummy_image;
   }

   ts::image<const unsigned char> get_ground_truth(size_t i) const
   {
      assert(i < m_gt.size());
      return m_gt[i];
   }

   dtf::dense_sampling get_samples(size_t i) const
   {
      return dtf::all_pixels(get_ground_truth(i));
   }
};

template <typename _Pixel>
class dummy_feature {
public:
   static ts::image<typename _Pixel> pre_process(const ts::image<typename _Pixel>& image) { return image; }
   template <typename _It> bool operator()(unsigned int x, unsigned int y,
      const ts::image<typename _Pixel>& data, _It off) const { return true; }
};

// Return the adjustment constant that was subtracted.
static dtf::weight_t adjust_weights(std::vector<dtf::weight_t>& weights, int label_count)
{
   dtf::weight_t alpha(0);
   for (int li = 0; li < label_count; ++li)
      alpha += weights[li*label_count+li];
   alpha /= dtf::weight_t(label_count);

   for (size_t vi = 0; vi < weights.size(); ++vi)
      weights[vi] -= alpha;
   return alpha;
}

// Check a small pairwise model against NMF inference ground truth from grante
void RunNMFTest()
{
   typedef ts::bgr24 pixel_type;
   static const unsigned char label_count = 2;
   typedef dataset<label_count, pixel_type> dataset_type;
   typedef dtf::database_traits<dataset_type> DataTraits;

   dtf::factor_graph<DataTraits> graph;
   typedef dummy_feature<pixel_type> feature;

   // Add a single pairwise factor type
   dtf::factor<DataTraits, 2, feature> pairwise;
   pairwise[0] = dtf::offset_t(0, 0);
   pairwise[1] = dtf::offset_t(1, 0);
   ts::binary_tree_array<feature> tree;
   pairwise.set_tree(tree);
   graph.move_factor(std::move(pairwise));

   std::vector<dtf::weight_t> W;
   W.push_back(0.5);
   W.push_back(-0.8);
   W.push_back(0.2);
   W.push_back(0.7);
   dtf::dtf_set_all_weights(graph, W.begin(), W.end());

   // Dummy image, one pairwise factor
   ts::image<pixel_type> image(2, 1);

   // Mean field test
   dtf::inference::naive_mean_field<DataTraits> nmf(graph, image);
   double prev_log_z = -std::numeric_limits<double>::infinity();
   for (int iter = 0; iter < 10; ++iter) {
      double log_z = nmf.sweep();
      double conv = log_z - prev_log_z;
      prev_log_z = log_z;
      std::cout << "NMF iter " << iter << "  logZ >= " << log_z << "  conv " << conv << std::endl;
   }
   // NMF lower bound ground truth: 1.3452977
   double lb_error = std::fabs(prev_log_z - 1.3452977);

   // Ground truth unary marginals:
   //    at (0,0): [0.3113, 0.6887],
   //    at (1,0): [0.7190, 0.2810].
   auto nmf_marg = nmf.marginals();
   for (int xi = 0; xi < 2; ++xi) {
      for (int label = 0; label < label_count; ++label) {
         std::cout << "marg(" << xi << ",0)[" << label << "] = " << nmf_marg(xi,0)[label] << std::endl;
      }
   }

   double marg_error = 0;
   marg_error += std::fabs(nmf_marg(0,0)[0]-0.3113);
   marg_error += std::fabs(nmf_marg(0,0)[1]-0.6887);
   marg_error += std::fabs(nmf_marg(1,0)[0]-0.7190);
   marg_error += std::fabs(nmf_marg(1,0)[1]-0.2810);

   // Check whether inference result agrees
   if (marg_error > 1.0e-3 || lb_error > 1.0e-6) {
      std::cout << "# NMF test FAILED, marg_error " << marg_error << ", lb_error " << lb_error << std::endl;
   } else {
      std::cout << "# NMF test passed" << std::endl;
   }
   std::cout << "NMF test" << std::endl;
}

void Run()
{
   RunNMFTest();
   static const unsigned char label_count = 4;
   static const size_t image_size = 32;
   static const size_t sample_count = 1000;
   typedef ts::bgr24 pixel_type;
   typedef dataset<label_count, pixel_type> dataset_type;
   typedef dtf::database_traits<dataset_type> DataTraits;

   // Setup factor graph
   dtf::factor_graph<DataTraits> graph;
   typedef dummy_feature<pixel_type> feature;
   std::vector<dtf::prior_t> priors;

   std::vector<dtf::weight_t> w_gt;

   // Add two pairwise factor types, horizontal and vertical
   dtf::factor<DataTraits, 2, feature> pairwise_h;
   pairwise_h[0] = dtf::offset_t(0, 0);
   pairwise_h[1] = dtf::offset_t(1, 0);
   ts::binary_tree_array<feature> tree;
   pairwise_h.set_tree(tree);
   graph.move_factor(std::move(pairwise_h));
   priors.push_back(dtf::prior_normal(10.0, pairwise_h.get_weight_count()));

   std::vector<dtf::weight_t> weights0;
   for (int l1 = 0; l1 < label_count; ++l1) {
      for (int l2 = 0; l2 < label_count; ++l2) {
         weights0.push_back(0.3*std::fabs(static_cast<dtf::weight_t>(l1-l2)));
      }
   }
   std::for_each(weights0.begin(), weights0.end(), [&](dtf::weight_t w) { w_gt.push_back(w); });

   // Vertical
   dtf::factor<DataTraits, 2, feature> pairwise_v;
   pairwise_v[0] = dtf::offset_t(0, 0);
   pairwise_v[1] = dtf::offset_t(0, 1);
   pairwise_v.set_tree(tree);
   graph.move_factor(std::move(pairwise_v));
   std::for_each(weights0.begin(), weights0.end(), [&](dtf::weight_t w) { w_gt.push_back(w); });
   priors.push_back(dtf::prior_normal(10.0, pairwise_v.get_weight_count()));

   // Set ground truth weights
   dtf::dtf_set_all_weights(graph, w_gt.begin(), w_gt.end());

   // Dummy image
   ts::image<pixel_type> image(image_size, image_size);

   // MCMC/Gibbs test
	float mpm_ms = ts::timing_ms([&](){
		dtf::multmcmc<DataTraits> mcmc(graph, image, 3, 8, 1.0, 0.05);
		mcmc.perform_burnin(100000, 100, 1.01);
		auto posterior_marg = mcmc.perform_inference(500);
	});
	std::cout << "Gibbs inference in " << mpm_ms << "ms" << std::endl;

   // Obtain a fixed number of samples from the model
   dtf::inference::gibbs_chain<DataTraits> gibbs(graph, image);

	// 1. Burn-in sweeps
	std::cout << "Performing burn-in sweeps..." << std::endl;
   float ms = ts::timing_ms([&]() {
	   for (int burn_in = 0; burn_in < 2000; ++burn_in)
	   	gibbs.sweep();
      });
   std::cout << "... Done in " << ms << " ms." << std::endl;

   // 2. Sampling sweeps
   std::cout << "Performing sampling sweeps..." << std::endl;
   std::vector<ts::image<const unsigned char>> gt;
   gt.reserve(sample_count);
   for (int si = 0; si < sample_count; ++si) {
      for (int spacing = 0; spacing < 10; ++spacing)
         gibbs.sweep();
      gt.push_back(gibbs.state());
      //ts::save(gibbs.state(), L"label_gibbs.png", true, false);
   }

   // Create dataset from samples
   dataset_type ds(image_size, gt);

   // Reset model weights
   std::vector<dtf::weight_t> w_zero(w_gt.size());
   std::fill(w_zero.begin(), w_zero.end(), dtf::weight_t(0));
   dtf::dtf_set_all_weights(graph, w_zero.begin(), w_zero.end());

   // Optimize the factor weights
   std::cout << "Estimating model parameters from " << gt.size() << " samples..." << std::endl;
   dtf::learning::OptimizeWeights(graph, priors, ds, 1e-6, 50);

   // Extract learned weights
   std::vector<std::vector<dtf::weight_t>> f_weights(2);
   size_t fi = 0;
   dtf::visit_dtf_factors(graph,
      [&fi, &f_weights](const dtf::dtf_factor_base<DataTraits>& factor) -> void {
         size_t wsize = factor.energies_size();
         f_weights[fi].resize(wsize);
         std::copy(factor.energies(), factor.energies() + wsize, f_weights[fi].begin());
         fi += 1;
      });

   // Adjust the weights by adding a global constant, in order to have zero diagonal.
   // The reason why we need to do this to compare the absolute weights: the model is
   // not identifiable, and in fact adding any constant to all weights does not change
   // the model.
   adjust_weights(f_weights[0], label_count);
   adjust_weights(f_weights[1], label_count);

   std::vector<double> errors(2, 0.0);
   for (size_t i = 0; i < f_weights.size(); ++i) {
      std::cout << std::endl;
      std::cout << "Factor " << i << " weights:" << std::endl;
      size_t idx = 0;
      for (int l1 = 0; l1 < label_count; ++l1) {
         for (int l2 = 0; l2 < label_count; ++l2) {
            std::cout << "   " << f_weights[i][idx];
            errors[i] += std::pow(f_weights[i][idx]-weights0[idx], 2.0);
            idx += 1;
         }
         std::cout << std::endl;
      }
      errors[i] = std::sqrt(errors[i]);
      std::cout << "   # L2 norm error to ground truth: " << errors[i] << std::endl;
   }

   std::cout << std::endl;
   if (errors[0] < 0.1 && errors[1] < 0.1) {
      std::cout << "# Test case PASSED" << std::endl;
      return;
   } else {
      std::cout << "# Test case FAILED" << std::endl;
      throw std::exception();
   }
}
}
