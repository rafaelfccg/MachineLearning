#define _INTSAFE_H_INCLUDED_ // Avoid warnings in VS2010: http://connect.microsoft.com/VisualStudio/feedback/details/621653/including-stdint-after-intsafe-generates-warnings

/*
   Copyright Microsoft Corporation 2012
   
   Author: Sebastian Nowozin (senowozi)
   
   Date: 16 May 2012
*/

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdio>

#include "dataset.h"
#include "timing.h"
#include "dtf_mcmc.h"
#include "dtf.h"

#include "image.h"
#include "imageio.h"
#include "features.h"

// Image and data set types
typedef ts::dataset<2, ts::bgr24> Data;
typedef dtf::database_traits<Data> DataTraits;

// Features used
typedef ChineseCharacterInpainting::simple_feature TFeature;
typedef ChineseCharacterInpainting::simple_feature_sampler TFeatureSampler;

typedef ts::decision_tree<TFeature, DataTraits::label_count> DecisionTree;

// Learn a pairwise factor decision tree
void LearnPairwiseTree(const Data& train, TFeatureSampler& sampler, const std::array<dtf::offset_t, 2>& offsets,
   dtf::factor_graph<DataTraits>& graph, std::vector<dtf::prior_t>& priors,
   unsigned int feature_count, unsigned int levels_pw, int prep_cache_images, unsigned int min_samples,
   double sigma_pw)
{
   std::cout << "Learning decision tree for pairwise potential ("
      << offsets[0].x << "," << offsets[0].y << ") to ("
      << offsets[1].x << "," << offsets[1].y << ") ..." << std::endl;

   dtf::factor<DataTraits, 2, TFeature> factor(std::begin(offsets));
   auto tree = dtf::training::LearnDecisionTree(offsets, train, sampler,
      feature_count, levels_pw, prep_cache_images, min_samples);
   factor.set_tree(tree.tree());
   graph.push_back(factor);
   priors.push_back(dtf::prior_normal(sigma_pw, factor.get_weight_count()));
}

// sigma_unary, sigma_pw: prior regularization parameters.
static dtf::factor_graph<DataTraits> InitFactorGraph(std::vector<DecisionTree>& forest,
   std::vector<dtf::prior_t>& priors, const Data& train,
   double sigma_unary = 0.1, double sigma_pw = 1.0e-2,
   unsigned int levels_unary = 15, unsigned int levels_pw = 6)
{
   dtf::factor_graph<DataTraits> graph;

   // Setup feature sampler
   TFeatureSampler sampler_unary(80, 1, 21308U);
   TFeatureSampler sampler_pw(80, 2, 5829U);

   // General training parameters (as in ICCV 2011 paper)
   unsigned int feature_count = 2000;
   unsigned int min_samples = 16;
   size_t unary_tree_count = 1;
   size_t pw_tree_count = 1;

   int prep_cache_images = 1100;

   // Train a set of unary factors
   for (size_t ti = 0; ti < unary_tree_count; ++ti) {
      dtf::factor<DataTraits, 1, TFeature> factor;
      auto tree = dtf::training::LearnDecisionTreeUnary(train, sampler_unary, feature_count,
         levels_unary, prep_cache_images, min_samples);
      forest.push_back(tree);

      factor.set_tree(tree.tree());
      graph.push_back(factor);
      priors.push_back(dtf::prior_normal(sigma_unary, factor.get_weight_count()));
   }

   // Densely add horizontal and vertical edges at different ranges
   int dense_range = 2;

   for (size_t ti = 0; ti < pw_tree_count; ++ti)
   {
      for (int off_x = 0; off_x <= dense_range; ++off_x)
      {
         for (int off_y = -dense_range; off_y <= dense_range; ++off_y)
         {
            // Simple graph, no redundant pairwise potentials
            if (off_x == 0 && off_y <= 0)
               continue;

            std::array<dtf::offset_t, 2> offsets = { dtf::offset_t(0, 0), dtf::offset_t(off_x, off_y) };
            LearnPairwiseTree(train, sampler_pw, offsets, graph, priors,
               feature_count, levels_pw, prep_cache_images, min_samples, sigma_pw);
         }
      }
   }

   // ICCV 2011 style connectivity
   int local_offset = 3;
   int local_field = 3;
   for (int ry = -local_field; ry <= local_field; ++ry)
   {
      for (int rx = 0; rx <= local_field; ++rx)
      {
         if ((rx == 0 && ry == 0) || (rx == 0 && ry <= 0))
            continue;   // avoid duplicates

         std::array<dtf::offset_t, 2> offsets = { dtf::offset_t(0, 0),
            dtf::offset_t(rx*local_offset, ry*local_offset) };
         LearnPairwiseTree(train, sampler_pw, offsets, graph, priors,
            feature_count, levels_pw, prep_cache_images, min_samples, sigma_pw);
      }
   }

   return graph;
}

// Confusion matrix only on unobserved gray pixels
template <typename Data, typename _TClassifier>
ts::image<unsigned int> confusion_matrix_unobserved(const Data& database, const _TClassifier& classifier)
{
   ts::image<unsigned int> confusion(database.label_count, database.label_count);
   std::fill(confusion.begin(), confusion.end(), 0u);

   ts::thread_locals<ts::image<unsigned int>> locals(confusion);
   for (dtf::database_iterator<Data> i(database); i; ++i)
   {
      const auto input = i.training();
      auto labelling = classifier(input);
      const auto& gt = i.ground_truth();
      ts::parallel_for_each_pixel_xy(gt, [&](unsigned int x, unsigned int y)
      {
         const ts::bgr24& pix = input(x,y);
         if (pix[0] >= 64 && pix[0] <= 192) {
            (*locals)(labelling(x, y), gt(x, y))++;
         }
      });
   }
   locals.for_each([&](const ts::image<unsigned int>& local)
   {
      ts::parallel_for_each_pixel_xy(local, [&](unsigned int x, unsigned int y)
      {
         confusion(x, y) += local(x, y);
      });
   });
   return confusion;
}

template <typename Data, typename _TClassifier>
double classification_accuracy_unobserved(const Data& database, const _TClassifier& classifier)
{
   ts::image<unsigned int> confusion = confusion_matrix_unobserved(database, classifier);
   unsigned int total = 0, correct = 0;
   ts::for_each_pixel_xy(confusion, [&](unsigned int x, unsigned int y)
   {
      total += confusion(x, y);
      if (x == y)
         correct += confusion(x, y);
   });
   return static_cast<double>(correct) / total;
}

// Random forest classification
static double RFInference(const std::string& name, const std::vector<DecisionTree>& forest, const Data& data) {
   dtf::classify::decision_forest_classifier<DataTraits::label_count, TFeature> unary_rf(forest);
   double acc;
   float rf_ms = ts::timing_ms([&]() {
      acc = classification_accuracy_unobserved(data, unary_rf);
   });
   std::cout << "RF accuracy on '" << name << "': " << 100.0 * acc << "%   ("
      << (rf_ms/static_cast<float>(data.size())) << " ms/image)" << std::endl;
   return acc;
}

// Approximate MAP inference by simulated annealing
static double MAPInferenceSA(const std::string& name, const dtf::factor_graph<DataTraits>& graph, const Data& data) {
   dtf::classify::simulated_annealing_classifier<DataTraits> map_sa(graph, 300, 100.0, 0.05);
   double acc;
   float map_ms = ts::timing_ms([&]() {
      acc = classification_accuracy_unobserved(data, map_sa);
   });
   std::cout << "MAP (SA) accuracy on '" << name << "': " << 100.0 * acc << "%   ("
      << (map_ms/static_cast<float>(data.size())) << " ms/image)" << std::endl;
   return acc;
}

// Approximate maximum-posterior-marginal (MPM) inference by naive mean field approximation
static double MPMInferenceNMF(const std::string& name, const dtf::factor_graph<DataTraits>& graph, const Data& data) {
   dtf::classify::mpm_nmf_classifier<DataTraits> mpm(graph, 1.0e-5, 100);
   double acc;
   float map_ms = ts::timing_ms([&]() {
      acc = classification_accuracy_unobserved(data, mpm);
   });
   std::cout << "MPM (NMF) accuracy on '" << name << "': " << 100.0 * acc << "%   ("
      << (map_ms/static_cast<float>(data.size())) << " ms/image)" << std::endl;
   return acc;
}

// Convert marginal foreground distribution to a RGB image visualization
ts::image<ts::bgr24>
posterior_visualization(dtf::inference::naive_mean_field<DataTraits>::inference_result_type& posterior)
{
   ts::image<ts::bgr24> vis(posterior.width(), posterior.height());
   ts::parallel_for_each_pixel_xy(posterior, [&](int x, int y) -> void
   {
      ts::bgr24::value_type v = static_cast<ts::bgr24::value_type>(std::max(0.0, std::min(255.0, 255.0*posterior(x, y)[1])));
      ts::bgr24& out = vis(x, y);
      out[0] = out[1] = out[2] = v;
   });
   return vis;
}

// Compute count marginals from MCMC to floating point marginals
dtf::inference::naive_mean_field<DataTraits>::inference_result_type
count_to_probability(const dtf::multmcmc<DataTraits>::inference_result_type& count)
{
   dtf::inference::naive_mean_field<DataTraits>::inference_result_type marginals(count.width(), count.height());
   ts::parallel_for_each_pixel_xy(count, [&](int x, int y) -> void
   {
      const dtf::multmcmc<DataTraits>::marginal_type& mc = count(x, y);
      size_t sum = std::accumulate(mc.cbegin(), mc.cend(), 0);

      dtf::inference::naive_mean_field<DataTraits>::marginal_type& marg = marginals(x, y);
      for (dtf::label_t li = 0; li < DataTraits::label_count; ++li)
         marg[li] = static_cast<double>(mc[li]) / static_cast<double>(sum);
   });
   return marginals;
}

struct inference_results {
   double rf_train;
   double rf_test;
   double mapsa_train;
   double mapsa_test;
   double mpm_train;
   double mpm_test;
};

static struct inference_results Inference(const dtf::factor_graph<DataTraits>& graph,
   const std::vector<DecisionTree>& forest,
   const Data& train, const Data& test, bool output_images = false)
{
   if (output_images) {
      // Output test set predictions
      dtf::classify::decision_forest_classifier<DataTraits::label_count, TFeature> unary_rf(forest);
      dtf::classify::simulated_annealing_classifier<DataTraits> map_sa(graph, 300, 100.0, 0.5);
      dtf::classify::mpm_nmf_classifier<DataTraits> mpm(graph, 1.0e-5, 100);
      for (size_t idx = 0; idx < test.size(); ++idx) {
         const std::wstring& bname = test.get_basename(idx);
         std::wcout << "Predicting on '" << bname << "' ..." << std::endl;

         std::wstringstream input_filename;
         input_filename << bname << ".png";
         std::wstringstream gt_filename;
         gt_filename << bname << "_gt.png";
         std::wstringstream pred_rf_filename;
         pred_rf_filename << bname << "_pred_rf.png";
         std::wstringstream pred_mpm_filename;
         pred_mpm_filename << bname << "_pred_mpm.png";
         std::wstringstream pred_nmf_filename;
         pred_nmf_filename << bname << "_pred_nmf.png";
         std::wstringstream pred_sa_filename;
         pred_sa_filename << bname << "_pred_sa.png";
         std::wstringstream pred_mcmc_filename;
         pred_mcmc_filename << bname << "_pred_mcmc.png";

         auto input = test.get_training(idx);
         ts::save(input, input_filename.str(), true, false);
         ts::save(test.get_ground_truth(idx), gt_filename.str(), true, false);
         ts::save(unary_rf(input), pred_rf_filename.str(), true, false);
         ts::save(map_sa(input), pred_sa_filename.str(), true, false);

         auto mpm_marg = mpm.get_mpm_and_marginals(input);
         ts::save(mpm_marg.first, pred_mpm_filename.str(), true, false);
         ts::save(posterior_visualization(mpm_marg.second), pred_nmf_filename.str(), false, true);

#if 0
         // Expensive MCMC inference (multiple chains, multiple temperatures)
         dtf::multmcmc<DataTraits> mcmc(graph, input, 4, 24, 20000.0);
         mcmc.perform_burnin(3000, 200, 1.1);
         auto mcmc_res = mcmc.perform_inference(1000);
         ts::save(posterior_visualization(count_to_probability(mcmc_res)), pred_mcmc_filename.str(), false, true);
#endif
      }
   }

   // First, get test performances
   struct inference_results res;

   res.rf_test = RFInference("test", forest, test);
   res.mapsa_test = MAPInferenceSA("test", graph, test);
   res.mpm_test = MPMInferenceNMF("test", graph, test);

   res.rf_train = RFInference("train", forest, train);
   res.mapsa_train = MAPInferenceSA("train", graph, train);
   res.mpm_train = MPMInferenceNMF("train", graph, train);

   return res;
}

static void Usage(void)
{
   std::cout << "Usage: ChineseCharacterInpainting.exe <model.graph> [sigma_unary] [sigma_pw] "
      << "[levels_unary] [levels_pw] [result_file] [output_image_flag]" << std::endl;
   std::cout << std::endl;
   std::cout << " model.graph: Model output file.  If it already exists, training will be skipped." << std::endl;
   std::cout << " sigma_unary: Prior sigma parameter for unary factors (>0.0), default: 1.0" << std::endl;
   std::cout << "    sigma_pw: Prior sigma parameter for pairwise factors (>0.0), default: 1.0e-2" << std::endl;
   std::cout << "levels_unary: Maximum tree depth for unary factors (>0), default: 15" << std::endl;
   std::cout << "   levels_pw: Maximum tree depth for pairwise factors (>0), default: 6" << std::endl;
   std::cout << " result_file: Filename to write summary of train/test performances to.  Default: none." << std::endl;
   std::cout << "output_image_flag: If given then test images are written." << std::endl;
   std::cout << std::endl;
}

void main(int argc, const char* argv[])
{
   // Load dataset
   Data chinese_train(L"..\\data\\chinese-inpaint-small", L"train", 0.5);
   std::cout << "TRAIN: " << chinese_train.size() << " images" << std::endl;
   Data chinese_test(L"..\\data\\chinese-inpaint-small", L"test", 1.0);
   std::cout << "TEST: " << chinese_test.size() << " images" << std::endl;

   // Forest and DTF
   std::vector<DecisionTree> forest;
   dtf::factor_graph<DataTraits> graph;

   // Parse arguments
   if (argc < 2) {
      Usage();
      return;
   }
   std::string model_file(argv[1]);
   std::cout << "Model file: '" << model_file << "'" << std::endl;

   // Parameters
   double sigma_unary = 1.0;
   double sigma_pw = 1.0e-2;
   if (argc >= 4) {
      bool parse_success = sscanf_s(argv[2], "%lf", &sigma_unary) == 1;
      parse_success &= sscanf_s(argv[3], "%lf", &sigma_pw) == 1;
      if (parse_success == false || sigma_unary <= 0.0 || sigma_pw <= 0.0) {
         std::cerr << "ERROR: Failed to parse sigma_unary or sigma_pw argument." << std::endl;
         std::cerr << std::endl;
         Usage();
         return;
      }
   }

   unsigned int levels_unary = 15;
   unsigned int levels_pw = 6;
   if (argc >= 6)
   {
      bool parse_success = sscanf_s(argv[4], "%d", &levels_unary) == 1;
      parse_success &= sscanf_s(argv[5], "%d", &levels_pw) == 1;
      if (parse_success == false || levels_unary <= 0 || levels_pw <= 0) {
         std::cerr << "ERROR: Failed to parse levels_unary or levels_pw argument." << std::endl;
         std::cerr << std::endl;
         Usage();
         return;
      }
   }

   std::string result_file;
   if (argc >= 7)
      result_file = std::string(argv[6]);
   std::cout << "Successfully parsed all arguments." << std::endl;

   bool output_images = false; // Write test set predictions of different methods as PNG files
   if (argc >= 8)
      output_images = true;

   try {
      std::ifstream ifile(model_file.c_str(), std::ios::binary | std::ios_base::in);

      // Add factor types to the factory prior to load
      ts::factory<dtf::factor_base<DataTraits>> factory;
      factory.add_type<dtf::factor<DataTraits, 1, TFeature>>();
      factory.add_type<dtf::factor<DataTraits, 2, TFeature>>();

      // Load data from the file
      graph.load(ifile, factory);
      ifile >> forest;
      std::cout << "Successfully deserialized model from file '" << model_file << "'." << std::endl;
   }
   catch (std::exception&)
   {
      std::cout << "Training model." << std::endl;
      std::cout << "Parameter summary" << std::endl;
      std::cout << "   Prior sigma: unary " << sigma_unary << ", pairwise " << sigma_pw << std::endl;
      std::cout << "   Tree levels: unary " << levels_unary << ", pairwise " << levels_pw << std::endl;
      std::cout << std::endl;

      float est_ms;
      float training_ms = ts::timing_ms([&]() {

         // Declare graph of factor types and build the factor structure
         std::vector<dtf::prior_t> priors;
         float tree_ms = ts::timing_ms([&]() {
            graph = InitFactorGraph(forest, priors, chinese_train, sigma_unary, sigma_pw, levels_unary, levels_pw);
         });
         std::cout << "Decision tree induction took " << (tree_ms/1000.0) << "s" << std::endl;

         // Optimize the DTF weights
         std::cout << "Estimating model parameters (" << dtf::dtf_count_weights(graph)
            << " parameters in " << dtf::dtf_count_factors(graph) << " factor types)" << std::endl;
         std::cout << "Total number of pseudolikelihood terms: "
            << dtf::compute::CountTotalPixels(chinese_train) << std::endl;

         est_ms = ts::timing_ms([&]() {
            dtf::learning::OptimizeWeights(graph, priors, chinese_train, 1.0e-5, 100, 100);
         });
      });
      std::cout << "Estimation of " << dtf::dtf_count_weights(graph)
         << " parameters took " << (est_ms/1000.0) << "s" << std::endl;
      std::cout << "Total training time: " << 0.001f * training_ms << "s." << std::endl;

      // Save data to disk
      std::ofstream ofile(model_file.c_str(),
         std::ios::binary | std::ios_base::out | std::ios_base::trunc);
      graph.save(ofile);
      ofile << forest;
      std::cout << "Successfully serialized model to file." << std::endl;
   }

   // Perform test-time classifications and compute error metrics
   struct inference_results res;
   float inference_ms = ts::timing_ms([&]() { res = Inference(graph, forest, chinese_train, chinese_test, output_images); });
   std::cout << "Total inference time: " << 0.001f * inference_ms << "s." << std::endl;

   if (result_file.empty() == false) {
      std::ofstream resfile(result_file.c_str());
      resfile << sigma_unary << " " << sigma_pw << " " << levels_unary << " " << levels_pw << " "
         << res.rf_train << " " << res.mapsa_train << " " << res.mpm_train << " "
         << res.rf_test << " " << res.mapsa_test << " " << res.mpm_test << std::endl;
      resfile.close();
      std::cout << "Wrote performance results to file '" << result_file << "'." << std::endl;
   }

}
