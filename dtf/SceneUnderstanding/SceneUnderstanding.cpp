#define _INTSAFE_H_INCLUDED_ // Avoid warnings in VS2010: http://connect.microsoft.com/VisualStudio/feedback/details/621653/including-stdint-after-intsafe-generates-warnings

/*
   Copyright Microsoft Corporation 2012
   
   Author: Sebastian Nowozin (senowozi)
   
   Date: 17th October 2012
*/

#include <iostream>
#include <fstream>
#include <cstdio>

#include "dtf_mcmc.h"

#include "dataset.h"
#include "features.h"
#include "combinedfeature.h"
#include "timing.h"

#include "image.h"
#include "imageio.h"
#include "gradientmap.h"

typedef ts::dataset<8, ts::bgr24> Data;
typedef dtf::database_traits<Data> DataTraits;

// Use four basic feature types
typedef SceneUnderstanding::simple_pixel_feature TPixelFeature;
typedef SceneUnderstanding::location_feature TLocationFeature;

//#define EXPENSIVE_FEATURES 1

#ifdef EXPENSIVE_FEATURES
typedef SceneUnderstanding::color_feature TColorFeature;
typedef SceneUnderstanding::oriented_gradient_feature<8> TGradientFeature;
typedef combined_feature<
   combined_feature<TPixelFeature, TLocationFeature>,
   combined_feature<TGradientFeature, TColorFeature>> TUnaryFeature;
#else
typedef combined_feature<TPixelFeature, TLocationFeature> TUnaryFeature;
#endif

// Pairwise factor features
typedef SceneUnderstanding::contrast_feature TContrastFeature;
#ifdef EXPENSIVE_FEATURES
typedef combined_feature<
   combined_feature<TContrastFeature, TLocationFeature>,
   combined_feature<TGradientFeature, TColorFeature>> TPairwiseFeature;
#else
typedef combined_feature<
   combined_feature<TContrastFeature, TLocationFeature>, TPixelFeature> TPairwiseFeature;
#endif

typedef ts::decision_tree<TUnaryFeature, DataTraits::label_count> DecisionTree;

// sigma_unary, sigma_pw: prior regularization parameters.
// levels_unary, levels_pw: maximum tree depth for unary and pairwise trees.
static dtf::factor_graph<DataTraits> InitFactorGraph(std::vector<DecisionTree>& forest,
   std::vector<dtf::prior_t>& priors, const Data& train,
   double sigma_unary = 0.1, double sigma_pw = 1.0e-2,
   unsigned int levels_unary = 16, unsigned int levels_pw = 6,
   int feat_box_max = 8, double feat_radius = 5.0)
{
   dtf::factor_graph<DataTraits> graph;

#ifdef EXPENSIVE_FEATURES
   // Setup feature sampler
   SceneUnderstanding::simple_pixel_feature_sampler sampler_pixelfeat(feat_box_max, feat_radius);
   SceneUnderstanding::color_feature_sampler sampler_colorfeat;
   SceneUnderstanding::oriented_gradient_feature_sampler<8> sampler_gradfeat(feat_box_max, feat_radius);
   SceneUnderstanding::location_feature_sampler sampler_locationfeat;

   // Unary sampler
   combined_feature_sampler<decltype(sampler_pixelfeat), decltype(sampler_locationfeat)>
      sampler_comb1(sampler_pixelfeat, sampler_locationfeat, 0.5);
   combined_feature_sampler<decltype(sampler_gradfeat), decltype(sampler_colorfeat)>
      sampler_comb2(sampler_gradfeat, sampler_colorfeat, 0.5);
   combined_feature_sampler<decltype(sampler_comb1), decltype(sampler_comb2)>
      sampler_unary(sampler_comb1, sampler_comb2, 0.5);

   // Pairwise sampler
   SceneUnderstanding::contrast_feature_sampler sampler_contrastfeat;
   combined_feature_sampler<decltype(sampler_contrastfeat), decltype(sampler_locationfeat)>
      sampler_pw_comb1(sampler_contrastfeat, sampler_locationfeat, 0.5);
   combined_feature_sampler<decltype(sampler_pw_comb1), decltype(sampler_comb2)>
      sampler_pw(sampler_pw_comb1, sampler_comb2, 0.5);
#else
   // Unary sampler
   SceneUnderstanding::simple_pixel_feature_sampler sampler_pixelfeat(feat_box_max, feat_radius);
   SceneUnderstanding::location_feature_sampler sampler_locationfeat;
   combined_feature_sampler<decltype(sampler_pixelfeat), decltype(sampler_locationfeat)>
      sampler_unary(sampler_pixelfeat, sampler_locationfeat, 0.5);

   // Pairwise sampler
   SceneUnderstanding::contrast_feature_sampler sampler_contrastfeat;
   combined_feature_sampler<decltype(sampler_contrastfeat), decltype(sampler_locationfeat)>
      sampler_pw_comb1(sampler_contrastfeat, sampler_locationfeat, 0.5);
   combined_feature_sampler<decltype(sampler_pw_comb1), decltype(sampler_pixelfeat)>
      sampler_pw(sampler_pw_comb1, sampler_pixelfeat, 0.5);
#endif

   // General training parameters
   unsigned int feature_count = 1024;
   unsigned int min_samples = 16;
   int prep_cache_images = 1;

   // Train a set of unary factors
   size_t unary_tree_count = 1;
   for (size_t ti = 0; ti < unary_tree_count; ++ti) {
      // Combined features
      dtf::factor<DataTraits, 1, TUnaryFeature> factor;
      auto tree = dtf::training::LearnDecisionTreeUnary(train, sampler_unary, feature_count,
         levels_unary, prep_cache_images, min_samples);
      forest.push_back(tree);

      factor.set_tree(tree.tree());
      graph.push_back(factor);
      priors.push_back(dtf::prior_normal(sigma_unary, factor.get_weight_count()));
   }
   if (levels_pw <= 0)
      return graph;

   // Add horizontal and vertical edges at different ranges
   size_t pw_tree_count = 1;
   std::vector<dtf::offset_t> pairwise_pots;
   pairwise_pots.push_back(dtf::offset_t(0, 1));
   pairwise_pots.push_back(dtf::offset_t(1, 0));
   pairwise_pots.push_back(dtf::offset_t(1, 1));
   pairwise_pots.push_back(dtf::offset_t(1, -1));

   std::vector<int> ranges;
   ranges.push_back(1);
// ranges.push_back(15);

   for (size_t ti = 0; ti < pw_tree_count; ++ti)
   {
      for (size_t ri = 0; ri < ranges.size(); ++ri)
      {
         for (size_t pi = 0; pi < pairwise_pots.size(); ++pi)
         {
            int off_x = pairwise_pots[pi].x*ranges[ri];
            int off_y = pairwise_pots[pi].y*ranges[ri];
            std::array<dtf::offset_t, 2> offsets = { dtf::offset_t(0, 0), dtf::offset_t(off_x, off_y) };
            std::cout << "Learning decision tree for pairwise potential to ("
               << off_x << "," << off_y << ") ..." << std::endl;

            dtf::factor<DataTraits, 2, TPairwiseFeature> factor(std::begin(offsets));
            auto tree = dtf::training::LearnDecisionTree(offsets, train, sampler_pw,
               feature_count, levels_pw, prep_cache_images, min_samples);
            factor.set_tree(tree.tree());
            graph.push_back(factor);
            priors.push_back(dtf::prior_normal(sigma_pw, factor.get_weight_count()));
         }
      }
   }
   return graph;
}

// Random forest classification
static double RFInference(const std::string& name, const std::vector<DecisionTree>& forest, const Data& data)
{
   dtf::classify::decision_forest_classifier<DataTraits::label_count, TUnaryFeature> unary_rf(forest);
   double acc;
   float rf_ms = ts::timing_ms([&]() {
      acc = dtf::classify::classification_accuracy(data, unary_rf);
   });
   std::cout << "RF accuracy on '" << name << "': " << 100.0 * acc << "%   ("
      << (rf_ms/static_cast<float>(data.size())) << " ms/image)" << std::endl;
   return acc;
}

// Approximate MAP inference by simulated annealing
static double MAPInferenceSA(const std::string& name, const dtf::factor_graph<DataTraits>& graph, const Data& data)
{
   dtf::classify::simulated_annealing_classifier<DataTraits> map_sa(graph, 250);
   double acc;
   float map_ms = ts::timing_ms([&]() {
      acc = dtf::classify::classification_accuracy(data, map_sa);
   });
   std::cout << "MAP (SA) accuracy on '" << name << "': " << 100.0 * acc << "%   ("
      << (map_ms/static_cast<float>(data.size())) << " ms/image)" << std::endl;
   return acc;
}

// Approximate maximum-posterior-marginal (MPM) inference by naive mean field approximation
static double MPMInferenceNMF(const std::string& name, const dtf::factor_graph<DataTraits>& graph, const Data& data)
{
   dtf::classify::mpm_nmf_classifier<DataTraits> mpm(graph, 1.0e-5, 40);
   double acc;
   float map_ms = ts::timing_ms([&]() {
      acc = dtf::classify::classification_accuracy(data, mpm);
   });
   std::cout << "MPM (NMF) accuracy on '" << name << "': " << 100.0 * acc << "%   ("
      << (map_ms/static_cast<float>(data.size())) << " ms/image)" << std::endl;
   return acc;
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
   // Output some test set predictions
   if (output_images) {
      dtf::classify::decision_forest_classifier<DataTraits::label_count, TUnaryFeature> unary_rf(forest);

      for (size_t idx = 0; idx < test.size(); ++idx) {
         const std::wstring& bname = test.get_basename(idx);

         std::wstringstream gt_filename;
         gt_filename << bname << "_gt.png";

         std::wstringstream pred_rf_filename;
         pred_rf_filename << bname << "_pred_rf.png";

         auto input = test.get_training(idx);
         ts::save(test.get_ground_truth(idx), gt_filename.str(), true, false);
         ts::save(unary_rf(input), pred_rf_filename.str(), true, false);
      }

      dtf::classify::simulated_annealing_classifier<DataTraits> map_sa(graph, 100);
      dtf::classify::mpm_nmf_classifier<DataTraits> mpm(graph, 1.0e-5, 40);
      for (size_t idx = 0; idx < test.size(); ++idx) {
         const std::wstring& bname = test.get_basename(idx);

         std::wstringstream pred_filename;
         pred_filename << bname << "_pred_mpm.png";
         std::wstringstream pred_sa_filename;
         pred_sa_filename << bname << "_pred_sa.png";

         auto input = test.get_training(idx);
         ts::save(mpm(input), pred_filename.str(), true, false);
         ts::save(map_sa(input), pred_sa_filename.str(), true, false);
      }
   }

   struct inference_results res;
   res.rf_train = RFInference("train", forest, train);
   res.rf_test = RFInference("test", forest, test);

   res.mapsa_train = MAPInferenceSA("train", graph, train);
   res.mapsa_test = MAPInferenceSA("test", graph, test);

   res.mpm_train = MPMInferenceNMF("train", graph, train);
   res.mpm_test = MPMInferenceNMF("test", graph, test);

   return res;
}

void GradientMapTest(void)
{
   // Load a RGB image
   typedef ts::bgr24 pixel_type;
   auto rgb_image = ts::load<pixel_type>(L"..\\data\\dags-regions\\images\\1001794.jpg");

   float gmap_compute_ms = ts::timing_median_ms(30, [&](){
      SceneUnderstanding::gradientmap<pixel_type, 8u> gmap(rgb_image);
      double resp = gmap.compute_mean_response(2, 20, 30, 60, 35);
   });
   std::cout << "Gradient map computation time for a " << rgb_image.width()
      << "-by-" << rgb_image.height() << " image: " << gmap_compute_ms << "ms." << std::endl;

   //ts::save(gibbs.state(), L"label_gibbs.png", true, false);
}

static void Usage(void)
{
   std::cout << "Usage: SceneUnderstanding.exe <model.graph> [sigma_unary] [sigma_pw] "
      << "[levels_unary] [levels_pw] [feat_box_max] [feat_radius] [result_file]" << std::endl;
   std::cout << std::endl;
   std::cout << " model.graph: Model output file.  If it already exists, training will be skipped." << std::endl;
   std::cout << " sigma_unary: Prior sigma parameter for unary factors (>0.0), default: 1.0" << std::endl;
   std::cout << "    sigma_pw: Prior sigma parameter for pairwise factors (>0.0), default: 0.005" << std::endl;
   std::cout << "levels_unary: Maximum tree depth for unary factors (>0), default: 18" << std::endl;
   std::cout << "   levels_pw: Maximum tree depth for pairwise factors (>0), default: 8" << std::endl;
   std::cout << "feat_box_max: Maximum box side length for integral image features (>0), default: 4" << std::endl;
   std::cout << " feat_radius: Feature test radius multiplier (offset from factor), (>0), default: 5.0" << std::endl;
   std::cout << " result_file: Filename to write summary of train/test performances to.  Default: none." << std::endl;
   std::cout << std::endl;
}

void main(int argc, const char* argv[])
{
#if 0
   // FIXME: move this test to a separate test case
   GradientMapTest();
#endif

   // Parse arguments
   if (argc < 2) {
      Usage();
      return;
   }
   std::string model_file(argv[1]);
   std::cout << "Model file: '" << model_file << "'" << std::endl;

   // Essential training parameters
   double sigma_unary = 1.0;
   double sigma_pw = 0.005;
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
   unsigned int levels_unary = 18;
   unsigned int levels_pw = 8;
   if (argc >= 6) {
      bool parse_success = sscanf_s(argv[4], "%d", &levels_unary) == 1;
      parse_success &= sscanf_s(argv[5], "%d", &levels_pw) == 1;
      if (parse_success == false || levels_unary <= 0 || levels_pw < 0) {
         std::cerr << "ERROR: Failed to parse levels_unary or levels_pw argument." << std::endl;
         std::cerr << std::endl;
         Usage();
         return;
      }
   }

   int feat_box_max = 4;
   double feat_radius = 5.0;
   if (argc >= 8) {
      bool parse_success = sscanf_s(argv[6], "%d", &feat_box_max) == 1;
      parse_success &= sscanf_s(argv[7], "%lf", &feat_radius) == 1;
      if (parse_success == false || feat_box_max <= 0 || feat_radius <= 0) {
         std::cerr << "ERROR: Failed to parse feat_box_max or feat_radius argument." << std::endl;
         std::cerr << std::endl;
         Usage();
         return;
      }
   }

   std::string result_file;
   if (argc >= 9)
      result_file = std::string(argv[8]);
   std::cout << "Successfully parsed all arguments." << std::endl;

   // Load dataset
   std::cout << "Attempting to load data set.  If program fails, please read the README file."
      << std::endl;
   Data dags_train(L"..\\data\\dags-regions", L"fold-1-train", 0.1);
   std::cout << "TRAIN: " << dags_train.size() << " images" << std::endl;
   Data dags_test(L"..\\data\\dags-regions", L"fold-1-test", 1.0);
   std::cout << "TEST: " << dags_test.size() << " images" << std::endl;

   std::vector<DecisionTree> forest;
   dtf::factor_graph<DataTraits> graph;

   try {
      std::ifstream ifile(model_file.c_str(), std::ios::binary | std::ios_base::in);

      // Add factor types to the factory prior to load
      ts::factory<dtf::factor_base<DataTraits>> factory;
      factory.add_type<dtf::factor<DataTraits, 1, TUnaryFeature>>();
      factory.add_type<dtf::factor<DataTraits, 2, TUnaryFeature>>();
      factory.add_type<dtf::factor<DataTraits, 2, TPairwiseFeature>>();

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
      std::cout << "       Feature: box size " << feat_box_max << ", radius " << feat_radius << std::endl;
      std::cout << std::endl;

      float est_ms;
      float training_ms = ts::timing_ms([&]() {

         // Declare graph of factor types and build the factor structure
         std::vector<dtf::prior_t> priors;
         float tree_ms = ts::timing_ms([&]() {
            graph = InitFactorGraph(forest, priors, dags_train, sigma_unary, sigma_pw,
               levels_unary, levels_pw, feat_box_max, feat_radius);
         });
         std::cout << "Decision tree induction took " << (tree_ms/1000.0) << "s" << std::endl;

         if (levels_pw <= 0) {
            std::ofstream ofile(model_file.c_str(),
               std::ios::binary | std::ios_base::out | std::ios_base::trunc);
            ofile << forest;
            std::cout << "Successfully serialized forest to file." << std::endl;

            double acc_test = RFInference("test", forest, dags_test);
            double acc_train = RFInference("train", forest, dags_train);

            std::exit(0);
         }


         // Optimize the DTF weights
         std::cout << "Estimating model parameters (" << dtf::dtf_count_weights(graph)
            << " parameters in " << dtf::dtf_count_factors(graph) << " factor types)" << std::endl;
         std::cout << "Total number of pseudolikelihood terms: "
            << dtf::compute::CountTotalPixels(dags_train) << std::endl;

         est_ms = ts::timing_ms([&]() {
            dtf::learning::OptimizeWeights(graph, priors, dags_train, 1.0e-5, 100, 50);
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
   float inference_ms = ts::timing_ms([&]()
   {
      res = Inference(graph, forest, dags_train, dags_test, true);
   });
   std::cout << "Total inference time: " << 0.001f * inference_ms << "s." << std::endl;

   if (result_file.empty() == false) {
      std::ofstream resfile(result_file.c_str());
      resfile << levels_unary << " " << levels_pw << " " << sigma_unary << " " << sigma_pw << " "
         << feat_box_max << " " << feat_radius << " "
         << res.rf_train << " " << res.mapsa_train << " " << res.mpm_train << " "
         << res.rf_test << " " << res.mapsa_test << " " << res.mpm_test << std::endl;
      resfile.close();
      std::cout << "Wrote performance results to file '" << result_file << "'." << std::endl;
   }
}
