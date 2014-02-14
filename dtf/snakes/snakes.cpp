#include "util.h"
#include "timing.h"
#include <fstream>

namespace Snakes
{

template <unsigned char _length>
class Database : public CachedDatabase<_length, ts::bgr24>
{
public:
   Database(const std::wstring& folder, unsigned int start, unsigned int stop) : 
      CachedDatabase<_length, ts::bgr24>(std::wstring(folder).append(L"/images/case%04d.png"), 
                                         std::wstring(folder).append(L"/labels/case%04d.png"), 
                                         start, stop) {}
};

typedef Database<11> Data;
typedef dtf::database_traits<Data> DataTraits;

// Carsten's feature that tests for membership in a voxel of a 2x2x2 RGB cube
class RgbValueFeature
{
public:
   static const ts::image<ts::bgr24>& pre_process(const ts::image<ts::bgr24>& input) { return input; }

   RgbValueFeature() : m_x(0), m_y(0), m_voxel(0), m_variable(0) {}
   RgbValueFeature(int variable, int channel, int x, int y) : m_variable(variable), m_x(x), m_y(y)
   {
      std::array<int, 5> voxels = { 1, 2, 4, 3, 6 };
      m_voxel = voxels[channel];
   }

   template <typename _It>
   bool operator()(unsigned int x, unsigned int y, const ts::image<ts::bgr24>& prep, _It offsets) const
   {
      // Return true iff the colour value at the relevant position is in our particular voxel of a 2x2x2 RGB cube
      int xx = clamp<int>(m_x + x + offsets[m_variable].x, 0, prep.width() - 1);
      int yy = clamp<int>(m_y + y + offsets[m_variable].y, 0, prep.height() - 1);
      ts::bgr24 pix = prep(xx, yy);
      int voxel = ((pix[2] >> 5) & 0x4) | ((pix[1] >> 6) & 0x2) | (pix[0] >> 7);
      return voxel == m_voxel;
   }
private:
   int m_x, m_y, m_voxel, m_variable;
};

class RgbValueFeatureSampler
{
public:
   RgbValueFeatureSampler(int shift, int cycle = 1) : m_channel(0), m_count(0), m_max_shift(shift), m_cycle(cycle) {}

   template <typename PrepCache>
   std::vector<RgbValueFeature> operator()(PrepCache& cache, unsigned int count)
   {
      std::vector<RgbValueFeature> rv(count);
      for (unsigned int i = 0; i < count; i++)
      {
         if (++m_count == m_cycle)
         {
            m_count = 0;
            if (++m_channel == 5)
               m_channel = 0;
         }
         int shift_x = irand(-m_max_shift, m_max_shift + 1);
         int shift_y = irand(-m_max_shift, m_max_shift + 1);
         rv[i] = RgbValueFeature(m_count, m_channel, shift_x, shift_y);
      }
      return rv;
   }
private:
   const int m_max_shift, m_cycle;
   int m_channel, m_count;
};

typedef ts::decision_tree<RgbValueFeature, DataTraits::label_count> DecisionTree;

dtf::factor_graph<DataTraits> InitFactorGraph(
                     std::vector<DecisionTree>& forest,
                     std::vector<dtf::prior_t>& priors,
                     const Data& train)
{
   dtf::factor_graph<DataTraits> graph;

   // Add some unary factors to the graph
   RgbValueFeatureSampler feature_sampler_unary(8);
   for (int i = 0; i < 4; i++)
   {
      dtf::factor<DataTraits, 1, RgbValueFeature> factor;
      DecisionTree trained_tree = dtf::training::LearnDecisionTreeUnary(train, feature_sampler_unary, 500, 20, 200, 8);
      forest.push_back(trained_tree);
      factor.set_tree(trained_tree.tree());
      priors.push_back(dtf::prior_normal(0.08, factor.get_weight_count()));
      graph.push_back(factor);
   }

   // Add some pairwise factors
   RgbValueFeatureSampler feature_sampler_pairwise(3, 2);
   for (int i = 0; i < 4; i++)
   {
      { // Horizontal pair
         std::array<dtf::offset_t, 2> offsets = { dtf::offset_t(0, 0), dtf::offset_t(1, 0) };
         dtf::factor<DataTraits, 2, RgbValueFeature> factor(std::begin(offsets));
         auto trained_tree = dtf::training::LearnDecisionTree(offsets, train, feature_sampler_pairwise, 500, 15, 200, 8);
         factor.set_tree(trained_tree.tree());
         priors.push_back(dtf::prior_normal(0.09, factor.get_weight_count()));
         graph.push_back(factor);
      }
      { // Vertical pair
         std::array<dtf::offset_t, 2> offsets = { dtf::offset_t(0, 0), dtf::offset_t(0, 1) };
         dtf::factor<DataTraits, 2, RgbValueFeature> factor(std::begin(offsets));
         auto trained_tree = dtf::training::LearnDecisionTree(offsets, train, feature_sampler_pairwise, 500, 15, 200, 8);
         factor.set_tree(trained_tree.tree());
         priors.push_back(dtf::prior_normal(0.09, factor.get_weight_count()));
         graph.push_back(factor);
      }
   }
   return graph;
}

void Inference(const dtf::factor_graph<DataTraits>& graph, const std::vector<DecisionTree>& forest, const Data& train, const Data& test)
{
   // MAP inference on unary factors via arg-min
   double acc_train, acc_test;
   dtf::classify::unaries_MAP_classifier<DataTraits> unary_pl(graph);
   float ms = ts::timing_ms([&]() {
      acc_train = dtf::classify::classification_accuracy(train, unary_pl);
      acc_test = dtf::classify::classification_accuracy(test, unary_pl); });
   std::cout << "PL Unary: Accuracy: Train " << 100.0 * acc_train << "%; Test " << 100.0 * acc_test << "%. Time " << ms << " ms" << std::endl;

   // MAP inference on whole factor graph via TRW
   dtf::classify::pairwise_map_classifier<DataTraits> pairwise_pl(graph, 200, 1e-5);
   ms = ts::timing_ms([&]() {
      acc_train = dtf::classify::classification_accuracy(train, pairwise_pl);
      acc_test = dtf::classify::classification_accuracy(test, pairwise_pl); });
   std::cout << "PL Pairwise: Accuracy: Train " << 100.0 * acc_train << "%; Test " << 100.0 * acc_test << "%. Time " << ms << " ms" << std::endl;

   // Random forest classification
   dtf::classify::decision_forest_classifier<DataTraits::label_count, RgbValueFeature> unary_rf(forest);
   ms = ts::timing_ms([&]() {
      acc_train = dtf::classify::classification_accuracy(train, unary_rf);
      acc_test = dtf::classify::classification_accuracy(test, unary_rf); });
   std::cout << "RF Unary: Accuracy: Train " << 100.0 * acc_train << "%; Test " << 100.0 * acc_test << "%. Time " << ms << " ms" << std::endl;

   // Simulated annealing
   dtf::classify::simulated_annealing_classifier<DataTraits> sa(graph);
   ms = ts::timing_ms([&]() {
      acc_train = dtf::classify::classification_accuracy(train, sa);
      acc_test = dtf::classify::classification_accuracy(test, sa); });
   std::cout << "SA: Accuracy: Train " << 100.0 * acc_train << "%; Test " << 100.0 * acc_test << "%. Time " << ms << " ms" << std::endl;

   // Save some output images
   auto input = train.get_training(0);
   ts::save(unary_rf(input), L"label_rf.png", true, false);
   ts::save(unary_pl(input), L"label_pl.png", true, false);
   ts::save(pairwise_pl(input), L"label_trw.png", true, false);
   ts::save(sa(input), L"label_sa.png", true, false);
}

void Run()
{
   // Define training and test data sources
   Data snakes_train(L"../data/SnakeColorArrowsLength10Lock", 1, 201);
   Data snakes_test( L"../data/SnakeColorArrowsLength10Lock", 201, 301);
   std::string filename("snakes.graph");

   std::vector<DecisionTree> forest;
   dtf::factor_graph<DataTraits> graph;

   try
   {
      // Load from file if it is available, otherwise throw
      std::ifstream ifile(filename, std::ios::binary | std::ios_base::in);

      // Add factor types to the factory prior to load
      ts::factory<dtf::factor_base<DataTraits>> factory;
      factory.add_type<dtf::factor<DataTraits, 1, RgbValueFeature>>();
      factory.add_type<dtf::factor<DataTraits, 2, RgbValueFeature>>();

      // Load data from the file
      graph.load(ifile, factory);
      ifile >> forest;
   }
   catch (std::exception&)
   {
      // Declare graph of factor types and build the factor structure
      std::vector<dtf::prior_t> priors;
      graph = InitFactorGraph(forest, priors, snakes_train);

      // Optimize the DTF weights
      dtf::learning::OptimizeWeights(graph, priors, snakes_train, 1e-5, 300);

      // Save data to disk
      std::ofstream ofile(filename, std::ios::binary | std::ios_base::out | std::ios_base::trunc);
      graph.save(ofile);
      ofile << forest;
   }

   // Perform test-time classifications and compute error metrics
   Inference(graph, forest, snakes_train, snakes_test);
}

}

void main(int argc, const wchar_t* argv[])
{
   Snakes::Run();
}
