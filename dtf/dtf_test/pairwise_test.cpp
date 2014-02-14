#include "dtf.h"
#include "imageio.h"

namespace PairwiseTest
{

class MyTrainingDatabase // Dummy training database
{
public:
   static const dtf::label_t label_count = 2;
   MyTrainingDatabase() : m_image(2, 1) { m_image(0, 0) = 0; m_image(1, 0) = 1; }
   size_t size() const { return 1; }
   ts::image<unsigned char> get_training(size_t index) const { return m_image; }
   ts::image<unsigned char> get_ground_truth(size_t index) const { return m_image; }
   dtf::dense_sampling get_samples(size_t index) const { return dtf::all_pixels(m_image); }
protected:
   ts::image<unsigned char> m_image;
};

class LookAtFeature // Dummy feature
{
public:
   static ts::image<unsigned char> pre_process(const ts::image<unsigned char>& image) { return image; }
   template <typename _It> bool operator()(unsigned int x, unsigned int y, const ts::image<unsigned char>& data, _It off) const
      { return data(x, y) > 0; }
};

// Effective conditional energies of the first unary factor are:
//    E_0 = [0.55, 0.40],
// and of the second unary factor, at (1,0):
//    E_1 = [0.90, 0.20].
//
// Data-independent pairwise energy E_pw(v0,v1) is:
//    E_pw(0,0) = 0.1,   (total: 0.1 + 0.55 + 0.90 = 1.55)
//    E_pw(1,0) = 0.2,   (total: 0.2 + 0.40 + 0.90 = 1.50)
//    E_pw(0,1) = 0.3,   (total: 0.3 + 0.55 + 0.20 = 1.05)
//    E_pw(1,1) = 0.4.   (total: 0.4 + 0.40 + 0.20 = 1.00)
//
// Conditional energies for p(y_0|y_1=1) are:
//    [0.85, 0.80],
// whereby p(y_0|y_1=1) is [0.4875, 0.5125], such that
//    -log p(y_0=0|y_1=1) = 0.7185.
//
// Similarly, conditional energies for p(y_1|y_0=0) are:
//    [1.0, 0.5],
// whereby p(y_1|y_0=0) is [0.3775, 0.6225], such that
//    -log p(y_1=1|y_0=0) = 0.4741.
//
// The nlpl is the mean of the two, hence 0.5963.
void Run()
{
   MyTrainingDatabase database;
   typedef dtf::database_traits<decltype(database)> data_traits;
   dtf::factor_graph<data_traits> graph;
   std::vector<dtf::weight_t> x, grad;

   {
      dtf::factor<data_traits, 1, LookAtFeature> unary;
      unary[0] = dtf::offset_t(0, 0);

      ts::binary_tree_array<LookAtFeature> tree;
      auto split = tree.push_back(LookAtFeature());
      tree.set_split_leaves(split, 0, 1);
      unary.set_tree(tree);
      graph.move_factor(std::move(unary));

      std::array<dtf::weight_t, 6> weights = { 0.15, 0.5, 0.4, -0.1, 0.75, -0.3 };
      std::for_each(weights.begin(), weights.end(), [&](dtf::weight_t w) { x.push_back(w); });
   }
   {
      dtf::factor<data_traits, 2, LookAtFeature> pairwise;
      pairwise[0] = dtf::offset_t(0, 0);
      pairwise[1] = dtf::offset_t(1, 0);

      ts::binary_tree_array<LookAtFeature> tree;
      pairwise.set_tree(tree);
      graph.move_factor(std::move(pairwise));

      std::array<dtf::weight_t, 4> weights = { 0.1, 0.2, 0.3, 0.4 };
      std::for_each(weights.begin(), weights.end(), [&](dtf::weight_t w) { x.push_back(w); });
   }
   grad.resize(x.size());
   dtf::objective<decltype(database)> objective(graph, database);
   dtf::numeric_t result = objective.Eval(x, grad);
   if (fabs(result - 0.5963) > 0.001)
      throw std::exception("Objective didn't match hand-calculated value.");

   std::array<double, 10> grad_comp = { 0.0675, -0.0675, 0.2562, -0.2562, -0.1888, 0.1888,
                                          -0.1888, 0.0, 0.4450, -0.2562 };
   for (int i = 0; i < sizeof(grad_comp) / sizeof(grad_comp[0]); i++)
   {
      if (fabs(grad_comp[i] - grad[i]) > 0.001)
         throw std::exception("Gradient didn't match hand-calculated value.");
   }
   if (!objective.check_derivative(3.0))
      throw std::exception("check_derivative failed.");

   dtf::dtf_set_all_weights(graph, x.begin(), x.end());
   ts::image<dtf::label_t> labelling(2, 1);
   labelling(0, 0) = 0; labelling(1, 0) = 0;
   if (std::fabs(graph.energy(labelling, database.get_training(0)) - 1.55) > 1e-5)
      throw std::exception("Energy didn't match hand-calculated value.");
   labelling(0, 0) = 1; labelling(1, 0) = 0;
   if (std::fabs(graph.energy(labelling, database.get_training(0)) - 1.5) > 1e-5)
      throw std::exception("Energy didn't match hand-calculated value.");
   labelling(0, 0) = 0; labelling(1, 0) = 1;
   if (std::fabs(graph.energy(labelling, database.get_training(0)) - 1.05) > 1e-5)
      throw std::exception("Energy didn't match hand-calculated value.");
   labelling(0, 0) = 1; labelling(1, 0) = 1;
   if (std::fabs(graph.energy(labelling, database.get_training(0)) - 1.0) > 1e-5)
      throw std::exception("Energy didn't match hand-calculated value.");
}

}