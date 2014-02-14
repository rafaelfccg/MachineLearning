#pragma once

#include "dtf.h"

namespace SimpleTest
{
class MyImage // Dummy image class
{
public:
   unsigned int width() const { return 1; }
   unsigned int height() const { return 1; }
   unsigned char operator()(int x, int y) const { return 0; }
};

class MyTrainingDatabase // Dummy training database
{
public:
   static const dtf::label_t label_count = 2;
   size_t size() const { return 1; }
   MyImage get_training(size_t index) const { return MyImage(); }
   MyImage get_ground_truth(size_t index) const { return MyImage(); }
   dtf::dense_sampling get_samples(size_t index) const { return dtf::all_pixels(MyImage()); }
};

class AlwaysTrueFeature // Dummy feature
{
public:
   static const MyImage& pre_process(const MyImage& image) { return image; }
   template <typename _It> bool operator()(unsigned int x, unsigned int y, const MyImage& data, _It off) const
      { return true; }
};

void Run()
{
   // The train/test data
   MyTrainingDatabase database;
   typedef dtf::database_traits<decltype(database)> data_traits;

   // The factor graph, with a unary factor at each pixel
   dtf::factor_graph<data_traits> graph;

   // Add factor and prior
   dtf::factor<data_traits, 1, AlwaysTrueFeature> factor;
   graph.push_back(factor);

   // A set of weights for evaluation
   std::vector<double> w(2), grad(2);
   w[0] = 0.4; w[1] = -0.1;
   
   // Evaluate pseudo-likelihood and gradients
   dtf::objective<decltype(database)> objective(graph, database);
   double nlpl = objective.Eval(w, grad);

   // Check results
   if (fabs(nlpl - 0.974) > 0.001)
      throw std::exception("Objective didn't match hand-calculated value.");
   if (fabs(grad[0] - 0.6225) > 0.0001 || fabs(grad[1] - -0.6225) > 0.0001)
      throw std::exception("Gradient didn't match hand-calculated value.");
   if (!objective.check_derivative(3.0))
      throw std::exception("check_derivative failed.");
}
}