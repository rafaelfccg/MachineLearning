#pragma once

#include "dtf.h"
#include "imageio.h"
#include <sstream>

namespace StandardTest
{

/*
   Author your training database here.
*/
class SplatDatabase 
{
public:
   static const dtf::label_t label_count = 2;
   static const size_t image_count = 3;

   SplatDatabase() : m_train(image_count), m_gt(image_count)
   {
      for (unsigned int i = 0; i < image_count; i++)
      {
         std::wstringstream s;
         s << "splat" << i;
         m_train[i] = ts::load<ts::bgr24>(s.str() + L".png");
         m_gt[i] = ts::load<dtf::label_t>(s.str() + L"_gt.png");
      }
   }
   typedef ts::image<ts::bgr24> training_image;
   typedef ts::image<dtf::label_t> gt_image;
   size_t size() const { return image_count; }
   training_image get_training(size_t index) const { return m_train[index]; }
   gt_image get_ground_truth(size_t index) const { return m_gt[index]; }
   dtf::dense_sampling get_samples(size_t index) const { return dtf::all_pixels(m_gt[index]); }
private:
   mutable std::vector<training_image> m_train;
   mutable std::vector<gt_image> m_gt;
};

/*
   Author your features here.
*/
class PixelDifferenceFeature
{
public:
   static const ts::image<ts::bgr24>& pre_process(const ts::image<ts::bgr24>& image)
   {
      return image;
   }
   template <typename _It>
   bool operator()(int x, int y, const ts::image<ts::bgr24>& image, _It offsets) const
   {
      unsigned int sx1 = clamp(x + x1, 0, image.width());
      unsigned int sy1 = clamp(y + y1, 0, image.height());
      unsigned int sx2 = clamp(x + x2, 0, image.width());
      unsigned int sy2 = clamp(y + y2, 0, image.height());
      unsigned char a = image(sx1, sy1)[c1];
      unsigned char b = image(sx2, sy2)[c2];
      int diff = static_cast<int>(a) - static_cast<int>(b);
      return diff >= lo && diff < hi;
   }
   int c1, x1, y1;
   int c2, x2, y2;
   int lo, hi;
private:
   static unsigned int clamp(int x, unsigned int a, unsigned int b)
   {
      return static_cast<unsigned int>(std::min(std::max(x, static_cast<int>(a)), static_cast<int>(b) - 1));
   }
};

inline int irand(int imin, int inext)
{
   if (imin >= inext)
      throw std::invalid_argument("irand: imin >= inext");
   int range = inext - imin;
   int bits = 0;
   int rangeCopy = range;
   while (rangeCopy > 0)
   {
      rangeCopy >>= 1;
      ++bits;
   }
   if (bits > 15)
      throw std::invalid_argument("irand: bits > 15");
   int mask = (1 << bits) - 1;
   int r;
   do
   {
      r = rand() & mask;
   } while (r >= range);
   return r + imin;
}

inline float frand(float fmin, float fmax)
{
   float frange = fmax - fmin;
   float norm = static_cast<float>(rand()) / RAND_MAX;
   return fmin + norm * frange;
}

inline PixelDifferenceFeature RandomFeature()
{
   PixelDifferenceFeature feature;
   feature.c1 = irand(0, 3);
   feature.c2 = irand(0, 3);
   feature.x1 = irand(-256, 256);
   feature.y1 = irand(-256, 256);
   feature.x2 = irand(-256, 256);
   feature.y2 = irand(-256, 256);
   feature.lo = irand(-255, 256);
   feature.hi = irand(feature.lo, 256);
   return feature;
}

void Recurse(ts::binary_tree_array<PixelDifferenceFeature>& tree, 
            ts::binary_tree_array<PixelDifferenceFeature>::iterator parent,
            size_t& leaf, int level, int levels)
{
   tree.set_split_data(parent, RandomFeature());
   if (level == levels - 2)
   {
      tree.set_split_child(parent, false, tree.leaf_to_child(leaf++));
      tree.set_split_child(parent, true, tree.leaf_to_child(leaf++));
      return;
   }

   auto left = tree.push_back(PixelDifferenceFeature());
   tree.set_split_child(parent, false, left);
   Recurse(tree, left, leaf, level + 1, levels);

   auto right = tree.push_back(PixelDifferenceFeature());
   tree.set_split_child(parent, true, right);
   Recurse(tree, right, leaf, level + 1, levels);
}

template <typename _DataTraits, typename _TFactor>
void AddFactor(dtf::factor_graph<_DataTraits>& graph, _TFactor& factor)
{
   // Generate random features
   ts::binary_tree_array<PixelDifferenceFeature> tree;
   
   // Add the tree nodes with random feature data:
   size_t leaf = 0;
   auto root = tree.push_back(PixelDifferenceFeature());
   Recurse(tree, root, leaf, 0, 8);
   factor.set_tree(ts::tree_order_by_breadth(tree));
   
   // Generate random weights
   // NOTE: Doesn't really matter whether weights are at nodes or leaves
   for (auto w = factor.energies(); w != factor.energies() + factor.energies_size(); ++w)
      *w = frand(-3.0f, 3.0f); 

   graph.push_back(factor);
}

void Run()
{
   typedef PixelDifferenceFeature TFeature;
   SplatDatabase database;
   typedef dtf::database_traits<decltype(database)> data_traits;
   dtf::factor_graph<data_traits> graph;

   // Add unary
   dtf::factor<data_traits, 1, TFeature> unary;
   unary[0] = dtf::offset_t(0, 0);
   AddFactor(graph, unary);

   // Add horizontal pairwise
   dtf::factor<data_traits, 2, TFeature> horz;
   horz[0] = dtf::offset_t(0, 0);
   horz[1] = dtf::offset_t(1, 0);
   AddFactor(graph, horz);

   // Add vertical pairwise
   dtf::factor<data_traits, 2, TFeature> vert;
   vert[0] = dtf::offset_t(0, 0);
   vert[1] = dtf::offset_t(0, 1);
   AddFactor(graph, vert);

   dtf::objective<decltype(database)> function(graph, database);
   if (!function.check_derivative(3.0, 10))
      throw std::exception("check_derivative failed.");
   dtf::learning::OptimizeWeights(graph, database);
}

}