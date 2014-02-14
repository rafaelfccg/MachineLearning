/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 4 November 2011
   
*/
#pragma once

#include "dtf.h"
#include <iostream>

namespace dtf
{

namespace training
{

inline double log2(double x)
{
   return 1.4426950408889634073599246810019 * std::log(x);
}

template <typename T, unsigned int label_count>
inline T histogram_total(const std::array<T, label_count>& h)
{
   T total = 0;
   for (unsigned int i = 0; i < label_count; i++)
      total += h[i];
   return total;
}

template <typename T, unsigned int label_count>
inline bool histogram_is_delta(const std::array<T, label_count>& h)
{
   bool first = false;
   for (unsigned int i = 0; i < label_count; i++)
   {
      if (h[i] > 0 && first)
         return false;
      else if (h[i] > 0)
         first = true;
   }
   return true;
}

template <typename T, unsigned int label_count>
double histogram_entropy(const std::array<T, label_count>& S)
{
   double entropy = 0.0, eps = 1e-6;
   double scale = 1.0 / (histogram_total(S) + eps * label_count);
   for (unsigned int label = 0; label < label_count; label++)
   {
      double f = scale * (eps + S[label]);
      entropy -= f * log2(f);
   }
   return entropy;
}

template <typename T, unsigned int label_count>
double information_gain(const std::array<T, label_count>& left, const std::array<T, label_count>& right)
{
   unsigned int total_left = histogram_total(left);
   unsigned int total_right = histogram_total(right);
   unsigned int total = total_left + total_right;
   if (total_left == 0 || total_right == 0)
      return 0.0;

   std::array<T, label_count> parent;
   for (unsigned int label = 0; label < label_count; label++)
      parent[label] = left[label] + right[label];

   double inv = 1.0 / total;
   return histogram_entropy(parent) - inv * 
      (histogram_entropy(left) * total_left + histogram_entropy(right) * total_right);
}

template <typename _Tnode, typename _Func>
inline void visit_leaves(const ts::binary_tree_array<_Tnode>& tree, _Func func)
{
   for (ts::breadth_first_iterator<_Tnode> i(tree); i; ++i)
   {
      if (tree.is_leaf(i->it))
         func(*i, tree.child_to_leaf(i->it));
   }
}

class dense_histogram
{
public:
   dense_histogram() : m_dim4(0), m_dim34(0), m_dim234(0) {}
   dense_histogram(size_t leaves, unsigned int energies, size_t features, unsigned int responses) : m_dim4(0), m_dim34(0), m_dim234(0) 
   {
      alloc(leaves, energies, features, responses);
   }
   dense_histogram(dense_histogram&& rhs) : m_dim4(std::move(rhs.m_dim4)), m_dim34(std::move(rhs.m_dim34)), m_dim234(std::move(rhs.m_dim234)), m_arr(std::move(rhs.m_arr)) 
   {
   }
   dense_histogram& operator =(dense_histogram&& rhs)
   {
      m_dim4 = std::move(rhs.m_dim4);
      m_dim34 = std::move(rhs.m_dim34);
      m_dim234 = std::move(rhs.m_dim234);
      m_arr = std::move(rhs.m_arr);
      return *this;
   }

   size_t alloc(size_t leaves, unsigned int energies, size_t features, unsigned int responses)
   {
      while (leaves > 0)
      {
         try
         {
            size_t total = leaves * energies * features * responses;
            m_arr.resize(total);
            m_dim4 = responses;
            m_dim34 = m_dim4 * features;
            m_dim234 = m_dim34 * energies;
            break;
         }
         catch (std::bad_alloc&)
         {
            leaves >>= 1;
         }
      }
      return leaves;
   }

   void increment(size_t leaf, unsigned int energy, size_t feature, unsigned int response)
   {
      size_t index = m_dim234 * leaf + m_dim34 * energy + m_dim4 * feature + response;
      m_arr[index]++;
   }

   template <unsigned int energy_count>
   std::array<unsigned int, energy_count> get_histogram(size_t leaf, size_t feature, unsigned int response) const
   {
      std::array<unsigned int, energy_count> rv;
      size_t start = m_dim234 * leaf + m_dim4 * feature + response;
      for (unsigned int i = 0; i < energy_count; i++)
         rv[i] = m_arr[start + m_dim34 * i];
      return rv;
   }

   size_t leaf_count() const { return m_arr.empty() ? 0 : m_arr.size() / m_dim234; }

protected:
   // Prevent accidental copy of potentially very large data structure
   dense_histogram(const dense_histogram& rhs);
   dense_histogram& operator=(const dense_histogram& rhs);

   size_t m_dim4, m_dim34, m_dim234;
   std::vector<unsigned int> m_arr;
};

template <typename _Gt, unsigned int variable_count>
inline rect<int> bounding_box(const _Gt& gt, const std::array<offset_t, variable_count>& offsets)
{
   rect<int> gtrect(0, 0, gt.width(), gt.height());
   rect<int> box;
    for each (offset_t offset in offsets)
         box |= offset;
   return gtrect.deflate_rect(box);
}

inline ts::thread_locals<range<size_t>> thread_partition(size_t size)
{
   ts::thread_locals<range<size_t>> locals;
   for (unsigned int i = 0; i < locals.size(); i++)
   {
      locals[i].start = size * i / locals.size();
      locals[i].stop = size * (i + 1) / locals.size();
   }
   return locals;
}

template <typename T>
inline size_t leaves_size(const ts::binary_tree_array<T>& tree)
{
   typedef ts::binary_tree_array<T> tree_type;
   size_t leaf_count = 0;
   visit_leaves(tree, [&](const tree_type::node_info& node, size_t leaf_index) 
   { 
      leaf_count = std::max(leaf_count, leaf_index + 1); 
   });
   return leaf_count;
}


template <typename _TDataTraits>
class feature_prep_base
{
public:
   virtual void clear() = 0;
};

template <typename _TDataTraits, typename _TFeature>
class feature_prep_cache : public feature_prep_base<_TDataTraits>
{
public:
   typedef typename _TDataTraits::input_type input_type;
   typedef typename std::remove_reference<decltype(_TFeature::pre_process(input_type()))>::type pre_process_type;

   feature_prep_cache(size_t capacity) : m_capacity(capacity) {}

   template <typename _TDatabase>
   const pre_process_type& get(const _TDatabase& database, size_t index)
   {
      // Return existing if appropriate
      auto i = m_map.find(index);
      if (i == m_map.end())
      {      
         // Create new data item
         auto input = database.get_training(static_cast<unsigned int>(index));
         i = m_map.insert(std::make_pair(index, ts::make_unique<pre_process_type>(_TFeature::pre_process(input)))).first;
         m_queue.push_back(index);
      
         // Erase old data items
         while (m_queue.size() > m_capacity)
         {
            m_map.erase(m_map.find(m_queue.front()));
            m_queue.pop_front();
         }
      }
      else
      {
         // Put reference at back of queue
         auto j = std::find(m_queue.begin(), m_queue.end(), index);
         if (j != m_queue.end())
            m_queue.erase(j);
         m_queue.push_back(index);
      }
      return *((*i).second);
   }

   virtual void clear() override
   {
      m_map.clear();
      m_queue.clear();
   }

protected:
   typedef std::unique_ptr<pre_process_type> pre_process_ptr;

   size_t m_capacity;
   std::map<size_t, pre_process_ptr> m_map;
   std::list<size_t> m_queue;
};

template <typename _TDatabase>
class pre_process_cache
{
protected:
   typedef typename database_traits<_TDatabase>::input_type input_type;
   template <typename _TFeature> struct feature_cache { typedef feature_prep_cache<database_traits<_TDatabase>, _TFeature> type; };

public:
   template <typename _TFeature> struct pre_process { typedef typename feature_cache<_TFeature>::type::pre_process_type type; };

   pre_process_cache(const _TDatabase& database, size_t capacity)
      : m_database(database), m_capacity(capacity) 
   {
      if (capacity < 1)
         throw std::exception("preprocess_cache capacity must be at least 1");
   }

   template <typename _TFeature>
   const typename pre_process<_TFeature>::type& pre_processed(size_t index)
   {
      std::string feature_name(typeid(_TFeature).name());
      auto i = m_preps.find(feature_name);
      if (i == m_preps.end())
      {
         auto preps = ts::make_unique<feature_cache<_TFeature>::type>(m_capacity);
         i = m_preps.insert(std::make_pair(feature_name, std::move(preps))).first;
      }
      
      feature_cache<_TFeature>::type& prep = dynamic_cast<feature_cache<_TFeature>::type&>(*((*i).second));
      return prep.get(m_database, index);
   }

   size_t database_size() const
   {
      return m_database.size();
   }

   void clear()
   {
      for (auto i = m_preps.begin(); i != m_preps.end(); ++i)
         (*i).second->clear();
   }
protected:
   size_t m_capacity;
   const _TDatabase& m_database;
   std::map<std::string, std::unique_ptr<feature_prep_base<database_traits<_TDatabase>>>> m_preps;
};

template <typename _TDatabase, typename feature_t, unsigned int variable_count = 1>
class tree_trainer
{
public:
   static const unsigned int label_count = _TDatabase::label_count;
   static const unsigned int energy_count = power<label_count, variable_count>::raise;
   typedef std::array<unsigned int, energy_count> histogram_t;
   typedef ts::binary_tree_array<feature_t> tree_type;

   tree_trainer(  const _TDatabase& database,
                  const std::array<offset_t, variable_count>& offsets,
                  size_t prep_cache_capacity,
                  std::ostream& console = std::cout) : 
      m_database(database), m_prep_cache(database, prep_cache_capacity), m_offsets(offsets), m_console(console), m_min_samples(2)
   {
   }
   void set_min_split_samples(unsigned int samples) 
   { 
      m_min_samples = std::max(samples, 2u); 
   }

   bool training_round(const std::vector<feature_t>& features,
               tree_type& tree,
               std::vector<histogram_t>& posteriors,
               std::vector<bool>& flags,
               unsigned int level_count)
   {
      // We need to visit each leaf node in the tree and decide whether it is admissible.
      std::vector<size_t> hist_leaf_indices(posteriors.size(), -1);
      std::vector<std::pair<tree_type::iterator, bool>> nodes;
      visit_leaves(tree, [&](const tree_type::node_info& node, size_t leaf_index)
      {
         // Need to check whether this leaf has already been fully investigated
         if (node.depth + 1 < level_count && !flags[leaf_index])
         {
            unsigned int samples = histogram_total(posteriors[leaf_index]);
            if (!histogram_is_delta(posteriors[leaf_index]) && samples >= m_min_samples)
            {
               hist_leaf_indices[leaf_index] = nodes.size();
               nodes.push_back(std::pair<ts::binary_tree_array<feature_t>::iterator, bool>(node.parent, node.direction_from_parent));
            }
         }
      });

      // Allocate the histogram: leaf node * class label * feature index * feature response
      m_console << "Allocating for " << nodes.size() << " leaf nodes" << std::endl;
      dense_histogram hist = build_full_histogram(features, tree, hist_leaf_indices, nodes.size());
      if (hist.leaf_count() == 0)
         return false;

      // Evaluate the quality of each feature at each leaf node
      // On finding a suitable feature, set it into the tree, adding new leaves as children
      double min_score = 0.0;
      for (size_t admissible_leaf_index = 0; admissible_leaf_index < hist.leaf_count(); admissible_leaf_index++)
      {
         std::pair<size_t, double> best = find_best_feature<energy_count>(hist, admissible_leaf_index, features.size());

         // We need to map from the leaf-index (used by the histogram) to the node-index (used by the tree).
         tree_type::iterator parent = nodes[admissible_leaf_index].first;
         bool direction = nodes[admissible_leaf_index].second;
         if (best.second > min_score)
         {
            auto split = tree.push_back(features[best.first]);
            size_t leaf_left = tree.is_leaf(parent) ? 0 : tree.child_to_leaf(tree.get_split_child(parent, direction));
            size_t leaf_right = posteriors.size();
            posteriors[leaf_left] = hist.get_histogram<energy_count>(admissible_leaf_index, best.first, false);
            posteriors.push_back(hist.get_histogram<energy_count>(admissible_leaf_index, best.first, true));
            flags[leaf_left] = false;
            flags.push_back(false);
            if (!tree.is_leaf(parent))
               tree.set_split_child(parent, direction, split);
            tree.set_split_child(split, false, tree.leaf_to_child(leaf_left));
            tree.set_split_child(split, true, tree.leaf_to_child(leaf_right));
         }
         else
         {
            size_t leaf_index = tree.is_leaf(parent) ? 0 : tree.child_to_leaf(tree.get_split_child(parent, direction));
            flags[leaf_index] = true;
         }
      }
      return true;
   }

   template <typename _TFeatureSampler>
   ts::decision_tree<feature_t, energy_count> train(
               _TFeatureSampler& sampler,
               unsigned int feature_count,
               unsigned int level_count)
   {
      // The first step is to build the histogram at any existing leaf nodes
      tree_type tree(m_tree);
      std::vector<histogram_t> posteriors = build_leaf_histograms(tree);
      std::vector<bool> flags(posteriors.size(), false);

      // Now perform training rounds, until stopping criteria are met.
      while (true)
      {
         // Sample the features
         std::vector<feature_t> features = sample_features(sampler, feature_count);
         if (!training_round(features, tree, posteriors, flags, level_count))
            break;
         m_console << "Tree now has " << tree.get_node_count() << " nodes." << std::endl;
      }
      return ts::decision_tree<feature_t, energy_count>(ts::tree_order_by_breadth(tree), std::move(posteriors));
   }

protected:
   template <unsigned int energy_count>
   static std::pair<size_t, double> find_best_feature(const dense_histogram& hist, size_t admissible_leaf_index, size_t feature_count)
   {
      double best_score = 0.0;
      size_t best_feature = 0;
      for (size_t feature_index = 0; feature_index < feature_count; feature_index++)
      {
         auto hist_left = hist.get_histogram<energy_count>(admissible_leaf_index, feature_index, false);
         auto hist_right = hist.get_histogram<energy_count>(admissible_leaf_index, feature_index, true);
         double score = information_gain(hist_left, hist_right);
         if (score > best_score)
         {
            best_score = score;
            best_feature = feature_index;
         }
      }
      return std::pair<size_t, double>(best_feature, best_score);
   }

   dense_histogram build_full_histogram(
      const std::vector<feature_t>& features,
      const tree_type& tree,
      const std::vector<size_t>& hist_leaf_indices,
      size_t leaf_count)
   {
      // Allocate the histogram: leaf node * class label * feature index * feature response
      dense_histogram hist(leaf_count, energy_count, features.size(), 2);
      size_t alloc_leaf_count = hist.leaf_count();
      if (alloc_leaf_count == 0)
         return hist;

      // We want to partition the features into distinct subsets, so that each feature will be used by
      // exactly one thread. This allows us to get coarse-grained parallelism without requiring atomic addition.
      ts::thread_locals<range<size_t>> local_features = thread_partition(features.size());

      // Stripe through the training data, updating the features histogram
      m_console << "Process training images: ";
      for (database_iterator<_TDatabase> data(m_database); data; ++data)
      {
         m_console << ".";
         rect<int> bbox = bounding_box(data.ground_truth(), m_offsets);
         auto prep = m_prep_cache.pre_processed<feature_t>(data.index());

         // Iterate over admissible samples within the training image
         local_features.parallel_for_each([&](const range<size_t>& feature_range)
         {
            for (auto var = data.samples_begin(); var != data.samples_end(); ++var)
            {
               offset_t factor_origin = *var - m_offsets[0];
               if (bbox.contains(factor_origin))
               {
                  size_t tree_leaf_index = tree.get_leaf_index([&](const feature_t& feature) 
                     { return feature(factor_origin.x, factor_origin.y, prep, m_offsets); });
                  size_t hist_leaf_index = hist_leaf_indices[tree_leaf_index];

                  if (hist_leaf_index < alloc_leaf_count)
                  {
                     // Map the ground truth labels of all connected variables to a single index
                     int offset = 1, energy_index = 0;
                     for (unsigned int v = 0; v < variable_count; v++)
                     {
                        offset_t pos = factor_origin + m_offsets[v];
                        assert(data.ground_truth()(pos.x, pos.y) < label_count);
                        energy_index += offset * data.ground_truth()(pos.x, pos.y);
                        offset *= label_count;
                     }		   

                     // We want to be able to do non-interlocked thread-safe increments to the histogram.
                     // This is possible if we parallelize only over features.
                     for (size_t feature_index = feature_range.start; feature_index != feature_range.stop; ++feature_index)
                     {
                        // NOTE: INNER LOOP
                        bool response = features[feature_index](factor_origin.x, factor_origin.y, prep, m_offsets);
                        hist.increment(hist_leaf_index, energy_index, feature_index, response);
                     }
                  }
               }
            }
         });
      }
      m_console << std::endl;
      return hist;
   }

   std::vector<histogram_t> build_leaf_histograms(const tree_type& tree)
   {
      histogram_t empty = { 0 };
      size_t leaf_count = leaves_size(tree);
      std::vector<histogram_t> posteriors(leaf_count, empty);

      ts::thread_locals<std::vector<histogram_t>> histograms(posteriors);
      for (database_iterator<_TDatabase> data(m_database); data; ++data)
      {
         rect<int> bbox = bounding_box(data.ground_truth(), m_offsets);
         auto prep = m_prep_cache.pre_processed<feature_t>(data.index());
         
         ts::parallel_for(data.samples_begin(), data.samples_end(), [&](decltype(data.samples_begin()) var)
         {
            offset_t factor_origin = *var - m_offsets[0];
            if (bbox.contains(factor_origin))
            {
               auto node = tree.root();
               while (!tree.is_leaf(node))
                  node = tree.get_split_child(node, tree.get_split_data(node)(factor_origin.x, factor_origin.y, prep, m_offsets));
               size_t leaf_index = tree.child_to_leaf(node);

               // Map the ground truth labels of all connected variables to a single index
               int offset = 1, energy_index = 0;
               for (unsigned int v = 0; v < variable_count; v++)
               {
                  offset_t pos = factor_origin + m_offsets[v];
                  assert(data.ground_truth()(pos.x, pos.y) < label_count);
                  energy_index += offset * data.ground_truth()(pos.x, pos.y);
                  offset *= label_count;
               }		   
               histograms.local()[leaf_index][energy_index]++;
            }
         });
      }
      // Then reduce the histograms
      histograms.for_each([&](const std::vector<histogram_t>& vh) 
      {
         for (size_t i = 0; i < vh.size(); i++)
            for (label_t j = 0; j < energy_count; j++)
               posteriors[i][j] += vh[i][j];
      });
      return posteriors;
   }

   template <typename _TFeatureSampler>
   std::vector<feature_t> sample_features(_TFeatureSampler& sampler, unsigned int count)
   {
      return sampler(m_prep_cache, count);
   }

   const _TDatabase& m_database;
   std::array<offset_t, variable_count> m_offsets;
   std::ostream& m_console;
   ts::binary_tree_array<feature_t> m_tree;
   unsigned int m_min_samples;
   pre_process_cache<_TDatabase> m_prep_cache;
};

template <typename _TFeatureSampler>
struct sampled_feature
{
private:
   static _TFeatureSampler* _sampler;
   static int _int;
public:
   typedef typename std::remove_reference<decltype(_sampler->operator()(_int, 0u))>::type::value_type type;
};

template <typename _TDatabase, typename _TFeatureSampler>
inline auto LearnDecisionTreeUnary(
   const _TDatabase& database,
   _TFeatureSampler& feature_sampler,
   unsigned int feature_count,
   unsigned int level_count,
   unsigned int prep_cache = 100,
   unsigned int min_samples = 2,
   std::ostream& console = std::cout)
      -> ts::decision_tree<typename sampled_feature<_TFeatureSampler>::type, 
                           _TDatabase::label_count>
{
   typedef sampled_feature<_TFeatureSampler>::type feature_type;
   std::array<dtf::offset_t, 1> offsets = { dtf::offset_t(0, 0) };
   tree_trainer<_TDatabase, feature_type> trainer(database, offsets, prep_cache, console);
   trainer.set_min_split_samples(min_samples);
   return trainer.train(feature_sampler, feature_count, level_count);
}

template <unsigned int _variable_count, typename _TDatabase, typename _TFeatureSampler>
inline auto LearnDecisionTree(
   const std::array<offset_t, _variable_count>& offsets,
   const _TDatabase& database,
   _TFeatureSampler& feature_sampler,
   unsigned int feature_count,
   unsigned int level_count,
   unsigned int prep_cache = 100,
   unsigned int min_samples = 2,
   std::ostream& console = std::cout)
      -> ts::decision_tree<typename sampled_feature<_TFeatureSampler>::type,
                           power<_TDatabase::label_count, _variable_count>::raise>
{
   typedef sampled_feature<_TFeatureSampler>::type feature_type;
   tree_trainer<_TDatabase, feature_type, _variable_count> trainer(database, offsets, prep_cache, console);
   trainer.set_min_split_samples(min_samples);
   return trainer.train(feature_sampler, feature_count, level_count);
}

}
}