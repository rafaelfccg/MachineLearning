/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 4 November 2011
   
*/
#pragma once

#include "dtf.h"
#include "image.h"
// NOTE: For performance it is best to compile with the flags
//    /openmp /O2 / Ob2 /Oi

namespace dtf
{

typedef std::function<double(weight_cit, grad_it, double)> prior_t;

struct prior_none
{
   double operator()(weight_cit weights, grad_it grads, double scale) const { return 0; }
};

class prior_normal
{
public:
   prior_normal(double sigma, size_t dim) : m_sigma(sigma), m_dim(dim) {}

   // -log p(w) = 0.5 sigma^-2 * w'w + dim*log(sigma sqrt(2 pi))
   // \nabla_w -log p(w) = sigma^-2 w
   double operator()(weight_cit weights, grad_it grads, double scale) const
   {
      static const double pi = 3.14159265358979323846264338;
      const double logp_constant = std::log(m_sigma * std::sqrt(2.0 * pi)) * m_dim;
      const double mconst = 1.0 / (m_sigma * m_sigma);
      double nlogp = 0.0;
      for (size_t d = 0; d < m_dim; ++d) 
      {
         nlogp += weights[d] * weights[d];
         grads[d] += scale * mconst * weights[d];
      }
      return scale * (0.5 * mconst * nlogp + logp_constant);
   }
private:
   double m_sigma;
   size_t m_dim;
};

namespace compute
{

template <typename _TFeature, unsigned int variable_count, typename _TPrep>
inline path_t FindPath(const ts::binary_tree_array<_TFeature>& tree,
                        offset_t xy, 
                        const _TPrep& prep,
                        typename std::array<offset_t, variable_count>::const_iterator offsets)
{
   path_t path = 0, mask = 1;
   auto node = tree.root();
   while (!tree.is_leaf(node))
   {
      bool b = tree.get_split_data(node)(static_cast<unsigned int>(xy.x), static_cast<unsigned int>(xy.y), prep, offsets);
      path |= mask & (-(path_t)b);
      node = tree.get_split_child(node, b);
      mask <<= 1;
   }
   return path;
}

template <typename _SplitData>
class path_iterator
{
public:
   typedef ts::binary_tree_array<_SplitData> tree_type;

   path_iterator(const tree_type& tree, path_t path) : m_tree(tree), m_leaf(false), m_node(tree.root()), m_path(path) {}

   path_iterator& operator++()
   {
      m_leaf = m_tree.is_leaf(m_node);
      if (!m_leaf)
      {
         m_node = m_tree.get_split_child(m_node, m_path & 1);
         m_path >>= 1;
      }
      return *this;
   }
   operator bool() const
   {
      return !m_leaf;
   }
   const size_t operator *() const
   {
      return m_tree.get_index(m_node);   
   }
protected:
   const tree_type& m_tree;
   typename tree_type::iterator m_node;
   bool m_leaf;
   path_t m_path;
};

template <typename _SplitData>
path_iterator<_SplitData> begin_path(const ts::binary_tree_array<_SplitData>& tree, path_t path)
{
   return path_iterator<_SplitData>(tree, path);
}

template <typename _SplitData, typename _Prep, typename _Offsets>
class test_iterator
{
public:
   typedef ts::binary_tree_array<_SplitData> tree_type;

   test_iterator(const tree_type& tree, const offset_t& factor_origin, const _Prep& prep, const _Offsets& offsets) 
      : m_tree(tree), m_node(tree.root()), m_leaf(false), m_factor_origin(factor_origin), m_prep(prep), m_offsets(offsets) {}

   test_iterator& operator++()
   {
      m_leaf = m_tree.is_leaf(m_node);
      if (!m_leaf)
         m_node = m_tree.get_split_child(m_node, m_tree.get_split_data(m_node)(m_factor_origin.x, m_factor_origin.y, m_prep, m_offsets));
      return *this;
   }
   operator bool() const
   {
      return !m_leaf;
   }
   const size_t operator *() const
   {
      return m_tree.get_index(m_node);   
   }
protected:
   const tree_type& m_tree;
   typename tree_type::iterator m_node;
   bool m_leaf;
   offset_t m_factor_origin;
   const _Prep& m_prep;
   const _Offsets& m_offsets;
};

template <typename _SplitData, typename _Prep, typename _Offsets>
test_iterator<_SplitData, _Prep, _Offsets> begin_test(const ts::binary_tree_array<_SplitData>& tree, const offset_t& factor_origin,
                                    const _Prep& prep, const _Offsets& offsets)
{
   return test_iterator<_SplitData, _Prep, _Offsets>(tree, factor_origin, prep, offsets);
}


template <label_t label_count, unsigned int variable_count, bool final = true>
struct weights_visitor
{
   template <typename _WIt, typename _VIt, typename _Func>
   inline static void visit(_WIt weights, _VIt vars, unsigned int variable, const std::array<label_t, variable_count>& gtLabels, _Func func)
   {
      _WIt pw = weights;
      int offset = 1, stride;
      for (unsigned int i = 0; i < variable_count; i++)
      {
         if (i == variable)
            stride = offset;
         else
            pw += offset * gtLabels[i];
         offset *= label_count;
      }
      _WIt w = pw;
      for (label_t i = 0; i < label_count - 1; ++i)
      {
         func(*w, vars[i]);
         w += stride;
      }
      func(*w, vars[label_count - 1]);
      if (final)
         func(pw[stride * gtLabels[variable]], vars[label_count]);
   }
};

template <label_t label_count, bool final>
struct weights_visitor<label_count, 1, final>
{
   template <typename _WIt, typename _VIt, typename _Func>
   inline static void visit(_WIt weights, _VIt vars, unsigned int variable, const std::array<label_t, 1>& gtLabels, _Func func)
   {
      for (label_t i = 0; i < label_count; ++i)
         func(weights[i], vars[i]);
      if (final)
         func(weights[gtLabels[0]], vars[label_count]);
   }
};

template <bool final>
struct weights_visitor<2, 1, final>
{
   template <typename _WIt, typename _VIt, typename _Func>
   inline static void visit(_WIt weights, _VIt vars, unsigned int variable, const std::array<label_t, 1>& gtLabels, _Func func)
   {
      func(weights[0], vars[0]);
      func(weights[1], vars[1]);
      if (final)
         func(weights[gtLabels[0]], vars[2]);
   }
};

template <bool final>
struct weights_visitor<2, 2, final>
{
   template <typename _WIt, typename _VIt, typename _Func>
   inline static void visit(_WIt weights, _VIt vars, unsigned int variable, const std::array<label_t, 2>& gtLabels, _Func func)
   {
      unsigned int var2 = 1 - variable;
      _WIt pw = weights + (gtLabels[var2] << var2);
      func(pw[0], vars[0]);
      func(pw[1 << variable], vars[1]);
      if (final)
         func(pw[gtLabels[variable] << variable], vars[2]);
   }
};

// TODO: Check we are not unnecessarily copying data when copying database_iterator objects.

template <typename _TFactor>
ts::image<path_t> ComputeImagePaths(
                        const _TFactor& f,
                        const typename _TFactor::labelling_type& gt,
                        const typename _TFactor::samples_type& samples,
                        const typename _TFactor::pre_process_type& prep)
{
   typedef _TFactor::feature_type feature_type;

   const auto& tree = f.get_tree();
   const std::array<offset_t, f.variable_count>& offsets = f.get_variables();

   // Initialize the image of paths to -1 everywhere
   ts::image<path_t> paths(gt.width(), gt.height());
   std::fill(paths.begin(), paths.end(), -1);

   rect<int> gt_rect(0, 0, static_cast<int>(gt.width()), static_cast<int>(gt.height()));
   rect<int> process_rect = gt_rect.deflate_rect(f.bounding_box());

   for (unsigned int v = 0; v < _TFactor::variable_count; v++)
   {
      ts::parallel_for(samples.begin(), samples.end(), [&](_TFactor::sample_iterator pos)
      {
         // Factor instance location:
         offset_t xy = *pos - offsets[v];
         if (process_rect.contains(xy))
         {
            if (paths(xy.x, xy.y) == -1)
               paths(xy.x, xy.y) = FindPath<feature_type, f.variable_count>(tree, xy, prep, offsets.begin());
         }
      });
   }
   return paths;
}

// Visit, for this factor type and ground truth image, all (weight, variable/label) pairs. 
// For each sampled variable, we visit all connected factor instances of this type,
// determine the path through the tree at that factor instance, and then traverse
// the tree, visiting those elements of the weight matrices corresponding to the 
// ground truth labels at connected variables. Each weight element is paired with
// an element of a per-variable L+1 array. The variable array is used to store 
// accumulated per-label values at variable instances.
template <typename _TFactor, typename _TWIts, typename _TVIt, typename _Func>
inline void Accumulate( const _TFactor& f,
                        const ts::image<path_t>& paths,
                        const typename _TFactor::labelling_type& gt,
                        const typename _TFactor::samples_type& samples,
                        _TWIts weights,
                        _TVIt var_acc,
                        _Func func)
{
   rect<int> gt_rect(0, 0, gt.width(), gt.height());
   rect<int> process_rect = gt_rect.deflate_rect(f.bounding_box());

   // Visit each variable location in the sub-sampled set
   ts::parallel_for(samples.begin(), samples.end(), [&](_TFactor::sample_iterator sample)
   {
      std::array<label_t, f.variable_count> gtlabels;
      _TVIt coeff = var_acc + (sample - samples.begin()) * (f.label_count + 1);
      auto w = *weights;

      // Then for each factor instance connected to this variable...
      for (unsigned int i = 0; i < f.variable_count; i++)
      {
         // Work out the position of this factor instance, and check it is valid
         offset_t factor_instance = *sample - f[i];
         if (process_rect.contains(factor_instance))
         {
            for (unsigned int v = 0; v < f.variable_count; ++v)
               gtlabels[v] = gt(factor_instance.x + f[v].x, factor_instance.y + f[v].y);

            // Recall the path through the tree taken at this factor instance
            path_t path = paths(factor_instance.x, factor_instance.y);

            // Walk the tree and visit each node on this path
            for (auto it = begin_path(f.get_tree(), path); it; ++it)
            {
               // For each node on the path, locate the matrix of weights corresponding to this node
               decltype(w) pw = w + *it * f.get_weights_per_node();
   
               // Visit the label_count+1 weights in the matrix for this particular variable and GT combination.
               weights_visitor<f.label_count, f.variable_count>::visit(pw, coeff, i, gtlabels, func);
            }
         }
      }
   });
}

// Visit, for this factor type and ground truth image, all (weight, variable/label) pairs. 
// For each sampled variable, we visit all connected factor instances of this type,
// determine the path through the tree at that factor instance, and then traverse
// the tree, visiting those elements of the weight matrices corresponding to the 
// ground truth labels at connected variables. Each weight element is paired with
// an element of a per-variable L+1 array. The variable array is used to store 
// accumulated per-label values at variable instances.
template <typename _TFactor, typename _TWIts, typename _TVIt, typename _Func>
inline void Accumulate( const _TFactor& f,
                        const ts::image<size_t>& leaves,
                        const typename _TFactor::labelling_type& gt,
                        const typename _TFactor::samples_type& samples,
                        _TWIts weights,
                        _TVIt var_acc,
                        _Func func)
{
   rect<int> gt_rect(0, 0, gt.width(), gt.height());
   rect<int> process_rect = gt_rect.deflate_rect(f.bounding_box());

   // Visit each variable location in the sub-sampled set
   ts::parallel_for(samples.begin(), samples.end(), [&](_TFactor::sample_iterator sample)
   {
      std::array<label_t, f.variable_count> gtlabels;
      _TVIt coeff = var_acc + (sample - samples.begin()) * (f.label_count + 1);
      auto w = *weights;

      // Then for each factor instance connected to this variable...
      for (unsigned int i = 0; i < f.variable_count; i++)
      {
         // Work out the position of this factor instance, and check it is valid
         offset_t factor_instance = *sample - f[i];
         if (process_rect.contains(factor_instance))
         {
            for (unsigned int v = 0; v < f.variable_count; ++v)
               gtlabels[v] = gt(factor_instance.x + f[v].x, factor_instance.y + f[v].y);

            // Recall the path through the tree taken at this factor instance
            size_t leaf = leaves(factor_instance.x, factor_instance.y);

            // Locate the matrix of weights corresponding to this leaf
            decltype(w) pw = w + leaf * f.get_weights_per_node();
   
            // Visit the label_count+1 weights in the matrix for this particular variable and GT combination.
            weights_visitor<f.label_count, f.variable_count>::visit(pw, coeff, i, gtlabels, func);
         }
      }
   });
}

// At every pixel, compute
//    nlpl = varAcc(y*_i) + log sum_i exp -varAcc(i)
// Replace varrAcc(y_i) with
//       -exp(-varAcc(y_i)) / sigma_i
// Return
//    mean(nlpl)
template <label_t label_count>
inline numeric_t LogSumExpReplace(std::vector<numeric_t>& nlpl, std::vector<weight_t>& var_acc)
{
   // Use a parallel, numerically stable algorithm for mean computation
   ts::thread_locals<numeric_t> means(0);
   ts::thread_locals<size_t> counts(0);
   // At each sample:
   ts::parallel_for<size_t>(0, nlpl.size(), [&](size_t i)
   {
      auto src = var_acc.begin() + i * (label_count + 1);
      std::array<weight_t, label_count> probs;
      numeric_t logsumexp = robust_exp_norm(src, src + label_count, probs.begin());
      nlpl[i] = logsumexp + src[label_count];
      (*counts)++;
      *means += (nlpl[i] - *means) / *counts;
      for (label_t j = 0; j < label_count; j++)
         src[j] = -probs[j];
      src[label_count] = 1;
   });
   for (unsigned int i = 1; i < means.size(); i++)
   {
      counts[0] += counts[i];
      if (counts[0] > 0)
         means[0] += (means[i] - means[0]) * (counts[i] / static_cast<numeric_t>(counts[0]));
   }
   assert(counts[0] == nlpl.size());
   return means[0];
}

inline numeric_t AdjustMean(numeric_t mean_a, unsigned __int64 count_a, numeric_t mean_b, unsigned __int64 count_b)
{
   // sum_a = mean_a * count_a
   // sum_b = mean_b * count_b
   // return (sum_a + sum_b) / (count_a + count_b)
   //         = (mean_a * count_a + mean_b * count_b) / (count_a + count_b)
   numeric_t a = mean_a * (static_cast<numeric_t>(count_a) / static_cast<numeric_t>(count_a + count_b));
   numeric_t b = mean_b * (static_cast<numeric_t>(count_b) / static_cast<numeric_t>(count_a + count_b));
   return a + b;
}

inline unsigned __int64 Flush(grad_it gmean,                                // Global mean
                              unsigned __int64 gcount,                      // (Previous) global count
                              ts::thread_locals<std::vector<grad_acc_t>>& lacc, // Thread-local accumulators
                              unsigned int lcount)                          // Thread-locals count
{
   if (lcount == 0)
      return gcount;
   // Start by accumulating all local values
   for (unsigned int i = 1; i < lacc.size(); i++)
   {
      for (size_t j = 0; j < lacc[0].size(); j++)
         lacc[0][j] += lacc[i][j];
      std::fill(lacc[i].begin(), lacc[i].end(), 0);
   }
   // Then scale into the destination
   numeric_t gScale = gcount / static_cast<numeric_t>(gcount + lcount);
   numeric_t lScale = 1 / static_cast<numeric_t>(gcount + lcount);
   for (size_t i = 0; i < lacc[0].size(); i++)
      gmean[i] = static_cast<grad_t>((gScale * gmean[i] + lScale * lacc[0][i]));
   std::fill(lacc[0].begin(), lacc[0].end(), 0);
   return gcount + lcount;
}

// Here we are on a particular factor type, and we must accumulate at variables over connected factor instances.
// This is done by visiting variables, and pulling in the contributions from connected factor instances.
template <typename _TFactor, typename _TWIt>
ts::image<path_t> AccumulateVariables(     
                              const _TFactor& f,
                              const typename _TFactor::input_type& input,
                              const typename _TFactor::labelling_type& gt,
                              const typename _TFactor::samples_type& samples,
                              std::vector<weight_t>& var_acc,
                              _TWIt weights)
{
   // Allow the feature class to pre-process the training data
   auto prep = _TFactor::feature_type::pre_process(input);

   // Fill in an image of paths through the tree at locations needed for the sampled variables
   ts::image<path_t> paths = ComputeImagePaths(f, gt, samples, prep);

   Accumulate(f, paths, gt, samples, &weights, var_acc.begin(), [](weight_t weight, weight_t& varacc)
   { 
      varacc += weight; 
   });
   return paths;
}

// The gradient is defined as a sum over variables.
// At each variable the sum is defined over its connected factors of the relevant type.
// Therefore we can either visit the variables, pulling in the contributions from the connected factor instances,
// or we can visit the factor instances, pushing out the contributions into the connected variables.
// Thus far, we have chosen the latter option, visiting the factor instances and pushing contributions out to connected variables.
// However, it's not clear that this saves anything over visiting the variables and pulling in the factor contributions.
// And in the case that variables are subsampled, it makes sense to switch to the other paradigm.
// So we will plan to visit subsampled variables and accumulate the contributions from connected factor instances.
template <typename _TFactor>
void AccumulateGradients(  const _TFactor& f,
                           const ts::image<path_t>& paths,
                           const typename _TFactor::labelling_type& gt, 
                           const typename _TFactor::samples_type& samples,
                           const std::vector<weight_t>& var_acc,
                           ts::thread_locals<grad_acc_it>& grads)
{
   Accumulate(f, paths, gt, samples, grads, var_acc.begin(), [](grad_acc_t& grad, weight_t varacc)
   { 
      grad += varacc;
   });
}

// Accumulate the energies for factor instances of this type at variables of the current colour
template <typename _TFactor>
void AccumulateEnergies(   const _TFactor& f,
                           const std::vector<offset_t>& samples,
                           const ts::image<size_t>& leaves,
                           const ts::image<label_t>& labelling,
                           ts::image<std::array<weight_acc_t, _TFactor::label_count>>& energies,
                           const ts::image<bool>& flags)
{
   rect<int> gt_rect(0, 0, labelling.width(), labelling.height());
   rect<int> process_rect = gt_rect.deflate_rect(f.bounding_box());

   // Parallel-For each variable of the current colour
   ts::parallel_for(samples.cbegin(), samples.cend(), [&](const std::vector<offset_t>::const_iterator& sample)
   {
      std::array<label_t, f.variable_count> gtlabels;
      if (flags((*sample).x, (*sample).y))
      {
         std::array<weight_acc_t, f.label_count>& e = energies((*sample).x, (*sample).y);

         // For each factor instance connected to the current variable
         for (unsigned int i = 0; i < f.variable_count; i++)
         {
            // Work out the position of this factor instance, and check it is valid
            offset_t factor_instance = *sample - f[i];
            if (process_rect.contains(factor_instance))
            {
               // Determine the labels of variables connected to this factor instance
               for (unsigned int v = 0; v < f.variable_count; ++v)
                  gtlabels[v] = labelling(factor_instance.x + f[v].x, factor_instance.y + f[v].y);
               
               // Recall the leaf node index for this factor instance
               size_t leaf_index = leaves(factor_instance.x, factor_instance.y);

               // Look up the energy table for this factor instance
               weight_cit w = f.energies(leaf_index);

               // Visit the label_count+1 weights in the matrix for this particular variable and GT combination.
               weights_visitor<f.label_count, f.variable_count, false>::visit(w, e.begin(), i, gtlabels, [&](weight_t src, weight_acc_t& dst)
               {
                  dst += src;
               });
            }
         }
      }
   });
}

// Accumulate the energies for factor instances of this type at variables of the current colour
template <typename _TFactor>
void AccumulateEnergies(   const _TFactor& f,
                           const std::vector<offset_t>& samples,
                           const ts::image<size_t>& leaves,
                           const ts::image<label_t>& labelling,
                           ts::image<std::array<weight_acc_t, _TFactor::label_count>>& energies)
{
   rect<int> gt_rect(0, 0, labelling.width(), labelling.height());
   rect<int> process_rect = gt_rect.deflate_rect(f.bounding_box());

   // Parallel-For each variable of the current colour
   ts::parallel_for(samples.cbegin(), samples.cend(), [&](const std::vector<offset_t>::const_iterator& sample)
   {
      std::array<label_t, f.variable_count> gtlabels;
      std::array<weight_acc_t, f.label_count>& e = energies((*sample).x, (*sample).y);

      // For each factor instance connected to the current variable
      for (unsigned int i = 0; i < f.variable_count; i++)
      {
         // Work out the position of this factor instance, and check it is valid
         offset_t factor_instance = *sample - f[i];
         if (process_rect.contains(factor_instance))
         {
            // Determine the labels of variables connected to this factor instance
            for (unsigned int v = 0; v < f.variable_count; ++v)
               gtlabels[v] = labelling(factor_instance.x + f[v].x, factor_instance.y + f[v].y);
               
            // Recall the leaf node index for this factor instance
            size_t leaf_index = leaves(factor_instance.x, factor_instance.y);

            // Look up the energy table for this factor instance
            weight_cit w = f.energies(leaf_index);

            // Visit the label_count+1 weights in the matrix for this particular variable and GT combination.
            weights_visitor<f.label_count, f.variable_count, false>::visit(w, e.begin(), i, gtlabels, [&](weight_t src, weight_acc_t& dst)
            {
               dst += src;
            });
         }
      }
   });
}

// A convenient way to compute a weighted sum of energies for mean field updates
template <label_t label_count, unsigned int variable_count, typename It, typename Func>
inline void MeanFieldVisit(weight_cit weights, unsigned int variable,
	   const std::array<It, variable_count>& q_refs,
      Func func)
{
   std::array<label_t, variable_count> labels = { 0 };
   for (size_t e_i = 0; labels[variable_count - 1] < label_count; ++e_i)
   {
      numeric_t qprob = 1;
      for (unsigned int i = 0; i < variable_count; ++i)
      {
         if (i == variable)
            continue;
         qprob *= (*q_refs[i])[labels[i]];
      }
      func(labels[variable], qprob, weights[e_i]);
      
      ++labels[0];
      for (unsigned int i = 0; i < variable_count - 1; ++i)
      {
         if (labels[i] == label_count)
         {
            labels[i] = 0;
            ++labels[i+1];
         }
      }
   }
}

// A convenient way to compute a weighted sum of energies for mean field updates
template <label_t label_count, typename It, typename Func>
inline void MeanFieldVisit(weight_cit weights, unsigned int variable,
	   const std::array<It, 1>& q_refs,
      Func func)
{
   for (size_t e_i = 0; e_i < label_count; ++e_i)
      func(e_i, variable > 0 ? (*q_refs[0])[e_i] : 1, weights[e_i]);
}

// Accumulate the energies for factor instances of this type at variables of the current colour,
// given mean field distributions (q_field) at the neighboring variables.
template <typename _TFactor>
void AccumulateMeanFieldEnergies(   const _TFactor& f,
                           const std::vector<offset_t>& samples,
                           const ts::image<size_t>& leaves,
                           ts::image<std::array<weight_acc_t, _TFactor::label_count>>& q_field)
{
   rect<int> gt_rect(0, 0, q_field.width(), q_field.height());
   rect<int> process_rect = gt_rect.deflate_rect(f.bounding_box());

   // Parallel-For each variable of the current colour
   ts::parallel_for(samples.cbegin(), samples.cend(), [&](const std::vector<offset_t>::const_iterator& sample)
   {
      std::array<offset_t, f.variable_count> q_offsets;
      std::array<weight_acc_t, f.label_count>& dst_q = q_field((*sample).x, (*sample).y);
      std::array<decltype(q_field[0]), f.variable_count> q_refs;

      // For each factor instance connected to the current variable
      for (unsigned int i = 0; i < f.variable_count; i++)
      {
         // Work out the position of this factor instance, and check it is valid
         offset_t factor_origin = *sample - f[i];
         if (process_rect.contains(factor_origin))
         {
            // Recall the leaf node index for this factor instance
            size_t leaf_index = leaves(factor_origin.x, factor_origin.y);

            // Look up the energy table for this factor instance
            weight_cit w = f.energies(leaf_index);

            // Accumulate the mean field energy
            for (unsigned int j = 0; j < f.variable_count; ++j)
            {
               offset_t pos = factor_origin + f[j];
               q_refs[j] = q_field[pos.y] + pos.x;
            }
            MeanFieldVisit<f.label_count, f.variable_count>(w, i, q_refs, [&](label_t var_state, numeric_t qprob, weight_t energy)
            {
               dst_q[var_state] += qprob * energy;
            });
         }
      }
   });
}

// Sum the energies for factor instances of this type at variables,
// given mean field distributions (q_field) at all variables.
template <typename _TFactor>
numeric_t SumMeanFieldEnergies(const _TFactor& f,
                               const ts::image<size_t>& leaves,
                               const ts::image<std::array<weight_acc_t, _TFactor::label_count>>& q_field)
{
   rect<int> gt_rect(0, 0, q_field.width(), q_field.height());
   rect<int> process_rect = gt_rect.deflate_rect(f.bounding_box());
   ts::thread_locals<weight_acc_t> energy_sum(0);

   // Parallel-For each variable of the current colour
   dense_sampling samples = all_pixels(q_field);
   ts::parallel_for(samples.begin(), samples.end(), [&](dense_pixel_iterator sample)
   {
      std::array<offset_t, f.variable_count> q_offsets;
      std::array<decltype(q_field[0]), f.variable_count> q_refs;

      // Work out the position of this factor instance, and check it is valid
      offset_t factor_origin = *sample - f[0];
      if (process_rect.contains(factor_origin))
      {
         numeric_t cur_esum = 0;
               
         // Look up the energy table for this factor instance
         weight_cit w = f.energies(leaves(factor_origin.x, factor_origin.y));

         // Accumulate the mean field energy
         for (unsigned int j = 0; j < f.variable_count; ++j)
         {
            offset_t pos = factor_origin + f[j];
            q_refs[j] = q_field[pos.y] + pos.x;
         }
         MeanFieldVisit<f.label_count, f.variable_count>(w, f.variable_count, q_refs, [&](label_t var_state, numeric_t qprob, weight_t energy)
         {
            cur_esum += qprob * energy;
         });
         *energy_sum += cur_esum;
      }
   });
   return energy_sum.reduce(numeric_t(0), [](numeric_t& sum, weight_acc_t next) { sum += next; });
}

// Compute the negative log-pseudolikelihood for a given factor graph structure with associated trees.
template <typename _DataTraits, typename _TDatabase>
std::pair<numeric_t, unsigned __int64> ComputeObjective(
                     const factor_graph<_DataTraits>& structure,
                     const _TDatabase& database,
                     weight_cit weights)
{
   numeric_t mean = 0;
   unsigned __int64 count = 0;
   const label_t label_count = _TDatabase::label_count;

   // For each training image independently
   for (database_iterator<_TDatabase> i(database); i; ++i)
   {
      unsigned int sampled_variable_count = i.samples_end() - i.samples_begin();
      std::vector<weight_t> var_acc(sampled_variable_count * (label_count + 1), 0);
      std::vector<numeric_t> nlpl(sampled_variable_count);

      // Store the paths through the tree for each factor
      size_t pos = 0;
      visit_dtf_factors(structure, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         // Accumulate variables and store paths via runtime type-dispatch
         factor.accumulate_variables(i.training(), i.ground_truth(), i.samples(), var_acc, weights + pos);
         pos += factor.get_weight_count();
      });

      // We can now calculate l_i(w) via log-sum-exp
      // We can also calculate sigma_i at the same time, and overwrite c_i with -exp(-a_i)/sigma_i.
      numeric_t nlpl_mean = LogSumExpReplace<label_count>(nlpl, var_acc);
      mean = AdjustMean(mean, count, nlpl_mean, sampled_variable_count);
      count += sampled_variable_count;
   }
   return std::pair<numeric_t, unsigned __int64>(mean, count);
}

template <typename _TDatabase>
unsigned int CountTotalPixels(const _TDatabase& database)
{
   unsigned int total_count = 0;

   for (database_iterator<_TDatabase> i(database); i; ++i)
   {
      unsigned int sampled_variable_count = i.samples_end() - i.samples_begin();
      total_count += sampled_variable_count;
   }
   return total_count;
}

// Compute the negative log-pseudolikelihood for a given factor graph structure with associated trees.
template <typename _DataTraits, typename _TDatabase>
std::pair<numeric_t, unsigned __int64> ComputeObjectiveWithGradients(
                     const factor_graph<_DataTraits>& structure,
                     const _TDatabase& database,
                     weight_cit weights,
                     grad_it grads)
{
   numeric_t mean = 0;
   unsigned __int64 count = 0;
   const label_t label_count = _TDatabase::label_count;
   ts::thread_locals<std::vector<grad_acc_t>> grad_accs;
   ts::thread_locals<grad_acc_it> grad_its;

   // Zero the gradients
   size_t weight_count = dtf_count_weights(structure);
   std::fill(grads, grads + weight_count, 0);
   grad_accs.for_each([&](std::vector<grad_acc_t>& g)
   {
      g.resize(weight_count);
      std::fill(g.begin(), g.end(), 0);
   });

   // For each training image independently
   for (database_iterator<_TDatabase> i(database); i; ++i)
   {
      unsigned int sampled_variable_count = i.samples_end() - i.samples_begin();
      std::vector<weight_t> var_acc(sampled_variable_count * (label_count + 1), 0);
      std::vector<numeric_t> nlpl(sampled_variable_count);

      // Store the paths through the tree for each factor
      size_t pos = 0;
      std::vector<ts::image<path_t>> paths;
      visit_dtf_factors(structure, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         // Accumulate variables and store paths via runtime type-dispatch
         ts::image<path_t> img_paths = factor.accumulate_variables(i.training(), i.ground_truth(), i.samples(), var_acc, weights + pos);
         paths.push_back(std::move(img_paths));
         pos += factor.get_weight_count();
      });

      // We can now calculate l_i(w) via log-sum-exp
      // We can also calculate sigma_i at the same time, and overwrite c_i with - -a_i)/sigma_i.
      numeric_t nlpl_mean = LogSumExpReplace<label_count>(nlpl, var_acc);
      mean = AdjustMean(mean, count, nlpl_mean, sampled_variable_count);

      // Thereafter we can calculate gradients, independently for each factor/tree:
      pos = 0;
      auto path = paths.begin();
      visit_dtf_factors(structure, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         // Need to accumulate into thread-local gradient buffer
         for (unsigned int thread = 0; thread < grad_its.size(); thread++)
            grad_its[thread] = grad_accs[thread].begin() + pos;

         factor.accumulate_gradients(*path++, i.ground_truth(), i.samples(), var_acc, grad_its);
         pos += factor.get_weight_count();
      });
      // Combine the thread-local gradient accumulators into a global buffer
      count = Flush(grads, count, grad_accs, sampled_variable_count);
   }
   return std::pair<numeric_t, unsigned __int64>(mean, count);
}

// Compute the negative log-pseudolikelihood for a given factor graph structure with associated trees.
template <typename _DataTraits, typename _TDatabase>
numeric_t ComputeObjective(
                     const factor_graph<_DataTraits>& structure,
                     const _TDatabase& database,
                     weight_cit weights,
                     const std::vector<std::function<numeric_t(weight_cit, grad_it, double)>>& priors)
{
   if (!priors.empty() && priors.size() != dtf_count_factors(structure))
      throw std::exception("Number of priors does not match number of DTF factors in the graph.");

   std::pair<numeric_t, unsigned __int64> result = ComputeObjective(structure, database, weights);
   numeric_t mean = result.first;

   if (!priors.empty())
   {
      // Take prior into account:
      // Add (-1/N) log p(w) to the objective and \nabla_w (-1/N) log p(w) to the gradient.
      size_t pos = 0, prior_index = 0;
      double scale = result.second == 0 ? 1.0 : 1.0 / result.second;
      visit_dtf_factors(structure, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         mean += priors[prior_index++](weights + pos, scale);
         pos += factor.get_weight_count();
      });
   }
   return mean;
}

// Compute the negative log-pseudolikelihood for a given factor graph structure with associated trees.
template <typename _DataTraits, typename _TDatabase>
numeric_t ComputeObjectiveWithGradients(
                     const factor_graph<_DataTraits>& structure,
                     const _TDatabase& database,
                     weight_cit weights,
                     const std::vector<std::function<numeric_t(weight_cit, grad_it, double)>>& priors,
                     /* out */ grad_it grads)
{
   if (!priors.empty() && priors.size() != dtf_count_factors(structure))
      throw std::exception("Number of priors does not match number of DTF factors in the graph.");

   std::pair<numeric_t, unsigned __int64> result = ComputeObjectiveWithGradients(structure, database, weights, grads);
   numeric_t mean = result.first;

   if (!priors.empty())
   {
      // Take prior into account:
      // Add (-1/N) log p(w) to the objective and \nabla_w (-1/N) log p(w) to the gradient.
      size_t pos = 0, prior_index = 0;
      double scale = result.second == 0 ? 1.0 : 1.0 / result.second;
      visit_dtf_factors(structure, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         mean += priors[prior_index++](weights + pos, grads + pos, scale);
         pos += factor.get_weight_count();
      });
   }
   return mean;
}

template <typename _Factor>
numeric_t Energy(
   const _Factor& factor,
   const typename _Factor::labelling_type& labelling,
   const typename _Factor::input_type& input)
{
   auto prep = _Factor::feature_type::pre_process(input);
   rect<int> bbox = factor.bbox_variables(labelling.width(), labelling.height());

   ts::thread_locals<numeric_t> energy(0);
   dense_sampling samples = all_pixels(labelling);
   ts::parallel_for(samples.begin(), samples.end(), [&](dense_pixel_iterator var)
   {
      offset_t factor_origin = *var - factor[0];
      if (bbox.contains(factor_origin))
      {
         // Locate the offset within the weights table for this combination of variable values
         auto w = factor.energies(factor.leaf_index(factor_origin, prep));
         unsigned int stride = 1; 
         for (unsigned int i = 0; i < factor.variable_count; i++)
         {
            w += stride * labelling(factor_origin.x + factor[i].x, factor_origin.y + factor[i].y);
            stride *= factor.label_count;
         }
         *energy += *w;
      }
   });
   return energy.reduce(0, [](numeric_t& a, numeric_t b) { a += b; });
}

template <typename _Factor>
void AccumulateUnaries(const _Factor& factor,
                        const typename _Factor::input_type& input,
                        ts::image<std::array<numeric_t, _Factor::label_count>>& acc)
{
   auto prep = _Factor::feature_type::pre_process(input);
   std::array<offset_t, 1> offsets = { offset_t(0, 0) };

   ts::parallel_for_each_pixel_xy(input, [&](unsigned int x, unsigned int y)
   {
      std::array<numeric_t, factor.label_count>& dst = acc(x, y);
      auto src = factor.energies(factor.leaf_index(pos_t(x, y), prep));
      for (label_t label = 0; label < factor.label_count; label++)
         dst[label] += src[label];
   });
}

template <typename _TFactor, typename _Func>
void VisitInstances(const _TFactor& factor, const typename _TFactor::input_type& input, _Func func)
{
   const unsigned int cx = input.width(), cy = input.height();
   unsigned int energy_count = factor.get_weights_per_node();
   std::vector<pos_t> variables(_TFactor::variable_count);
   rect<int> r = factor.bbox_variables(cx, cy);
   auto tree = factor.get_tree();
   auto prep = _TFactor::feature_type::pre_process(input);

   offset_t factor_origin;
   for (factor_origin.y = r.top; factor_origin.y < r.bottom; ++factor_origin.y)
   {
      for (factor_origin.x = r.left; factor_origin.x < r.right; ++factor_origin.x)
      {
         // Set up the variable locations for this factor instance
         for (unsigned int i = 0; i < _TFactor::variable_count; ++i)
            variables[i] = factor_origin + factor[i];
         // Callback on factor instance
         auto table = factor.energies(factor.leaf_index(factor_origin, prep));
         func(std::begin(variables), std::end(variables), table, table + factor.get_weights_per_node());
      }
   }
}

// Accumulate the weights down the tree and store in the factor as leaf node values
template <typename _TFactor>
void SumWeights(_TFactor& factor, weight_cit wbegin, weight_cit wend)
{
   const _TFactor::tree_type& tree = factor.get_tree();
   unsigned int energy_count = factor.get_weights_per_node();
   unsigned int levels = 1;
   std::vector<weight_t> energies(levels * energy_count, 0);
   struct info
   {
      _TFactor::tree_type::iterator node;
      unsigned int depth;
   };
   std::stack<info> stack;
   info root = { tree.root(), 0 };
   stack.push(root);
   while (!stack.empty())
   {
      info i = stack.top();
      stack.pop();
      
      if (i.depth >= levels)
      {
         levels = i.depth + 1;
         energies.resize(levels * energy_count);
      }
      // Use a part of the energies array that is not already in use
      auto current = energies.begin() + i.depth * energy_count;
      auto parent = current == energies.begin() ? current : current - energy_count;

      std::transform(parent, parent + energy_count,                  // Using the parent energy table,
                     wbegin + tree.get_index(i.node) * energy_count, // add on the weights for the current node,
                     current, std::plus<weight_t>());                // and write to the current table
                     
      if (tree.is_leaf(i.node))
      {
         // Write the accumulated weights back to the factor object
         size_t leaf_index = tree.child_to_leaf(i.node);
         std::copy(current, current + energy_count, factor.energies(leaf_index));
      }
      else
      {
         info left = { tree.get_split_child(i.node, false), i.depth + 1 };
         info right = { tree.get_split_child(i.node, true), i.depth + 1 };
         stack.push(right);
         stack.push(left);
      }
   }
}

template <typename _TFactor>
ts::image<size_t> LeafImage(const _TFactor& f, const typename _TFactor::input_type& input)
{
   rect<int> input_rect(0, 0, static_cast<int>(input.width()), static_cast<int>(input.height()));
   rect<int> process_rect = input_rect.deflate_rect(f.bounding_box());

   auto prep = _TFactor::feature_type::pre_process(input);
   ts::image<size_t> leaf(input.width(), input.height());
   offset_t origin = f[0];

   // For each pixel that is a factor instance origin:
   dense_sampling samples = all_pixels(input);
   ts::parallel_for(samples.begin(), samples.end(), [&](dense_pixel_iterator pos)
   {
      // Factor instance location:
      offset_t xy = *pos - origin;
      if (process_rect.contains(xy))
      {
         // Evaluate the tree at this pixel
         leaf(xy.x, xy.y) = f.leaf_index(xy, prep);
      }
   });
   return leaf;
}

}
}
