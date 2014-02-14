/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 4 November 2011
   
*/
#pragma once

#include "dtf.h"
#include "dtf_inference.h"

namespace dtf
{
namespace classify
{

template <typename _TIt>
inline _TIt arg_max(_TIt start, _TIt stop)
{
   _TIt rv = start;
   auto maxval = std::numeric_limits<decltype(*start)>::min();
   for (_TIt i = start; i != stop; ++i)
   {
      if (*i >= maxval)
      {
         maxval = *i;
         rv = i;
      }
   }
   return rv;
}

template <typename _TIt>
inline _TIt arg_min(_TIt start, _TIt stop)
{
   _TIt rv = start;
   auto minval = std::numeric_limits<decltype(*start)>::max();
   for (_TIt i = start; i != stop; ++i)
   {
      if (*i <= minval)
      {
         minval = *i;
         rv = i;
      }
   }
   return rv;
}

template <typename T, size_t count>
inline size_t arg_max(const std::array<T, count>& h)
{
   return arg_max(h.begin(), h.end()) - h.begin();
}

template <typename T, size_t count>
inline size_t arg_min(const std::array<T, count>& h)
{
   return arg_min(h.begin(), h.end()) - h.begin();
}

template <typename T, label_t _Labels>
inline ts::image<label_t> arg_min(const ts::image<std::array<T, _Labels>>& energies)
{
   return ts::parallel_pointwise(energies, [&](const std::array<T, _Labels>& t)
   {
      return static_cast<label_t>(arg_min(t));
   });
}

// Classification by taking the arg-max of the posterior distribution at leaf nodes of a classification tree.
// Note that this method has not been optimized for the best possible performance
template <label_t label_count, typename feature_t, typename count_t = unsigned int>
class decision_tree_classifier
{
public:
   typedef std::array<count_t, label_count> histogram_t;
   typedef ts::decision_tree<feature_t, label_count, count_t> tree_t;

   decision_tree_classifier(const tree_t& tree) : m_tree(tree) {}

   template <typename _TImg>
   ts::image<label_t> operator()(const _TImg& input) const
   {
      ts::image<label_t> labelling(input.width(), input.height());
      auto prep = feature_t::pre_process(input);
      std::array<offset_t, 1> offsets = { offset_t(0, 0) };
      ts::parallel_for_each_pixel_xy(input, [&](unsigned int x, unsigned int y)
      {
         const histogram_t& leaf_distribution = m_tree.get_leaf_distribution([&](const feature_t& feature)
         {
            return feature(x, y, prep, offsets);
         });
         labelling(x, y) = static_cast<label_t>(arg_max(leaf_distribution));
      });
      return labelling;
   }

protected:
   const tree_t& m_tree;
};

// Classification by averaging the posterior distributions over multiple trees in a forest.
// Note that this method has not been optimized for the best possible performance
template <label_t label_count, typename TFeature, typename count_t = unsigned int>
class decision_forest_classifier
{
public:
   typedef TFeature feature_t;
   typedef std::array<count_t, label_count> histogram_t;
   typedef ts::decision_tree<feature_t, label_count, count_t> tree_t;

   decision_forest_classifier(const std::vector<tree_t>& forest) : m_forest(forest) {}

   template <typename _TImg>
   ts::image<label_t> operator()(const _TImg& input) const
   {
      ts::image<label_t> labelling(input.width(), input.height());
      auto prep = feature_t::pre_process(input);
      std::array<offset_t, 1> offsets = { offset_t(0, 0) };
      ts::parallel_for_each_pixel_xy(input, [&](unsigned int x, unsigned int y)
      {
         histogram_t distribution = { 0 };
         std::for_each(m_forest.begin(), m_forest.end(), [&](const tree_t& tree)
         {
            typedef std::remove_reference<decltype(tree)>::type tree_t; // Needed to compile under VC2010
            distribution += tree.get_leaf_distribution([&](const tree_t::feature_t& feature)
            {
               return feature(x, y, prep, offsets);
            });
         });
         labelling(x, y) = static_cast<label_t>(arg_max(distribution));
      });
      return labelling;
   }

protected:
   const std::vector<tree_t>& m_forest;
};

// Just the MAP prediction over all the unary factors in the graph
template <typename _DataTraits>
class unaries_MAP_classifier
{
public:
   unaries_MAP_classifier(const factor_graph<_DataTraits>& graph) : m_graph(graph) {}

   template <typename _TImg>
   ts::image<label_t> operator()(const _TImg& input) const
   {
      // Initialize the energy table for each variable
      ts::image<std::array<numeric_t, _DataTraits::label_count>> energies(input.width(), input.height());

      // Accumulate the energy table for the unary factors
      visit_dtf_factors(m_graph, [&](const dtf_factor_base<_DataTraits>& factor) 
      { 
         factor.accumulate_unary_energies(input, energies); 
      });

      // Take the arg-min of the energies at each variable
      return arg_min(energies);
   }
protected:
   const factor_graph<_DataTraits>& m_graph;
};

// Just the MAP prediction over all the unary factors in the graph
template <typename _DataTraits>
class pairwise_map_classifier
{
public:
   pairwise_map_classifier(const factor_graph<_DataTraits>& graph, unsigned int trw_max_iter = 30, double trw_min_conv = 1e-3) 
      : m_graph(graph), m_max_iter(trw_max_iter), m_min_conv(trw_min_conv) 
   {}

   template <typename _TImg>
   ts::image<label_t> operator()(const _TImg& input) const
   {
      return dtf::inference::map_trw(input, m_graph, m_max_iter, m_min_conv);
   }
protected:
   unsigned int m_max_iter;
   double m_min_conv;
   const factor_graph<_DataTraits>& m_graph;
};

template <typename _DataTraits>
class simulated_annealing_classifier
{
public:
   simulated_annealing_classifier(const factor_graph<_DataTraits>& graph, unsigned int sweeps = 5000,
                                    numeric_t Tstart = numeric_t(15.0), numeric_t Tend = numeric_t(0.05))
      : m_graph(graph), m_sweeps(sweeps), m_T_start(Tstart), m_T_end(Tend)
   {}

   ts::image<label_t> operator()(const typename _DataTraits::input_type& input) const
   {
      return inference::simulated_annealing(m_graph, input, m_sweeps, m_T_start, m_T_end);
   }
protected:
   unsigned int m_sweeps;
   numeric_t m_T_start, m_T_end;
   const factor_graph<_DataTraits>& m_graph;
};

// Mean field MPM, maximum posterior marginal classifier
template <typename DataTraits>
class mpm_nmf_classifier {
public:
   mpm_nmf_classifier(const factor_graph<DataTraits>& graph, numeric_t conv_tol = 1.0e-6,
      size_t max_iter = 50)
      : m_graph(graph), m_conv_tol(conv_tol), m_max_iter(max_iter)
   {
   }

   typename inference::naive_mean_field<DataTraits>::inference_result_type infer_marginals(
      const typename DataTraits::input_type& input) const
   {
      // Inference
      dtf::inference::naive_mean_field<DataTraits> nmf(m_graph, input);
      numeric_t log_z_lb_prev = -std::numeric_limits<numeric_t>::infinity();
      for (size_t iter = 1; iter <= m_max_iter; ++iter)
      {
         numeric_t log_z_lb = nmf.sweep();
         // NMF lower bound should be monotonic, but for extremly symmetric fields and large energies
         // can be numerically negative.
         if (std::fabs(log_z_lb - log_z_lb_prev) < m_conv_tol)
            break;

         log_z_lb_prev = log_z_lb;
      }
      return nmf.marginals();
   }

   ts::image<label_t> mpm_from_marginals(
      const typename dtf::inference::naive_mean_field<DataTraits>::inference_result_type& posterior) const
   {
      // MPM decisions
      ts::image<label_t> mpm_pred(posterior.width(), posterior.height());
      ts::parallel_pointwise_inplace(mpm_pred, posterior,
         [](label_t& mpm, const inference::naive_mean_field<DataTraits>::marginal_type& post) -> void
      {
         numeric_t max_val = 0;
         for (label_t li = 0; li < post.size(); ++li)
         {
            if (post[li] > max_val)
            {
               mpm = li;
               max_val = post[li];
            }
         }
      });
      return mpm_pred;
   }

   std::pair<ts::image<label_t>, typename dtf::inference::naive_mean_field<DataTraits>::inference_result_type>
   get_mpm_and_marginals(const typename DataTraits::input_type& input) const
   {
      auto posterior = infer_marginals(input);
      return std::pair<ts::image<label_t>, typename dtf::inference::naive_mean_field<DataTraits>::inference_result_type>(
         mpm_from_marginals(posterior), posterior);
   }

   ts::image<label_t> operator()(const typename DataTraits::input_type& input) const
   {
      return mpm_from_marginals(infer_marginals(input));
   }

protected:
   const factor_graph<DataTraits>& m_graph;
   double m_conv_tol;
   size_t m_max_iter;
};

template <typename _TDatabase, typename _TClassifier>
ts::image<unsigned int> confusion_matrix(const _TDatabase& database, const _TClassifier& classifier)
{
   ts::image<unsigned int> confusion(database.label_count, database.label_count);
   std::fill(confusion.begin(), confusion.end(), 0u);

   ts::thread_locals<ts::image<unsigned int>> locals(confusion);
   for (database_iterator<_TDatabase> i(database); i; ++i)
   {
      auto labelling = classifier(i.training());
      const auto& gt = i.ground_truth();
      ts::parallel_for_each_pixel_xy(gt, [&](unsigned int x, unsigned int y)
      {
         (*locals)(labelling(x, y), gt(x, y))++;
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

template <typename _TDatabase, typename _TClassifier>
double classification_accuracy(const _TDatabase& database, const _TClassifier& classifier)
{
   ts::image<unsigned int> confusion = confusion_matrix(database, classifier);
   unsigned int total = 0, correct = 0;
   ts::for_each_pixel_xy(confusion, [&](unsigned int x, unsigned int y)
   {
      total += confusion(x, y);
      if (x == y)
         correct += confusion(x, y);
   });
   return static_cast<double>(correct) / total;
}

}
}
