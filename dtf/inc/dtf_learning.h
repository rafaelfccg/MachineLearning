/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 4 November 2011
   
*/
#pragma once

#include "dtf.h"
#include "minimize.h"

namespace dtf
{

template <typename _TDatabase, typename _DataTraits = database_traits<_TDatabase>>
class objective
{
public:
   objective(  const dtf::factor_graph<_DataTraits>& structure, const _TDatabase& database)
      : m_database(database), m_structure(structure)
   {
   }
   objective(  const dtf::factor_graph<_DataTraits>& structure, const _TDatabase& database, const std::vector<prior_t>& priors)
      : m_database(database), m_structure(structure), m_priors(priors)
   {
   }
   bool check_derivative(double x_range, unsigned int test_count = 100,
		           double dim_eps = 1e-8, double grad_tol = 1e-5) const
   {
      // TODO
      return true;
   }
   template <typename XIt>
   double operator()(XIt x) const
   {
      return dtf::compute::ComputeObjective(m_structure, m_database, x, m_priors); 
   }
   template <typename XIt, typename GIt>
   double operator()(XIt x, GIt g) const
   {
      return dtf::compute::ComputeObjectiveWithGradients(m_structure, m_database, x, m_priors, g); 
   }
   double Eval(const std::vector<double>& x, std::vector<double>& grad) const
   {
      return operator()(x.begin(), grad.begin()); 
   }
protected:
   const _TDatabase& m_database;
   const dtf::factor_graph<_DataTraits>& m_structure;
   std::vector<prior_t> m_priors;
};

namespace learning
{

// Optimize the weights stored in the factor graph
template <typename _TDatabase, typename _DataTraits>
void OptimizeWeights(factor_graph<_DataTraits>& graph,
                     const _TDatabase& database,
                     double conv_tol = 1e-5, 
                     unsigned int max_iter = 0,
                     unsigned int lbfgs_m = 500)
{
   std::vector<weight_t> weights(dtf_count_weights(graph), 0);
   objective<_TDatabase, _DataTraits> fn(graph, database);
   ts::minimize::LBFGS<weight_t>(fn, static_cast<unsigned int>(weights.size()), weights, conv_tol, max_iter, lbfgs_m, true);
   dtf_set_all_weights(graph, weights.begin(), weights.end());
}

// Optimize the weights stored in the factor graph
template <typename _TDatabase, typename _DataTraits>
void OptimizeWeights(factor_graph<_DataTraits>& graph,
                     const std::vector<prior_t>& priors,
                     const _TDatabase& database,
                     double conv_tol = 1e-5, 
                     unsigned int max_iter = 0,
                     unsigned int lbfgs_m = 500)
{
   std::vector<weight_t> weights(dtf_count_weights(graph), 0);
   objective<_TDatabase, _DataTraits> fn(graph, database, priors);
   ts::minimize::LBFGS<weight_t>(fn, static_cast<unsigned int>(weights.size()), weights, conv_tol, max_iter, lbfgs_m, true);
   dtf_set_all_weights(graph, weights.begin(), weights.end());
}

}
}