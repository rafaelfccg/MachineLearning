/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 27 March 2012
   
*/
#pragma once

#include "dtf.h"

#include "trw-s/MRFEnergy.h"
#include "trw-s/MRFEnergy.cpp"
#include "trw-s/minimize.cpp"
#include "trw-s/treeProbabilities.cpp"

namespace dtf
{
namespace inference
{

/// MAP inference via TRW
template <typename _DataTraits, typename _Img>
ts::image<label_t> map_trw(const _Img& input, const factor_graph<_DataTraits>& graph, unsigned int maxIter = 30, double minConv = 1e-3)
{
   const label_t label_count = _DataTraits::label_count;
   unsigned int cx = input.width(), cy = input.height();

   typedef MRFEnergy<TypeGeneral>::NodeId NodeId;
   std::unique_ptr<MRFEnergy<TypeGeneral>> mrf(new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
   ts::image<NodeId> imgNodeIds(cx, cy);

   // Set up the inference problem
   // Allow the feature to perform its pre-processing step:
   //auto preProcessed = TFeature::PreProcess(input);
   std::array<TypeGeneral::REAL, label_count> Dx;
   std::array<TypeGeneral::REAL, label_count * label_count> V;

   // Accumulate the unary energies in this image:
   std::array<weight_acc_t, label_count> zero_weights = { 0 };
   ts::image<decltype(zero_weights)> accumulator(cx, cy);
   std::fill(accumulator.begin(), accumulator.end(), zero_weights);
   visit_dtf_factors(graph, [&](const dtf_factor_base<_DataTraits>& factor)
   {
      if (factor.variables() != 1)
         return;
      factor.visit_instances(input, [&](pos_cit variable_start, pos_cit variable_end, weight_cit energy_start, weight_cit energy_end)
      {
         auto dst = std::begin(accumulator(variable_start[0].x, variable_start[0].y));
         for (weight_cit i = energy_start; i != energy_end; ++i, ++dst)
            *dst += *i;
      });
   });
   ts::for_each_pixel_xy(accumulator, [&](unsigned int x, unsigned int y)
   {
      for (label_t i = 0; i < label_count; ++i)
         Dx[i] = accumulator(x, y)[i];
      imgNodeIds(x, y) = mrf->AddNode(TypeGeneral::LocalSize(label_count), TypeGeneral::NodeData(&Dx[0]));
   });

   visit_dtf_factors(graph, [&](const dtf_factor_base<_DataTraits>& factor)
   {
      if (factor.variables() != 2)
         return;
      factor.visit_instances(input, [&](pos_cit variable_start, pos_cit variable_end, weight_cit energy_start, weight_cit energy_end)
      {
         auto nodeIdA = imgNodeIds(variable_start[0].x, variable_start[0].y);
         auto nodeIdB = imgNodeIds(variable_start[1].x, variable_start[1].y);
         auto dst = std::begin(V);
         for (weight_cit i = energy_start; i != energy_end; ++i, ++dst)
            *dst = static_cast<TypeGeneral::REAL>(*i);
         mrf->AddEdge(nodeIdA, nodeIdB, TypeGeneral::EdgeData(TypeGeneral::GENERAL, &V[0]));
      });
   });

   MRFEnergy<TypeGeneral>::Options options;
   options.m_iterMax = maxIter; // maximum number of iterations
   options.m_eps = minConv;
   options.m_printIter = 0;
   TypeGeneral::REAL energy, lowerBound;
   mrf->Minimize_TRW_S(options, lowerBound, energy);

   return ts::pointwise_xy(imgNodeIds, [&](unsigned int x, unsigned int y) -> label_t
   {
      return static_cast<label_t>(mrf->GetSolution(imgNodeIds(x, y)));
   });
}

// Return a partitioning of the image plane into conditionally independent subsets of variables (colours)
template<typename _DataTraits, typename _Img>
std::vector<std::vector<offset_t>> greedy_graph_colouring(const factor_graph<_DataTraits>& graph, const _Img& input)
{
   // Build the list of variables connected to an origin
   auto connections = dtf_map_connected_variables(graph);

   // Determine the graph colouring using a greedy method
   ts::image<unsigned char> colouring(input.width(), input.height());
   std::fill(std::begin(colouring), std::end(colouring), 0xFF);

   std::vector<std::vector<offset_t>> samples;
   ts::for_each_pixel_xy(colouring, [&](unsigned int x, unsigned int y)
   {
      unsigned char colour; 
      for (colour = 0; ; ++colour)
      {
         bool conflict = false;
         // See whether any connected variables above this one or to its left have a conflict
         for (auto i = std::begin(connections); i != std::end(connections); ++i)
         {
            offset_t var = offset_t(x, y) + *i;
            if (colouring.contains(var.x, var.y))
            {
               if (colouring(var.x, var.y) == colour)
               {
                  conflict = true;
                  break;
               }
            }
         }
         if (!conflict)
            break;
      }
      if (colour >= samples.size())
         samples.resize(colour + 1);
      colouring(x, y) = colour;
      samples[colour].push_back(offset_t(x, y));
   });

   return samples;
}


/*
   Gibbs Sampling
   --------------

   In each sweep, visits each variable exactly once and re-samples the label

   In the simplest implementation, we would visit each variable in serial.
   At the variable, we compute a conditional distribution over the possible labels for that variable.
   The distribution is conditioned on the labels of connected variables, which are treated as fixed.
   First the energies are summed for each label over all connected factor instances.
   The energy values are divided by the temperature. This makes distributions more confident as the system cools.
   Exponentiating gives the conditional distribution values.
   This distribution is then sampled by using the standard cumulative distribution.
   The variable is then set to the label that is obtained by this sampling.

   In practice we prefer a parallel implementation.
   However, variables that are connected via a factor instance cannot be resampled simultaneously.
   If we solve a graph colouring problem, we can determine subsets of variables that are not connected.
   Each subset then contains variables that can be resampled in parallel,
   but the subsets themselves must be processed in serial.
   This is like the well known chequerboard schedule for message passing algorithms.
   Rather than solve a full graph colouring, it will be sufficient to use the bbox of a factor type and
   assign a different colour to each variable within the bbox.

   A significant issue is that this method will be called many times for the same input image.
   It would be most undesirable to have to re-compute the pre-processed input for each factor type
   at every call. Fortunately the leaf node for each factor/variable pair does not change for
   a given input image. Therefore we will store the leaf node indices for each factor and variable.

   If we do the outer loop over variables, and the inner loop over factor types, that leaves us
   with the inconvenient situation that we must make several virtual function calls and dispatches
   for each variable, hurting performance. Instead, we could maintain a buffer over variables of 
   1D accumulated energies. We loop over factor types at the outer level, and over variables of
   the current colour at the inner level, accumulating the energies. In a second pass, we visit
   all variables of the current colour, performing the resampling.

   TODO: We could also try an optimization where we store the 1D table of energies for each variable
   between sweeps. These energies would only need re-calculating if a connected variable has changed
   its label in the last sweep. As the temperature decreases, this could improve performance significantly.
*/
template <typename _DataTraits>
class gibbs_chain
{
public:
   gibbs_chain(const factor_graph<_DataTraits>& graph,
               const typename _DataTraits::input_type& input,
               unsigned long seed_base = std::mt19937::default_seed);

   void temperature(double T);
   double temperature() const { return temperatue_; }

   // Visit each variable exactly once and re-sample the label there.
   // Returns the energy of the new state.
   numeric_t sweep() { return temperature_ < 1.0 ? sweep_cache() : sweep_nocache(); }

   // Get the current labelling
   const ts::image<label_t>& state() const { return state_; }

private:
   numeric_t sweep_cache();
   numeric_t sweep_nocache();

   bool changed_temp_;
   double temperature_;
   ts::image<label_t> state_;
   std::vector<ts::image<size_t>> leaves_;
   const factor_graph<_DataTraits>& graph_;
   std::vector<std::vector<offset_t>> samples_;
   std::vector<offset_t> connections_;
   ts::thread_locals<std::mt19937> engines_;
   ts::image<bool> flags_;
   ts::image<std::array<weight_acc_t, _DataTraits::label_count>> energies_;
   ts::image<std::array<numeric_t, _DataTraits::label_count>> exp_energies_;
};

template <typename _DataTraits>
gibbs_chain<_DataTraits>::gibbs_chain(const factor_graph<_DataTraits>& graph, 
                                      const typename _DataTraits::input_type& input,
                                      unsigned long seed_base = std::mt19937::default_seed)
   : graph_(graph), state_(input.width(), input.height()), flags_(input.width(), input.height()),
     energies_(input.width(), input.height()), exp_energies_(input.width(), input.height()), 
     temperature_(1), changed_temp_(true)
{
   // Initialize each variable from a uniform iid
   for (unsigned int i = 0; i < engines_.size(); ++i)
      engines_[i].seed(seed_base + i);
   ts::parallel_for_each_pixel(state_, [&](label_t& label)
   {
      label = std::uniform_int_distribution<label_t>(0, graph.label_count - 1)(*engines_);
   });

   // Store the leaf node indices for each factor instance
   visit_dtf_factors(graph_, [&](const dtf_factor_base<_DataTraits>& factor)
   {
      leaves_.push_back(factor.leaf_image(input));
   });

   // Obtain graph colouring to perform independent Gibbs sweeps in parallel
   samples_ = greedy_graph_colouring(graph, input);
   connections_ = dtf_map_connected_variables(graph);
}

template <typename _DataTraits>
void gibbs_chain<_DataTraits>::temperature(double T)
{
   if (temperature_ >= 1.0 && T < 1.0) // Switching on caching
      std::fill(std::begin(flags_), std::end(flags_), true);
   if (temperature_ != T)
   {
      temperature_ = T;
      changed_temp_ = true;
   }
}

template <typename _DataTraits>
numeric_t gibbs_chain<_DataTraits>::sweep_nocache()
{
   double invT = 1.0 / temperature_;

   // Total energy of new state
   ts::thread_locals<weight_acc_t> e_total(0);

   // Zero init the energies
   memset(&energies_(0, 0), 0, energies_.image_bytes());

   // For each colour of the graph colouring
   for (auto samples = samples_.cbegin(); samples != samples_.cend(); ++samples)
   {
      // For each factor type
      size_t factor_index = 0;
      visit_dtf_factors(graph_, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         // Lookup stored leaf image for this factor type
         const ts::image<size_t>& leaves = leaves_[factor_index];

         // Accumulate the energies for factor instances of this type at variables of the current colour
         factor.accumulate_energies(*samples, leaves, state_, energies_);
         ++factor_index;
      });

      // Parallel-For each variable with the current colour
      ts::parallel_for((*samples).cbegin(), (*samples).cend(), [&](const std::vector<offset_t>::const_iterator var)
      {
         // Exponentiate the energies
         const std::array<weight_acc_t, graph_.label_count>& e = energies_((*var).x, (*var).y);
         std::array<numeric_t, graph_.label_count> prob;
         robust_exp_norm(e.begin(), e.end(), prob.begin(), invT);

         // Sample from the distribution using the cdf
         numeric_t u = std::uniform_real_distribution<numeric_t>()(*engines_);
         numeric_t cumsum = 0;
         label_t sampled_label;
         for (sampled_label = 0; sampled_label < graph_.label_count - 1; ++sampled_label)
         {
            cumsum += prob[sampled_label];
            if (u < cumsum)
               break;
         }
         
         // Set the current variable to the sampled value
         state_((*var).x, (*var).y) = sampled_label;
         *e_total += e[sampled_label];
      });
   }
   return e_total.reduce(numeric_t(0), [](numeric_t& sum, weight_acc_t next) { sum += next; } );
}

template <typename _DataTraits>
numeric_t gibbs_chain<_DataTraits>::sweep_cache()
{
   double invT = 1.0 / temperature_;

   // Total energy of new state
   ts::thread_locals<weight_acc_t> e_total(0);

   // For each colour of the graph colouring
   for (auto samples = samples_.cbegin(); samples != samples_.cend(); ++samples)
   {
      // Zero init the energies that need re-accumulating
      ts::parallel_for((*samples).cbegin(), (*samples).cend(), [&](const std::vector<offset_t>::const_iterator var)
      {
         if (flags_((*var).x, (*var).y))
         {
            std::array<weight_acc_t, graph_.label_count>& e = energies_((*var).x, (*var).y);
            std::fill(std::begin(e), std::end(e), weight_acc_t(0));
         }
      });

      // For each factor type
      size_t factor_index = 0;
      visit_dtf_factors(graph_, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         // Lookup stored leaf image for this factor type
         const ts::image<size_t>& leaves = leaves_[factor_index];

         // Accumulate the energies for factor instances of this type at variables of the current colour
         factor.accumulate_energies(*samples, leaves, state_, energies_, flags_);
         ++factor_index;
      });

      // Parallel-For each variable with the current colour
      ts::parallel_for((*samples).cbegin(), (*samples).cend(), [&](const std::vector<offset_t>::const_iterator var)
      {
         // Exponentiate the scaled energies
         // Scale the energies for the current variable by the inverse temperature
         if (changed_temp_ || flags_((*var).x, (*var).y))
         {
            std::array<numeric_t, graph_.label_count>& probs = exp_energies_((*var).x, (*var).y);
            std::array<weight_acc_t, graph_.label_count>& esrc = energies_((*var).x, (*var).y);
            robust_exp_norm(esrc.begin(), esrc.end(), probs.begin(), invT);
            flags_((*var).x, (*var).y) = false;
         }
      });

      // Parallel-For each variable with the current colour
      ts::parallel_for((*samples).cbegin(), (*samples).cend(), [&](const std::vector<offset_t>::const_iterator var)
      {
         // Sample a uniform random number
         std::array<numeric_t, graph_.label_count>& e = exp_energies_((*var).x, (*var).y);
         numeric_t u = std::uniform_real_distribution<numeric_t>()(*engines_);

         // Sample from the distribution using the cdf
         numeric_t cumsum = 0;
         label_t sampled_label;
         for (sampled_label = 0; sampled_label < graph_.label_count - 1; ++sampled_label)
         {
            cumsum += e[sampled_label];
            if (u < cumsum)
               break;
         }
         
         // Set the current variable to the sampled value
         if (state_((*var).x, (*var).y) != sampled_label)
         {
            state_((*var).x, (*var).y) = sampled_label;
            for (auto i = std::begin(connections_); i != std::end(connections_); ++i)
            {
               offset_t connected = *var + *i;
               if (flags_.contains(connected.x, connected.y))
                  flags_(connected.x, connected.y) = true;
            }
         }
         *e_total += energies_((*var).x, (*var).y)[sampled_label];
      });
   }

   changed_temp_ = false;
   return e_total.reduce(numeric_t(0), [](numeric_t& sum, weight_acc_t next) { sum += next; } );
}

template <typename _DataTraits>
ts::image<label_t> simulated_annealing(const factor_graph<_DataTraits>& graph,
                                       const typename _DataTraits::input_type& input,
                                       unsigned int sweeps = 5000,
                                       numeric_t temp_begin = numeric_t(15.0),
                                       numeric_t temp_end = numeric_t(0.05))
{
   gibbs_chain<_DataTraits> chain(graph, input);
   numeric_t alpha = std::exp(std::log(temp_end / temp_begin) / sweeps);
   numeric_t T = temp_begin;
   for (unsigned int i = 0; i < sweeps; ++i, T *= alpha)
   {
      chain.temperature(T);
      chain.sweep();
   }
   return chain.state();
}

// Naive mean field inference
// Reference:
// [Nowozin2011], Nowozin, Lampert, "Structured Learning and Prediction in Computer Vision", Section 3.3.1.
// [Koller2010], Koller, Friedman, "Probabilistic Graphical Models", Section 11.5.1.2.
template <typename _DataTraits>
class naive_mean_field
{
public:
   typedef std::array<weight_acc_t, _DataTraits::label_count> marginal_type;
   typedef ts::image<marginal_type> inference_result_type;

   naive_mean_field(const factor_graph<_DataTraits>& graph,
               const typename _DataTraits::input_type& input);

   // Visit each variable exactly once and update the mean field distribution there.
   // Returns the mean field lower bound to logZ.
   numeric_t sweep();

   // Get the current labelling
   const inference_result_type& marginals() const {
      return q_field;
   }

private:
   inference_result_type q_field;
   std::vector<ts::image<size_t>> leaves_;
   const factor_graph<_DataTraits>& graph_;
   std::vector<std::vector<offset_t>> samples_;
   std::vector<offset_t> connections_;
};

template <typename _DataTraits>
naive_mean_field<_DataTraits>::naive_mean_field(const factor_graph<_DataTraits>& graph, 
                                                const typename _DataTraits::input_type& input)
   : graph_(graph), q_field(input.width(), input.height())
{
   // Initial mean field distribution: uniform plus small symmetry breaking perturbation
   weight_acc_t uni = 1.0 / weight_acc_t(_DataTraits::label_count);
   ts::thread_locals<std::mt19937> eng;
   std::uniform_real_distribution<weight_acc_t> runi;
   ts::parallel_for_each_pixel(q_field, [&](inference_result_type::iterator::value_type& qdist)
   {
      std::fill(qdist.begin(), qdist.end(), uni);
      weight_acc_t qsum = 0;
      for (size_t li = 0; li < qdist.size(); ++li) {
         qdist[li] += 1.0e-2*runi(*eng);
         qsum += qdist[li];
      }
      for (size_t li = 0; li < qdist.size(); ++li)
         qdist[li] /= qsum;
   });

   // Store the leaf node indices for each factor instance
   visit_dtf_factors(graph_, [&](const dtf_factor_base<_DataTraits>& factor)
   {
      leaves_.push_back(factor.leaf_image(input));
   });
   
   // Build the list of variables connected to an origin
   connections_ = dtf_map_connected_variables(graph);

   // Obtain graph colouring to perform independent mean field updates in parallel
   samples_ = greedy_graph_colouring(graph, input);
}

template <typename _DataTraits>
numeric_t naive_mean_field<_DataTraits>::sweep()
{
   // Total logZ of new field
   ts::thread_locals<weight_acc_t> logZ_total(0);

   // For each colour of the graph colouring
   for (auto samples = samples_.cbegin(); samples != samples_.cend(); ++samples)
   {
      // Prepare accumulation of weighted energies: set q_field[colour] to zero
      ts::parallel_for((*samples).cbegin(), (*samples).cend(), [&](const std::vector<offset_t>::const_iterator var)
      {
         std::array<weight_acc_t, graph_.label_count>& qp = q_field((*var).x, (*var).y);
         std::fill(qp.begin(), qp.end(), 0.0);
      });

      // For each factor type
      size_t factor_index = 0;
      visit_dtf_factors(graph_, [&](const dtf_factor_base<_DataTraits>& factor)
      {
         // Lookup stored leaf image for this factor type
         const ts::image<size_t>& leaves = leaves_[factor_index];

         // Accumulate the energies, overwriting q_field for the offsets of the current color
         factor.accumulate_energies(*samples, leaves, q_field);
         ++factor_index;
      });

      // Parallel-For each variable with the current colour
      ts::parallel_for((*samples).cbegin(), (*samples).cend(), [&](const std::vector<offset_t>::const_iterator var)
      {
         // Exponentiate the energies
         std::array<weight_acc_t, graph_.label_count>& qp = q_field((*var).x, (*var).y);
         std::array<numeric_t, graph_.label_count> probs;
         *logZ_total += robust_exp_norm(qp.begin(), qp.end(), probs.begin());
         for (label_t i = 0; i < graph_.label_count; ++i)
         {
            *logZ_total += probs[i] * qp[i];
            qp[i] = probs[i];
         }
      });
   }

   // All site distributions have been updated now and we have computed the entropy.  Now
   // compute the expected energies under the mean field distribution.
   numeric_t energy_sum = 0;
   size_t factor_index = 0;
   visit_dtf_factors(graph_, [&](const dtf_factor_base<_DataTraits>& factor)
   {
      // Sum the weighted energies
      energy_sum += factor.sum_energies(leaves_[factor_index], q_field);
      ++factor_index;
   });

   numeric_t entropy_sum = logZ_total.reduce(numeric_t(0),
      [](numeric_t& sum, weight_acc_t next) { sum += next; });

   return entropy_sum - energy_sum;
}


}
}