/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 4 November 2011
   
*/
#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <memory>
#include <random>
#include "decision_trees.h"
#include "image.h"
#include "make_unique.h"

namespace dtf
{

template <typename T>
struct pos
{
   T x, y;

   pos() : x(0), y(0) {}
   pos(T x, T y) : x(x), y(y) {}
   template <typename T2> pos(const pos<T2>& rhs) : x(rhs.x), y(rhs.y) {}
   bool operator ==(const pos& rhs) const
   {
      return x == rhs.x && y == rhs.y;
   }
   bool operator !=(const pos& rhs) const
   {
      return !operator ==(rhs);
   }
   pos<T> operator -(const pos<T>& rhs) const
   {
      return pos<T>(x - rhs.x, y - rhs.y); 
   }
   pos<T> operator +(const pos<T>& rhs) const
   {
      return pos<T>(x + rhs.x, y + rhs.y);
   }
   pos<T> operator -() const
   {
      return pos<T>(-x, -y);
   }
};

template <typename T>
struct rect
{
   rect() : left(0), top(0), right(0), bottom(0) {}
   rect(T l, T t, T r, T b) : left(l), top(t), right(r), bottom(b) {}

   rect& operator |=(const pos<T>& pt)
   {
      left = std::min(left, pt.x);
      right = std::max(right, pt.x);
      top = std::min(top, pt.y);
      bottom = std::max(bottom, pt.y);
      return *this;
   }
   rect& operator +=(const pos<T>& pt)
   {
      left += pt.x;
      top += pt.y;
      right += pt.x;
      bottom += pt.y;
      return *this;
   }
   rect<T> deflate_rect(const rect& rhs) const
   {
      return rect(left - rhs.left, top - rhs.top, right - rhs.right, bottom - rhs.bottom);
   }

   T width() const { return right - left; }
   T height() const { return bottom - top; }

   bool contains(const pos<T>& pt) const
   {
      return pt.x >= left && pt.x < right && pt.y >= top && pt.y < bottom;
   }

   T left, top, right, bottom;
};

template <typename T>
inline T ipow(T a, unsigned int b)
{
   T rv = (T)1;
   while (b-- > 0)
      rv *= a;
   return rv;
}

typedef pos<int> offset_t;
typedef rect<unsigned int> box_t;
typedef pos<unsigned int> pos_t;
typedef double numeric_t; 
typedef numeric_t weight_t;
typedef numeric_t grad_t;
typedef unsigned char label_t;
typedef int path_t;
typedef std::vector<weight_t>::const_iterator weight_cit;
typedef std::vector<weight_t>::iterator weight_it;
typedef weight_t weight_acc_t;
typedef std::vector<weight_acc_t>::iterator weight_acc_it;
typedef std::vector<grad_t>::const_iterator grad_cit;
typedef std::vector<grad_t>::iterator grad_it;
typedef grad_t grad_acc_t;
typedef std::vector<grad_acc_t>::iterator grad_acc_it;
typedef std::vector<pos_t>::const_iterator pos_cit;

template <label_t label_count, unsigned int variable_count> struct power { static const unsigned int raise; };
template <label_t label_count> struct power<label_count, 1> { static const unsigned int raise = label_count; };
template <label_t label_count> struct power<label_count, 2> { static const unsigned int raise = label_count * label_count; };
template <label_t label_count> struct power<label_count, 3> { static const unsigned int raise = label_count * label_count * label_count; };
template <label_t label_count> struct power<label_count, 4> { static const unsigned int raise = power<label_count, 2>::raise * power<label_count, 2>::raise; };

template <typename _Titerator>
struct range
{
   _Titerator begin() const { return start; }
   _Titerator end() const { return stop; }

   _Titerator start;
   _Titerator stop;
};

template <typename _TDatabase>
struct training_data
{
private:
   static const _TDatabase* _dummy;
public:
   typedef typename std::remove_reference<decltype(_dummy->get_training(0))>::type training_type;
   typedef typename std::remove_reference<decltype(_dummy->get_ground_truth(0))>::type ground_truth_type;
   typedef typename std::remove_reference<decltype(_dummy->get_samples(0))>::type samples_type;
   typedef typename std::remove_reference<decltype(samples_type().begin())>::type sample_iterator;

   training_data(const training_type& t, const ground_truth_type& g, const samples_type& s) : 
      train(t), gt(g), samples(s) {}

   training_type train;
   ground_truth_type gt;
   samples_type samples;
};

template <typename _TDatabase>
class database_iterator
{
public:
   typedef typename training_data<_TDatabase>::training_type training_type;
   typedef typename training_data<_TDatabase>::ground_truth_type ground_truth_type;
   typedef typename training_data<_TDatabase>::sample_iterator sample_iterator;
   typedef typename training_data<_TDatabase>::samples_type sample_range;

   database_iterator(const _TDatabase& database, size_t pos = 0)
      : m_database(database), m_pos(pos),
         m_data(database.get_training(pos), database.get_ground_truth(pos), database.get_samples(pos))
   {
   }
   database_iterator& operator++()
   {
      if (++m_pos < m_database.size())
         get_data();
      return *this;
   }
   operator bool() const { return m_pos < m_database.size(); }
   bool operator !=(const database_iterator& rhs) const { return rhs.m_database != m_database || rhs.m_pos != m_pos; }
   const training_type& training() const { return m_data.train; }
   const ground_truth_type& ground_truth() const { return m_data.gt; }
   sample_iterator samples_begin() const { return m_data.samples.start; }
   sample_iterator samples_end() const { return m_data.samples.stop; }
   const sample_range& samples() const { return m_data.samples; }
   size_t index() const { return m_pos; }

protected:
   void get_data()
   {
      m_data.train = m_database.get_training(m_pos);
      m_data.gt = m_database.get_ground_truth(m_pos);
      m_data.samples = m_database.get_samples(m_pos);
   }
   size_t m_pos;
   training_data<_TDatabase> m_data;
   const _TDatabase& m_database;
};

template <label_t label_count, typename _Input, typename _Samples, typename _Labelling>
struct data_traits
{
   static const label_t label_count = label_count;
   typedef _Input input_type;
   typedef _Samples samples_type;
   typedef _Labelling labelling_type;
};

template <typename _Database>
struct database_traits
{
   static const label_t label_count = _Database::label_count;
   typedef typename training_data<_Database>::training_type input_type;
   typedef typename training_data<_Database>::samples_type samples_type;
   typedef typename training_data<_Database>::ground_truth_type labelling_type;
};

template <typename _DataTraits> class factor_visitor;

template <typename _Traits>
class factor_base
{
public:
   static const label_t label_count = _Traits::label_count;
   typedef typename _Traits::input_type input_type;
   typedef typename _Traits::samples_type samples_type;
   typedef typename _Traits::labelling_type labelling_type;

   factor_base() {}
   virtual ~factor_base() {}

   virtual void accept(factor_visitor<_Traits>& visitor);
   virtual void accept(factor_visitor<_Traits>& visitor) const;

   virtual numeric_t energy(const labelling_type& labelling, const input_type& input) const = 0;
   virtual unsigned int variables() const = 0;
   virtual offset_t offset(unsigned int var) const = 0;
   virtual void save(std::ostream& os) const = 0;
   virtual void load(std::istream& is) = 0;

   friend std::istream& operator >>(std::istream& is, factor_base& f)
   {
      f.load(is);
      return is;
   }
   friend std::ostream& operator <<(std::ostream& os, const factor_base& f)
   {
      f.save(os);
      return os;
   }
};

template <typename _DataTraits>
struct factor_graph
{
protected:
   typedef factor_base<_DataTraits> factor_base_type;
   typedef std::shared_ptr<factor_base_type> elem_type;
   typedef std::vector<elem_type> vec_type;
   typedef typename vec_type::iterator iterator;
   typedef typename vec_type::const_iterator const_iterator;
public:
   typedef typename _DataTraits::labelling_type labelling_type;
   typedef typename _DataTraits::input_type input_type;
   static const label_t label_count = _DataTraits::label_count;

   factor_graph() {}

   iterator begin() { return m_factors.begin(); }
   iterator end() { return m_factors.end(); }
   const_iterator begin() const { return m_factors.begin(); }
   const_iterator end() const { return m_factors.end(); }
   elem_type operator[](size_t index) const { return m_factors[index]; }

   unsigned int size() const { return static_cast<unsigned int>(m_factors.size()); }

   void accept(factor_visitor<_DataTraits>& visitor)
   {
      for each (const elem_type& f in m_factors)
         f->accept(visitor);
   }
   void accept(factor_visitor<_DataTraits>& visitor) const
   {
      for each (std::shared_ptr<const factor_base_type> f in m_factors)
         f->accept(visitor);
   }
   numeric_t energy(const labelling_type& labelling, const input_type& input) const
   {
      numeric_t sum = 0;
      for each (std::shared_ptr<const factor_base_type> f in m_factors)
         sum += f->energy(labelling, input);
      return sum;
   }
   void add_factor(elem_type factor)
   {
      m_factors.push_back(factor);
   }
   template <typename _TFactor> void move_factor(_TFactor&& factor)
   {
      m_factors.push_back(std::make_shared<_TFactor>(factor));
   }
   template <typename _TFactor> void push_back(const _TFactor& factor)
   {
      m_factors.push_back(elem_type(new _TFactor(factor)));
   }

   void load(std::istream& is, const ts::factory<factor_base<_DataTraits>>& factory)
   {
      m_factors.clear();
      unsigned int size = ts::read<unsigned int>(is);
      for (unsigned int i = 0; i < size; ++i)
         add_factor(ts::load_object(is, factory));
   }

   void save(std::ostream& os) const
   {
      ts::write<unsigned int>(os, size());
      for (unsigned int i = 0; i < size(); ++i)
         ts::save_object(os, *m_factors[i]);
   }

protected:
   vec_type m_factors;
};

template <typename _Traits>
class dtf_factor_base : public factor_base<_Traits>
{
public:
   virtual void accept(factor_visitor<_Traits>& visitor) override
   {
      visitor.visit_dtf_factor(*this);
   }
   virtual void accept(factor_visitor<_Traits>& visitor) const override
   {
      visitor.visit_dtf_factor(*this);
   }

   virtual void accumulate_unary_energies(const input_type& input, ts::image<std::array<numeric_t, _Traits::label_count>>& acc) const = 0;
   virtual ts::image<path_t> accumulate_variables(const input_type& input, const labelling_type& gt, const samples_type& samples, std::vector<weight_t>& var_acc, weight_cit weights) const = 0;
   virtual void accumulate_gradients(const ts::image<path_t>& paths, const labelling_type& gt, const samples_type& samples, const std::vector<weight_t>& var_acc, ts::thread_locals<grad_acc_it>& grads) const = 0;
   virtual void visit_instances(const input_type& input, std::function<void(pos_cit, pos_cit, weight_cit, weight_cit)> func) const = 0;
   virtual size_t accumulate_weights(weight_cit wbegin, weight_cit wend) = 0;
   virtual void accumulate_energies(const std::vector<offset_t>& samples, const ts::image<size_t>& leaves, const ts::image<label_t>& labelling, ts::image<std::array<weight_acc_t, label_count>>& energies, const ts::image<bool>& flags) const = 0;
   virtual void accumulate_energies(const std::vector<offset_t>& samples, const ts::image<size_t>& leaves, const ts::image<label_t>& labelling, ts::image<std::array<weight_acc_t, label_count>>& energies) const = 0;
   virtual void accumulate_energies(const std::vector<offset_t>& samples, const ts::image<size_t>& leaves, ts::image<std::array<weight_acc_t, label_count>>& q_field) const = 0;
   virtual numeric_t sum_energies(const ts::image<size_t>& leaves, const ts::image<std::array<weight_acc_t, label_count>>& q_field) const = 0;

   virtual ts::image<size_t> leaf_image(const input_type& input) const = 0;

   // Return the number of weights/coefficients used by this DTF factor
   size_t get_weight_count() const 
   {
      return m_total_weight_count;
   }
   unsigned int get_weights_per_node() const
   {
      return m_weights_per_node;
   }
   weight_it energies(size_t leaf_index = 0)
   {
      return m_leaf_weights.begin() + leaf_index * m_weights_per_node;
   }
   weight_cit energies(size_t leaf_index = 0) const
   {
      return m_leaf_weights.begin() + leaf_index * m_weights_per_node;
   }
   size_t energies_size() const { return m_leaf_weights.size(); }
   
   virtual void save(std::ostream& os) const 
   {
      ts::write(os, m_weights_per_node);
      ts::write(os, m_total_weight_count);
      ts::write_raw(os, m_leaf_weights);
   }

   virtual void load(std::istream& is)
   {
      ts::read(is, m_weights_per_node);
      ts::read(is, m_total_weight_count);
      ts::read_raw(is, m_leaf_weights);
   }

protected:
   void resize_weights(unsigned int variable_count, size_t node_count)
   {
      m_total_weight_count = m_weights_per_node * node_count;
      size_t leaf_count = (node_count + 1) >> 1;
      m_leaf_weights.resize(leaf_count * m_weights_per_node, 0);
   }

   unsigned int m_weights_per_node;
   size_t m_total_weight_count;
   std::vector<weight_t> m_leaf_weights;
};

template <  typename _DataTraits, 
            unsigned int _variable_count,
            typename _TFeature>
class factor : public dtf_factor_base<_DataTraits>
{
public:
   typedef _DataTraits data_traits;
   typedef _TFeature feature_type;
   typedef typename std::remove_reference<decltype(_TFeature::pre_process(input_type()))>::type pre_process_type;
   typedef decltype(samples_type().begin()) sample_iterator;
   typedef ts::binary_tree_array<_TFeature> tree_type;

   static const label_t label_count = _DataTraits::label_count;
   static const unsigned int variable_count = _variable_count;

   factor()
   {
      m_weights_per_node = ipow(static_cast<unsigned int>(label_count), variable_count);
      memset(&m_offsets[0], 0, sizeof(m_offsets));
      resize_weights(variable_count, 1);
   }
   template <typename _Offsets> factor(_Offsets offsets)
   {
      m_weights_per_node = ipow(static_cast<unsigned int>(label_count), variable_count);
      std::copy(offsets, offsets + m_offsets.size(), std::begin(m_offsets));
      resize_weights(variable_count, 1);
   }
   void set_tree(const tree_type& tree)
   {
      m_tree = tree;
      resize_weights(variable_count, tree.get_node_count());
   }
   const tree_type& get_tree() const 
   { 
      return m_tree; 
   }
   void set_variable(unsigned int index, int x, int y)
   {
      if (index >= variable_count)
         throw std::invalid_argument("Variable index too large");
      m_offsets[index].x = x;
      m_offsets[index].y = y;
   }
   unsigned int get_variable_count() const
   {
      return variable_count;
   }
   const std::array<offset_t, variable_count>& get_variables() const
   {
      return m_offsets;
   }
   offset_t operator[](unsigned int index) const
   {
      return m_offsets[index];
   }
   offset_t& operator[](unsigned int index)
   {
      return m_offsets[index];
   }
   rect<int> bounding_box() const
   {
      rect<int> r;
      for each (auto offset in m_offsets)
         r |= offset;
      return r;
   }
   rect<int> bbox_variables(unsigned int cx, unsigned int cy) const
   {
      rect<int> gtrect(0, 0, cx, cy);
      rect<int> box;
       for each (offset_t offset in m_offsets)
            box |= offset;
      return gtrect.deflate_rect(box);
   }
   size_t leaf_index(pos_t xy, const pre_process_type& prep) const
   {
      return m_tree.get_leaf_index([&](const feature_type& feature)
      {
         return feature(xy.x, xy.y, prep, m_offsets.begin());
      });
   }

   virtual unsigned int variables() const override 
   {
      return _variable_count;
   }
   virtual offset_t offset(unsigned int var) const override
   {
      return m_offsets[var];
   }
   // Here we can dispatch methods for the correct compile-time types...
   virtual numeric_t energy(const labelling_type& labelling, const input_type& input) const override
   {
      return compute::Energy(*this, labelling, input);
   }

   virtual void accumulate_unary_energies(const input_type& input, ts::image<std::array<numeric_t, label_count>>& acc) const override
   {
      if (_variable_count == 1)
         compute::AccumulateUnaries(*this, input, acc);
   }

   virtual ts::image<path_t> accumulate_variables(
                                 const input_type& input,
                                 const labelling_type& gt,
                                 const samples_type& samples,
                                 std::vector<weight_t>& var_acc,
                                 weight_cit weights) const override
   {
      return compute::AccumulateVariables(*this, input, gt, samples, var_acc, weights);
   }

   virtual void accumulate_gradients(const ts::image<path_t>& paths,
                                       const labelling_type& gt,
                                       const samples_type& samples,
                                       const std::vector<weight_t>& var_acc,
                                       ts::thread_locals<grad_acc_it>& grads) const override
   {
      compute::AccumulateGradients(*this, paths, gt, samples, var_acc, grads);
   }

   virtual void visit_instances(const input_type& input, std::function<void(pos_cit, pos_cit, weight_cit, weight_cit)> func) const override
   {
      compute::VisitInstances(*this, input, func);
   }

   virtual size_t accumulate_weights(weight_cit wbegin, weight_cit wend) override
   {
      compute::SumWeights(*this, wbegin, wend);
      return std::min<size_t>(wend - wbegin, m_total_weight_count);
   }

   virtual void accumulate_energies(const std::vector<offset_t>& samples,
                                    const ts::image<size_t>& leaves,
                                    const ts::image<label_t>& labelling,
                                    ts::image<std::array<weight_acc_t, label_count>>& energies,
                                    const ts::image<bool>& flags) const override
   {
      compute::AccumulateEnergies(*this, samples, leaves, labelling, energies, flags);
   }

   virtual void accumulate_energies(const std::vector<offset_t>& samples,
                                    const ts::image<size_t>& leaves,
                                    const ts::image<label_t>& labelling,
                                    ts::image<std::array<weight_acc_t, label_count>>& energies) const override
   {
      compute::AccumulateEnergies(*this, samples, leaves, labelling, energies);
   }

   virtual void accumulate_energies(const std::vector<offset_t>& samples,
                                    const ts::image<size_t>& leaves,
                                    ts::image<std::array<weight_acc_t, label_count>>& q_field) const override
   {
      compute::AccumulateMeanFieldEnergies(*this, samples, leaves, q_field);
   }

   virtual numeric_t sum_energies(const ts::image<size_t>& leaves,
                                  const ts::image<std::array<weight_acc_t, label_count>>& q_field) const override
   {
      return compute::SumMeanFieldEnergies(*this, leaves, q_field);
   }

   virtual ts::image<size_t> leaf_image(const input_type& input) const override
   {
      return compute::LeafImage(*this, input);
   }

   virtual void save(std::ostream& os) const
   {
      dtf_factor_base<_DataTraits>::save(os);
      for (size_t i = 0; i < m_offsets.size(); ++i)
         ts::write(os, m_offsets[i]);
      m_tree.write(os);
   }

   virtual void load(std::istream& is)
   {
      dtf_factor_base<_DataTraits>::load(is);
      for (size_t i = 0; i < m_offsets.size(); ++i)
         ts::read(is, m_offsets[i]);
      m_tree.read(is);
   }

protected:
   std::array<offset_t, variable_count> m_offsets;
   tree_type m_tree;
};

template <typename _Traits>
class factor_visitor
{
public:
   virtual void visit_factor(const factor_base<_Traits>& factor) { }
   virtual void visit_dtf_factor(const dtf_factor_base<_Traits>& factor) { visit_factor(factor); }

   virtual void visit_factor(factor_base<_Traits>& factor) { }
   virtual void visit_dtf_factor(dtf_factor_base<_Traits>& factor) { visit_factor(factor); }
};

template <typename _DataTraits>
void factor_base<_DataTraits>::accept(factor_visitor<_DataTraits>& visitor)
{
   visitor.visit_factor(*this);
}

template <typename _DataTraits>
void factor_base<_DataTraits>::accept(factor_visitor<_DataTraits>& visitor) const
{
   visitor.visit_factor(*this);
}

template <typename _DataTraits, typename _Func>
inline void visit_dtf_factors(const factor_graph<_DataTraits>& graph, _Func func)
{
   class dtf_factor_visitor : public factor_visitor<_DataTraits>
   {
   public:
      dtf_factor_visitor(_Func func) : m_func(func) {}
      virtual void visit_dtf_factor(const dtf_factor_base<_DataTraits>& factor) override
      {
         m_func(factor);
      }
   protected:
      _Func m_func;
   } visitor(func);
   graph.accept(visitor);
}

template <typename _DataTraits, typename _Func>
inline void visit_dtf_factors(factor_graph<_DataTraits>& graph, _Func func)
{
   class dtf_factor_visitor : public factor_visitor<_DataTraits>
   {
   public:
      dtf_factor_visitor(_Func func) : m_func(func) {}
      virtual void visit_dtf_factor(dtf_factor_base<_DataTraits>& factor) override
      {
         m_func(factor);
      }
   protected:
      _Func m_func;
   } visitor(func);
   graph.accept(visitor);
}

// Return an array of variables connected to an origin, e.g. for typical pairwise graph, would return
// {(1, 0), (-1, 0), (0, 1), (0, -1) }
template <typename _DataTraits>
inline std::vector<offset_t> dtf_map_connected_variables(const factor_graph<_DataTraits>& graph)
{
   std::vector<offset_t> connections;
   visit_dtf_factors(graph, [&](const dtf_factor_base<_DataTraits>& factor)
   {
      for (unsigned int v = 0; v < factor.variables(); ++v)
      {
         offset_t connection = factor.offset(v) - factor.offset(0);
         offset_t pair[] = { connection, -connection };
         std::for_each(std::begin(pair), std::end(pair), [&](offset_t j)
         {
            if (j == offset_t(0, 0))
               return;
            for (auto i = std::begin(connections); i != std::end(connections); ++i)
               if (*i == j)
                  return;
            connections.push_back(j);
         });
      }
   });
   return connections;
}

// Return the total number of all DTF coefficients/weights used in the factor graph
template <typename _DataTraits>
inline size_t dtf_count_weights(const factor_graph<_DataTraits>& graph)
{
   size_t count = 0;
   visit_dtf_factors(graph, [&](const dtf_factor_base<_DataTraits>& factor) { count += factor.get_weight_count(); });
   return count;
}

template <typename _DataTraits>
inline size_t dtf_count_factors(const factor_graph<_DataTraits>& graph)
{
   size_t count = 0;
   visit_dtf_factors(graph, [&](const dtf_factor_base<_DataTraits>& factor) { count++; });
   return count;
}

template <typename _DataTraits>
inline void dtf_set_all_weights(factor_graph<_DataTraits>& graph, weight_cit wbegin, weight_cit wend)
{
   size_t pos = 0;
   visit_dtf_factors(graph, [&](dtf_factor_base<_DataTraits>& factor) 
   { 
      pos += factor.accumulate_weights(wbegin + pos, wend); 
   });
}

class dense_pixel_iterator
{
public:
   dense_pixel_iterator() : m_pos(0, 0), m_end(0, 0) {}
   dense_pixel_iterator(int i) : m_pos(0, 0), m_end(0, 0) {}
   dense_pixel_iterator(int cx, int cy, int x = 0, int y = 0) : 
      m_end(cx, cy), m_pos(x, y) {}
   bool operator !=(const dense_pixel_iterator& rhs) const
   {
      return m_pos.y != rhs.m_pos.y || m_pos.x != rhs.m_pos.x;
   }
   bool operator <(const dense_pixel_iterator& rhs) const
   {
      if (m_pos.y < rhs.m_pos.y)
         return true;
      else if (m_pos.y > rhs.m_pos.y)
         return false;
      else 
         return m_pos.x < rhs.m_pos.x;
   }
   bool operator >=(const dense_pixel_iterator& rhs) const
   {
      return !operator <(rhs);
   }
   dense_pixel_iterator& operator++()
   {
      if (++m_pos.x >= m_end.x)
      {
         m_pos.x = 0;
         m_pos.y++;
      }
      return *this;
   }
   offset_t operator*() const
   {
      return m_pos;
   }
   int operator-(const dense_pixel_iterator& rhs) const
   {
      return (m_pos.y - rhs.m_pos.y) * m_end.x + m_pos.x - rhs.m_pos.x;
   }
   dense_pixel_iterator operator +(int i) const
   {
      int yd = i / m_end.x;
      int xd = i - yd * m_end.x;
      int x2 = m_pos.x + xd;
      int y2 = m_pos.y + yd;
      if (x2 >= m_end.x)
      {
         x2 -= m_end.x;
         y2++;
      }
      return dense_pixel_iterator(m_end.x, m_end.y, x2, y2);
   }
   static dense_pixel_iterator begin(unsigned int cx, unsigned int cy) { return dense_pixel_iterator(cx, cy); }
   static dense_pixel_iterator end(unsigned int cx, unsigned int cy) { return dense_pixel_iterator(cx, cy, 0, cy); }

protected:
   offset_t m_pos;
   offset_t m_end;
};

typedef range<dense_pixel_iterator> dense_sampling;

template <typename _TImage>
dense_sampling all_pixels(const _TImage& im)
{
   dense_sampling rv = { dense_pixel_iterator::begin(im.width(), im.height()), 
                         dense_pixel_iterator::end(im.width(), im.height()) };
   return rv;
}

class sparse_pixel_iterator
{
public:
   typedef std::vector<offset_t> pixel_list_base;
   typedef std::shared_ptr<pixel_list_base> pixel_list;

   sparse_pixel_iterator() : m_idx(0) {}
   sparse_pixel_iterator(size_t i) : m_idx(i) {}
   sparse_pixel_iterator(size_t i, pixel_list ppos)
      : m_idx(i), m_ppos(ppos) {}

   bool operator !=(const sparse_pixel_iterator& rhs) const
   {
      return m_idx != rhs.m_idx;
   }
   bool operator <(const sparse_pixel_iterator& rhs) const
   {
      return m_idx < rhs.m_idx;
   }
   sparse_pixel_iterator& operator++()
   {
      ++m_idx;
      return *this;
   }
   offset_t operator*() const
   {
      return (*m_ppos)[m_idx];
   }
   int operator-(const sparse_pixel_iterator& rhs) const
   {
      return static_cast<int>(m_idx) - static_cast<int>(rhs.m_idx);
   }
   sparse_pixel_iterator operator +(int i) const
   {
      return sparse_pixel_iterator(m_idx + i, m_ppos);
   }
   static sparse_pixel_iterator begin(pixel_list ppos)
   {
      return sparse_pixel_iterator(0, ppos);
   }
   static sparse_pixel_iterator end(pixel_list ppos)
   {
      return sparse_pixel_iterator(ppos->size(), ppos);
   }
protected:
   pixel_list m_ppos;
   size_t m_idx;
};

typedef range<sparse_pixel_iterator> sparse_sampling;

template <typename _TImage>
sparse_sampling sparse_subset_pixels(const _TImage& im, double fraction, unsigned long seed)
{
   unsigned int width = im.width();
   unsigned int height = im.height();

   std::mt19937 eng;
   eng.seed(seed);
   std::uniform_int_distribution<unsigned int> rint(0, width * height - 1);

   size_t count = static_cast<size_t>(fraction * width * height + 0.5);
   std::vector<unsigned int> pp_abs(count);
   for (size_t ci = 0; ci < count; ++ci)
      pp_abs[ci] = rint(eng);
   std::sort(pp_abs.begin(), pp_abs.end());

   // Translate into ordered x/y coordinates
   sparse_pixel_iterator::pixel_list ppos = std::make_shared<sparse_pixel_iterator::pixel_list_base>(count);
   for (size_t ci = 0; ci < count; ++ci)
      (*ppos)[ci] = offset_t(pp_abs[ci] % width, pp_abs[ci] / width);

   sparse_sampling rv = { sparse_pixel_iterator::begin(ppos), sparse_pixel_iterator::end(ppos) };
   return rv;
}

// Robust exponentiation: energies -> probabilities
//    p[i] = exp(-e[i] * s) / sum_i(exp(-e[i] * s))
// Return: log sum exp(-e[i] * s)
template <typename _EIt, typename _PIt>
double robust_exp_norm(_EIt energy_begin, _EIt energy_end, _PIt prob_begin, double invT = 1.0)
{
   typedef std::remove_reference<decltype(*prob_begin)>::type P;
   double sumexp = 0;
   double xmin = *std::min_element(energy_begin, energy_end);
   for (_EIt i = energy_begin; i != energy_end; ++i)
      sumexp += std::exp((xmin - *i) * invT);
   double lambda = xmin * invT - std::log(sumexp);
   _PIt p = prob_begin;
   for (_EIt i = energy_begin; i != energy_end; ++i, ++p)
      *p = static_cast<P>(std::exp(lambda - invT * *i));
   return -lambda;
}

}

#include "dtf_compute.h"
#include "dtf_learning.h"
#include "dtf_training.h"
#include "dtf_classify.h"
#include "dtf_inference.h"
