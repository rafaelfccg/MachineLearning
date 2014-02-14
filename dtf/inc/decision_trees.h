#pragma once

/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 23 August 2011
*/

#include <vector>
#include <ios>
#include <stack>
#include <list>
#include <queue>
#include "serialize.h"

namespace ts
{

template <typename T>
class binary_tree_array
{
public:
   typedef __int32 iterator;
   typedef __int32 const_iterator;
   typedef iterator child_type;

   struct split_node
   {
      split_node() : left(0), right(0) {}
      split_node(const T& t) : left(0), right(0), data(t) {}

      child_type left;
      child_type right;
      T data;
   };

   struct node_info
   { 
      iterator it; 
      unsigned int depth;
      unsigned int path;
      iterator parent; 
      bool direction_from_parent;
   };

   binary_tree_array() {}
   binary_tree_array(size_t size) : m_nodes(size) {}
   binary_tree_array(const binary_tree_array<T>& rhs) : m_nodes(rhs.m_nodes) {}
   binary_tree_array(binary_tree_array<T>&& rhs)
      : m_nodes(std::move(rhs.m_nodes)) {}

   static bool is_leaf(iterator node) { return node < 0; }
   static size_t child_to_leaf(child_type child) { return static_cast<size_t>(-child - 1); }
   static child_type leaf_to_child(size_t leaf) { return -static_cast<child_type>(leaf + 1); }
   
   // Returns the index of a node even if node is a leaf
   size_t get_index(iterator node) const 
   { 
      return is_leaf(node) ? size() + child_to_leaf(node) : node; 
   }
   size_t get_node_count() const
   {
      return (get_split_count() << 1) + 1;
   }
   iterator node_index_to_iterator(size_t index) const
   {
      return index >= size() ? leaf_to_child(index - size()) : static_cast<iterator>(index);
   }
   size_t get_split_count() const
   {
      return m_nodes.size();
   }
   void resize(size_t nodeCount)
   { 
      m_nodes.resize(nodeCount / 2); 
   }

   size_t size() const { return m_nodes.size(); }
   bool empty() const { return size() == 0; }
   void clear() { m_nodes.clear(); }

   iterator root() { return iterator(empty() ? -1 : 0); }
   const_iterator root() const { return const_iterator(empty() ? -1 : 0); }

   iterator push_back(const T& data)
   {
      split_node node(data);
      node.left = node.right = -1;
      m_nodes.push_back(node);
      return static_cast<int>(m_nodes.size()) - 1;
   }

   const_iterator get_split_child(const_iterator split, bool right) const
   {
      const split_node& parent = m_nodes[split];
      return right ? parent.right : parent.left;
   }

   template <typename TTest>
   size_t get_leaf_index(const TTest& test_function) const
   {
      const_iterator i = root();
      while (!is_leaf(i))
         i = get_split_child(i, test_function(get_split_data(i)));
      return child_to_leaf(i);
   }

   const T& get_split_data(const_iterator split) const
   {
      return m_nodes[split].data;
   }

   T& split_data(const_iterator split)
   {
      return m_nodes[split].data;
   }

   void set_split_data(const_iterator split, const T& data)
   {
      m_nodes[split].data = data;
   }

   void set_split_child(const_iterator split, bool right, child_type child)
   {
      (right ? m_nodes[split].right : m_nodes[split].left) = child;
   }

   void set_split_children(const_iterator split, child_type left_child, child_type right_child)
   {
      set_split_child(split, false, left_child);
      set_split_child(split, true, right_child);
   }

   void set_split_leaves(const_iterator split, size_t left_leaf, size_t right_leaf)
   {
      set_split_child(split, false, leaf_to_child(left_leaf));
      set_split_child(split, true, leaf_to_child(right_leaf));
   }

   std::ostream& write(std::ostream& stream) const
   {
      return ts::write_raw(stream, m_nodes);
   }
   std::istream& read(std::istream& stream) 
   {
      return ts::read_raw(stream, m_nodes);
   }
   friend std::ostream& operator <<(std::ostream& os, const binary_tree_array& tree)
   {
      return tree.write(os);
   }
   friend std::istream& operator >>(std::istream& is, binary_tree_array& tree)
   {
      return tree.read(is);
   }

protected:
   size_t index(iterator i) const { return i; }

   std::vector<split_node> m_nodes;
};

template <typename TFeature, size_t energy_count, typename THist = unsigned int>
class decision_tree
{
public:
   typedef std::array<THist, energy_count> distribution_t;
   typedef typename binary_tree_array<TFeature>::iterator iterator;
   typedef TFeature feature_t;
   typedef THist count_t;

   decision_tree() {}
   decision_tree(const decision_tree& rhs) : m_tree(rhs.m_tree), m_leaves(rhs.m_leaves) {}
   decision_tree(decision_tree&& rhs) : m_tree(std::move(rhs.m_tree)), m_leaves(std::move(rhs.m_leaves)) {}
   decision_tree(const binary_tree_array<TFeature>& tree, const std::vector<distribution_t>& leaves) : m_tree(tree), m_leaves(leaves) {}
   decision_tree(binary_tree_array<TFeature>&& tree, std::vector<distribution_t>&& leaves) : m_tree(tree), m_leaves(leaves) {}
  
   template <typename _Func>
   const distribution_t& get_leaf_distribution(_Func func) const
   {
      tree_t::const_iterator i = m_tree.root();
      while (!m_tree.is_leaf(i))
         i = m_tree.get_split_child(i, func(m_tree.get_split_data(i)));
      return m_leaves[m_tree.child_to_leaf(i)];
   }

   const binary_tree_array<TFeature>& tree() const
   {
      return m_tree;
   }
   
   void split_leaf_node(iterator parent, bool direction, const TFeature& feature,
                        const distribution_t& left, const distribution_t& right);

   std::ostream& write(std::ostream& stream) const
   {
      m_tree.write(stream);
      return ts::write_raw(stream, m_leaves);
   }
   std::istream& read(std::istream& stream)
   {
      m_tree.read(stream);
      return ts::read_raw(stream, m_leaves);
   }
   friend std::ostream& operator <<(std::ostream& os, const decision_tree& tree)
   {
      return tree.write(os);
   }
   friend std::istream& operator >>(std::istream& is, decision_tree& tree)
   {
      return tree.read(is);
   }
protected:
   typedef binary_tree_array<TFeature> tree_t;
   tree_t m_tree;
   std::vector<distribution_t> m_leaves;
};

template <typename TFeature, typename TLeaf>
class decision_forest
{
public:
   typedef binary_tree_array<TFeature> Tree;
   typedef typename Tree::child_type child_type;
   typedef typename Tree::iterator iterator;
   typedef typename Tree::const_iterator const_iterator;

   void clear()
   {
      m_trees.clear();
      m_leaves.clear();
   }

   size_t size() const { return m_trees.size(); }
   const Tree& operator[](size_t index) const { return m_trees[index]; }

   void resize(size_t tree_count) { m_trees.resize(tree_count); }
   Tree& operator[](size_t index) { return m_trees[index]; }
   std::vector<TLeaf>& get_leaves() { return m_leaves; }

   const TLeaf& get_leaf_data(child_type child) const
   {
      return m_leaves[Tree::child_to_leaf(child)];
   }
   
   template <typename TTest>
   const TLeaf& get_leaf(size_t tree_index, const TTest& test_function) const
   {
      const binary_tree_array<TFeature>& tree = m_trees[tree_index];
      const_iterator i = tree.head();
      while (!is_leaf(i))
         i = tree.get_split_child(i, test_function(tree.get_split_data(i)));
      return m_leaves[child_to_leaf(i)];
   }

   std::ostream& write(std::ostream& stream) const
   {
      ts::write<unsigned int>(stream, 101u);
      ts::write<unsigned int>(stream, static_cast<unsigned int>(m_trees.size()));
      for (size_t i = 0; i < m_trees.size(); i++)
         m_trees[i].write(stream);

      return write_raw(stream, m_leaves);
   }

   std::istream& read(std::istream& stream)
   {
      ts::read<unsigned int>(stream);
      unsigned int treeCount = ts::read<unsigned int>(stream);
      m_trees.resize(treeCount);
      for (unsigned int i = 0; i < treeCount; i++)
         m_trees[i].read(stream);

      return read_raw(stream, m_leaves);
   }

protected:
   std::vector<binary_tree_array<TFeature>> m_trees;
   std::vector<TLeaf> m_leaves;
};

template <typename T>
class breadth_first_iterator
{
public:
   typedef typename binary_tree_array<T>::node_info node_info;
   breadth_first_iterator(const binary_tree_array<T>& tree) : m_tree(tree) 
   {
      node_info root = { m_tree.root(), 0, 0, -1, false };
      m_nodes.push(root);
   }
   breadth_first_iterator& operator++()
   {
      if (!m_nodes.empty())
      {
         node_info parent = m_nodes.front();
         m_nodes.pop();
         if (!m_tree.is_leaf(parent.it))
         {
            for (int i = 0; i < 2; i++)
            {
               node_info child = { m_tree.get_split_child(parent.it, i > 0), 
                              parent.depth + 1,
                              parent.path | (i << parent.depth),
                              parent.it,
                              i > 0 };
               m_nodes.push(child);
            }
         }
      }
      return *this;
   }
   const node_info* operator ->() const
   {
      return &m_nodes.front();
   }
   const node_info& operator *() const
   {
      return m_nodes.front();
   }
   operator bool() const
   {
      return !m_nodes.empty();
   }
protected:
   const binary_tree_array<T>& m_tree;
   std::queue<node_info, std::list<node_info>> m_nodes;
};
//
//template <typename T, typename TLeaf>
//class breadth_first_leaf_iterator
//{
//public:
//   breadth_first_leaf_iterator& operator++()
//   {
//      while (m_it && !binary_tree_array<T>::is_leaf(*m_it.it))
//         ++m_it;
//   }
//   const TLeaf& operator*() const
//   {
//      return 
//protected:
//};

template <typename T>
binary_tree_array<T> tree_order_by_breadth(const binary_tree_array<T>& rhs)
{
   binary_tree_array<T> rv(rhs.size());
   if (rhs.empty())
      return rv;

   typedef typename binary_tree_array<T>::iterator iterator;
   struct Parent
   {
      iterator dfi;
      iterator bfiParent;
      unsigned char child;
   };
   std::queue<Parent, std::list<Parent>> parent;
   Parent as_parent = { rhs.root(), -1 };
   parent.push(as_parent);
   binary_tree_array<T>::iterator bfi = rv.root();

   while (!parent.empty())
   {
      Parent node = parent.front();
      parent.pop();
          
      if (rhs.is_leaf(node.dfi))
      {
         rv.set_split_child(node.bfiParent, node.child > 0, node.dfi);
      }
      else
      {
         if (node.bfiParent >= 0)
            rv.set_split_child(node.bfiParent, node.child > 0, bfi);
         rv.set_split_data(bfi, rhs.get_split_data(node.dfi));
         for (int i = 0; i < 2; i++)
         {
            Parent p = { rhs.get_split_child(node.dfi, i > 0), bfi, i };
            parent.push(p);
         }
         bfi++;
      }
   }
   return rv;
}

template <typename T>
class tree_builder_by_depth
{
public:
   tree_builder_by_depth() {}
   void add_split_node(const T& data)
   {
      binary_tree_array<T>::iterator node = m_tree.push_back(data);
      if (!m_parent.empty())
      {
         m_tree.set_split_child(m_parent.top().node, m_parent.top().children > 0, node);
         m_parent.top().children++;
      }
      // Push this node index onto parent stack
      Parent as_parent = {node, 0};
      m_parent.push(as_parent);
   }
   void add_leaf_node(size_t leaf_index)
   {
      m_tree.set_split_child(m_parent.top().node, m_parent.top().children > 0, m_tree.leaf_to_child(leaf_index));
      m_parent.top().children++;
      while (!m_parent.empty() && m_parent.top().children > 1)
         m_parent.pop();
   }
   bool complete() const
   {
      return m_parent.empty() && !m_tree.empty();
   }

   binary_tree_array<T> breadth_first() const
   {
      return tree_order_by_breadth(m_tree);
   }

protected:
   binary_tree_array<T> m_tree;
   struct Parent { typename binary_tree_array<typename T>::iterator node; unsigned char children; };
   std::stack<Parent> m_parent;
};


}