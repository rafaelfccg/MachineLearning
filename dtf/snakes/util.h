#pragma once

#include "imageio.h"
#include "dtf.h"

/*
   TODO:

   In core library, replace database with database iterator: we never use random access.
*/

template <unsigned char _LabelCount, typename _Pixel>
class CachedDatabase
{
public:
   static const dtf::label_t label_count = _LabelCount;

   CachedDatabase(const std::wstring& paths_input, const std::wstring& paths_gt, unsigned int start, unsigned int stop) : 
               m_paths_input(paths_input), m_paths_gt(paths_gt), m_cache(stop - start), m_start(start), m_stop(stop) {}

   size_t size() const { return m_stop - m_start; }

   ts::image<const _Pixel> get_training(size_t i) const
   {
      if (m_cache[i].input.empty())
         m_cache[i].input = ts::load<_Pixel>(format(m_paths_input, m_start + i));
      return m_cache[i].input;
   }

   ts::image<const unsigned char> get_ground_truth(size_t i) const
   {
      if (m_cache[i].gt.empty())
         m_cache[i].gt = ts::load<unsigned char>(format(m_paths_gt, m_start + i));
      return m_cache[i].gt;
   }

   dtf::dense_sampling get_samples(size_t i) const
   {
      return dtf::all_pixels(get_ground_truth(i));
   }

protected:
   static std::wstring format(const std::wstring& path, size_t index)
   {
      std::wstring str(MAX_PATH, L'\0');
      int size = wsprintf(&str[0], path.c_str(), index);
      str.resize(size);
      return str;
   }

   struct element
   {
      ts::image<_Pixel> input;
      ts::image<unsigned char> gt;
   };
   std::wstring m_paths_input, m_paths_gt;
   mutable std::vector<element> m_cache;
   unsigned int m_start, m_stop;
};

template <typename T> 
inline const T& clamp(const T& x, const T& a, const T& b) 
{ 
   return std::min(std::max(a, x), b); 
}

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