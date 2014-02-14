#pragma once

/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 1 July 2011
   
*/

#include <vector>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <cassert>
#include "parallel_for.h"

#pragma push_macro("TS_PIXEL_ITERATOR_CHECKS")

#ifndef NDEBUG
#define TS_PIXEL_ITERATOR_CHECKS
#endif

namespace ts
{

template <typename _Ty> class image; // Forward declaration

// Apply the const-ness of type T1 to the type T2.
template <typename T1, typename T2> struct apply_const { typedef typename std::remove_const<T2>::type type; };
template <typename T1, typename T2> struct apply_const<const T1, T2> { typedef const T2 type; };

template <typename _Ty>
class pixel_iterator
{
public:
   typedef typename std::random_access_iterator_tag iterator_category;
   typedef ptrdiff_t difference_type;
   typedef _Ty& reference;
   typedef _Ty* pointer;
   typedef typename std::remove_const<_Ty>::type value_type;

   pixel_iterator() : m_p(nullptr)
#ifdef TS_PIXEL_ITERATOR_CHECKS
      , m_startp(nullptr), m_endp(nullptr)
#endif
   {
   }

   reference operator *() const 
   { 
      check(m_p);
      return *m_p; 
   }
   pointer operator ->() const 
   { 
      check(m_p);
      return m_p; 
   }
   reference operator[](int x) const 
   { 
      check(m_p + x);
      return m_p[x]; 
   }
   bool operator <(const pixel_iterator<_Ty>& rhs) const
   {
      return m_p < rhs.m_p;
   }
   bool operator !=(const pixel_iterator<_Ty>& rhs) const
   {
      return m_p != rhs.m_p;
   }
   pixel_iterator<_Ty>& operator++()
   {
      ++m_p;
      return *this;
   }
   pixel_iterator<_Ty> operator++(int)
   {
      return spawn(m_p + 1);
   }
   pixel_iterator<_Ty> operator +(difference_type x) const
   {
      return spawn(m_p + x);
   }

   // Convert from "const pixel_iterator<T>&" to "const pixel_iterator<const T>&"
   operator typename const pixel_iterator<typename std::add_const<_Ty>::type>&() const
   {
      return reinterpret_cast<const pixel_iterator<typename std::add_const<_Ty>::type>&>(*this);
   }
protected:
   friend class image<typename std::remove_const<value_type>::type>;
   friend class image<const value_type>;
#ifdef TS_PIXEL_ITERATOR_CHECKS
   pixel_iterator(pointer startp, pointer p, pointer endp) : m_startp(startp), m_p(p), m_endp(endp) {}
#else
   pixel_iterator(pointer p) : m_p(p) {}
#endif
   pixel_iterator<_Ty> spawn(pointer p) const
   {
#ifdef TS_PIXEL_ITERATOR_CHECKS
      return pixel_iterator<_Ty>(m_startp, p, m_endp);
#else
      return pixel_iterator<_Ty>(p);
#endif
   }
   void check(pointer p) const
   {
#ifdef TS_PIXEL_ITERATOR_CHECKS
      // Check for valid pointer in debug mode
      assert(p >= m_startp && p < m_endp);
#endif
   }

   pointer m_p;
#ifdef TS_PIXEL_ITERATOR_CHECKS
   pointer m_startp, m_endp;
#endif
};

// TODO: Try making rowAlignBytes a template parameter

template <typename _Ty>
class image
{
public:
   typedef typename pixel_iterator<_Ty> iterator;
   typedef typename pixel_iterator<const _Ty> const_iterator;

   // Default constructor
   image() : m_width(0), m_height(0), m_p(nullptr), m_scanlineBytes(0), m_rowAlignBytes(1) {}

   // Constructor allocating image data
   image(unsigned int width, unsigned int height, unsigned char rowAlignBytes = 1) : 
      m_width(width), m_height(height), m_rowAlignBytes(rowAlignBytes),
      m_scanlineBytes((width * sizeof(_Ty) + rowAlignBytes - 1) & ~(rowAlignBytes - 1)),
      m_data((size_t)m_scanlineBytes * height + rowAlignBytes),
      m_p(&m_data[0] + m_rowAlignBytes - ((size_t)(&m_data[0]) & (m_rowAlignBytes - 1))) {}

   // Constructor that wraps existing image data
   image(unsigned int width, unsigned int height, _Ty* bits, int scanlineBytes) :
      m_width(width), m_height(height), m_rowAlignBytes(1), m_scanlineBytes(scanlineBytes), m_p(reinterpret_cast<decltype(m_p)>(bits)) {}

   // Copy constructor
   image(const image<_Ty>& rhs) : 
      m_width(rhs.m_width), m_height(rhs.m_height), m_scanlineBytes(rhs.m_scanlineBytes), m_rowAlignBytes(rhs.m_rowAlignBytes),
      m_data((size_t)m_scanlineBytes * m_height + m_rowAlignBytes),
      m_p(&m_data[0] + m_rowAlignBytes - ((size_t)(&m_data[0]) & (m_rowAlignBytes - 1)))
   {
      if (m_p != nullptr)
         memcpy(const_cast<unsigned char*>(m_p), rhs.m_p, image_bytes());
   }

   // Move constructor
   image(image&& rhs) : 
      m_width(rhs.m_width), m_height(rhs.m_height), m_p(rhs.m_p), m_data(std::move(rhs.m_data)), 
      m_scanlineBytes(rhs.m_scanlineBytes), m_rowAlignBytes(rhs.m_rowAlignBytes)
   {
      rhs.m_width = rhs.m_height = rhs.m_scanlineBytes = rhs.m_rowAlignBytes = 0;
      rhs.m_p = nullptr;
   }

   // Copy assignment operator
   image& operator =(const image& rhs)
   {
      if (this != &rhs)
      {
         resize(rhs.m_width, rhs.m_height, rhs.m_rowAlignBytes);
         if (m_p != nullptr)
         {
            if (image_bytes() == rhs.image_bytes())
               memcpy(m_p, rhs.m_p, image_bytes());
            else for (unsigned int y = 0; y < m_height; ++y)
               memcpy(Ptr(y), rhs.Ptr(y), m_width * pixel_bytes());
         }
      }
      return *this;
   }
  
   // Move assignment operator
   image& operator =(image&& rhs)
   {
      if (this != &rhs)
      {
         m_data = std::move(rhs.m_data);
         m_width = rhs.m_width;
         m_height = rhs.m_height;
         m_p = rhs.m_p;
         m_scanlineBytes = rhs.m_scanlineBytes;
         m_rowAlignBytes = rhs.m_rowAlignBytes;
         rhs.m_width = rhs.m_height = rhs.m_scanlineBytes = rhs.m_rowAlignBytes = 0;
         rhs.m_p = nullptr;
      }
      return *this;
   }
   
   unsigned int width() const { return m_width; }
   unsigned int height() const { return m_height; }

   bool empty() const { return m_width == 0 || m_height == 0; }
   bool contains(unsigned int x, unsigned int y) const { return x < m_width && y < m_height; }

   static unsigned int pixel_bytes() { return sizeof(_Ty); }
   int stride_bytes() const { return m_scanlineBytes; }
   size_t image_bytes() const { return m_height * m_scanlineBytes; }
   unsigned int scanline_padding() const { return abs(stride_bytes()) - width() * pixel_bytes(); }

   _Ty& operator()(unsigned int x, unsigned int y) { return operator[](y)[x]; }
   const _Ty& operator()(unsigned int x, unsigned int y) const { return operator[](y)[x]; }

   iterator operator[](unsigned int y) { return begin_row(y); }
   const_iterator operator[](unsigned int y) const { return begin_row(y); }

   void clear()
   {
      m_data.clear();
      m_width = m_height = m_scanlineBytes = m_rowAlignBytes = 0;
      m_p = nullptr;
   }

   void resize(unsigned int width, unsigned int height, unsigned char rowAlignBytes = 1)
   {
      if (width != m_width || height != m_height || rowAlignBytes != m_rowAlignBytes)
      {
         image rhs(width, height, rowAlignBytes);
         unsigned int cx = std::min(width, m_width);
         unsigned int cy = std::min(height, m_height);
         for (unsigned int y = 0; y < cy; y++)
            memcpy(&rhs[y][0], &(*this)[y][0], cx * sizeof(_Ty));
         *this = std::move(rhs);
      }
   }

   // Convert from "const image<const T>&" to "const image<T>&"
   operator typename const image<typename std::remove_const<_Ty>::type>&() const
   {
      return reinterpret_cast<const image<typename std::remove_const<_Ty>::type>&>(*this);
   }
   
   // Convert from "const image<T>&" to "const image<const T>&"
   operator typename const image<typename std::add_const<_Ty>::type>&() const
   {
      return reinterpret_cast<const image<typename std::add_const<_Ty>::type>&>(*this);
   }
   
   const_iterator begin() const
   {
      if (scanline_padding() != 0)
         throw std::exception("Cannot use 1D iterator with scanline padding");
      return begin_row(0);
   }
   iterator begin()
   {
      if (scanline_padding() != 0)
         throw std::exception("Cannot use 1D iterator with scanline padding");
      return begin_row(0);
   }

   const_iterator end() const
   {
      return begin_row(height());
   }

   iterator end()
   {
      return begin_row(height());
   }

protected:
   const_iterator begin_row(unsigned int y) const 
   { 
#ifdef TS_PIXEL_ITERATOR_CHECKS
      return const_iterator(Ptr(0), Ptr(y), Ptr(m_height)); 
#else
      return const_iterator(Ptr(y)); 
#endif
   }
   iterator begin_row(unsigned int y) 
   { 
#ifdef TS_PIXEL_ITERATOR_CHECKS
      return iterator(Ptr(0), Ptr(y), Ptr(m_height)); 
#else
      return iterator(Ptr(y)); 
#endif
   }
   iterator end_row(unsigned int y)
   {
      return begin_row(y) + m_width;
   }

   _Ty* Ptr(unsigned int y) { return reinterpret_cast<_Ty*>(BytePtr(y)); }
   const _Ty* Ptr(unsigned int y) const { return reinterpret_cast<const _Ty*>(BytePtr(y)); }
   unsigned char* BytePtr(unsigned int y) { return m_p + y * m_scanlineBytes; }
   const unsigned char* BytePtr(unsigned int y) const { return m_p + y * m_scanlineBytes; }

   unsigned int m_width, m_height;
   unsigned char m_rowAlignBytes;
   unsigned int m_scanlineBytes;
   std::vector<unsigned char> m_data;
   typename apply_const<_Ty, unsigned char>::type* m_p;
};
}

#pragma pop_macro("TS_PIXEL_ITERATOR_CHECKS")

#include <array>

namespace ts
{
typedef image<unsigned char> image_u8;
typedef std::array<unsigned char, 3> bgr24;
typedef std::array<unsigned char, 4> bgra32;
typedef image<bgr24> image_u8_x3;
typedef image_u8_x3 image_bgr24;
}

namespace ts
{
template <typename _Ty, typename _Func>
inline void for_each_pixel(const image<_Ty>& img, _Func func)
{
   // We need a 2D loop for efficiency
   for (unsigned int y = 0; y < img.height(); ++y)
   {
      image<_Ty>::const_iterator i = img[y];
      for (unsigned int x = 0; x < img.width(); ++x, ++i)
         func(*i);
   }
}

template <typename T, typename _Func>
inline void for_each_2d(T width, T height, _Func func)
{
   for (T y = 0; y < height; y++)
   {
      for (T x = 0; x < width; x++)
         func(x, y);
   }
}

template <typename _Ty, typename _Func>
inline void for_each_pixel_xy(const image<_Ty>& img, _Func func)
{
   for_each_2d(img.width(), img.height(), func);
}

template <typename _Func>
inline void parallel_for_each_2d(unsigned int width, unsigned int height, _Func func)
{
   ts::parallel_for(0u, width * height, [&](unsigned int xy)
   {
      unsigned int y = xy / width;
      unsigned int x = xy - y * width;
      func(x, y);
   });
}

template <typename _Timg, typename _Func>
inline void parallel_for_each_pixel(_Timg& img, _Func func)
{
   ts::parallel_for(0u, img.height(), [&](unsigned int y)
   {
      auto i = img[y];
      for (unsigned int x = 0; x < img.width(); ++x, ++i)
         func(*i);
   });
}

template <typename _Timg, typename _Func>
inline void parallel_for_each_pixel_xy(const _Timg& img, _Func func)
{
   parallel_for_each_2d(img.width(), img.height(), func);
}

template <typename _Ty, typename _Func>
inline auto pointwise(const image<_Ty>& img, _Func func) -> image<decltype(func(_Ty()))>
{
   typedef decltype(func(_Ty())) T2;
   image<T2> img2(img.width(), img.height());
   for (unsigned int y = 0; y < img.height(); ++y)
   {
      image<_Ty>::const_iterator i = img[y];
      image<T2>::iterator j = img2[y];
      for (unsigned int x = 0; x < img.width(); ++x, ++i, ++j)
         *j = func(*i);
   }
   return img2;
}

template <typename _Ty, typename _T2, typename _Func>
inline auto pointwise(const image<_Ty>& img, const image<_T2>& im2, _Func func) -> image<decltype(func(_Ty(), _T2()))>
{
   typedef decltype(func(_Ty(), _T2())) T3;
   const unsigned int cx = std::min(img.width(), im2.width());
   const unsigned int cy = std::min(img.height(), im2.height());
   image<T3> rv(cx, cy);
   for (unsigned int y = 0; y < cy; ++y)
   {
      image<_Ty>::const_iterator i = img[y];
      image<_T2>::const_iterator j = im2[y];
      image<T3>::iterator k = rv[y];
      for (unsigned int x = 0; x < cx; ++x, ++i, ++j, ++k)
         *k = func(*i, *j);
   }
   return rv;
}

template <typename _Ty, typename _Func>
inline auto pointwise_xy(const image<_Ty>& img, _Func func) -> image<decltype(func(0, 0))>
{
   typedef decltype(func(0, 0)) T2;
   image<T2> img2(img.width(), img.height());
   for (unsigned int y = 0; y < img.height(); ++y)
   {
      image<T2>::iterator j = img2[y];
      for (unsigned int x = 0; x < img.width(); ++x, ++j)
         *j = func(x, y);
   }
   return img2;
}

template <typename _Ty, typename _Func>
inline auto parallel_pointwise(const image<_Ty>& img, _Func func) -> image<decltype(func(_Ty()))>
{
   typedef decltype(func(_Ty())) T2;
   image<T2> img2(img.width(), img.height());
   parallel_for(0u, img.height(), [&](unsigned int y)
   {
      image<_Ty>::const_iterator i = img[y];
      image<T2>::iterator j = img2[y];
      for (unsigned int x = 0; x < img.width(); ++x, ++i, ++j)
         *j = func(*i);
   });
   return img2;
}

template <typename _Ty, typename _T2, typename _Func>
inline auto parallel_pointwise(const image<_Ty>& img, const image<_T2>& im2, _Func func) -> image<decltype(func(_Ty(), _T2()))>
{
   typedef decltype(func(_Ty(), _T2())) T3;
   const unsigned int cx = std::min(img.width(), im2.width());
   const unsigned int cy = std::min(img.height(), im2.height());
   image<T3> rv(cx, cy);
   parallel_for(0u, cy, [&](unsigned int y)
   {
      image<_Ty>::const_iterator i = img[y];
      image<_T2>::const_iterator j = im2[y];
      image<T3>::iterator k = rv[y];
      for (unsigned int x = 0; x < cx; ++x, ++i, ++j, ++k)
         *k = func(*i, *j);
   });
   return rv;
}

template <typename _Ty, typename _T2, typename _Func>
inline void parallel_pointwise_inplace(image<_Ty>& dest_src1, const image<_T2>& src2, _Func func)
{
   const unsigned int cx = std::min(dest_src1.width(), src2.width());
   const unsigned int cy = std::min(dest_src1.height(), src2.height());
   parallel_for(0u, cy, [&](unsigned int y)
   {
      image<_Ty>::iterator i = dest_src1[y];
      image<_T2>::const_iterator j = src2[y];
      for (unsigned int x = 0; x < cx; ++x, ++i, ++j)
         func(*i, *j);
   });
}

template <typename _Ty, typename _Func>
inline auto parallel_pointwise_xy(const image<_Ty>& img, _Func func) -> image<decltype(func(0, 0))>
{
   typedef decltype(func(0, 0)) T2;
   image<T2> img2(img.width(), img.height());
   parallel_for(0u, img.height(), [&](unsigned int y)
   {
      image<T2>::iterator j = img2[y];
      for (unsigned int x = 0; x < img.width(); ++x, ++j)
         *j = func(x, y);
   });
   return img2;
}

template <typename _Func>
inline auto parallel_pointwise_xy(unsigned int width, unsigned int height, _Func func) -> image<decltype(func(0, 0))>
{
   typedef decltype(func(0, 0)) T2;
   image<T2> img2(width, height);
   parallel_for(0u, img2.height(), [&](unsigned int y)
   {
      image<T2>::iterator j = img2[y];
      for (unsigned int x = 0; x < img2.width(); ++x, ++j)
         *j = func(x, y);
   });
   return img2;
}


// Reference-counted shared images using std::shared_ptr
template <typename _Ty> struct image_ptr { typedef std::shared_ptr<ts::image<_Ty>> T; };
template <typename _Ty> struct image_ptr<const _Ty> { typedef std::shared_ptr<const ts::image<_Ty>> T; };

// Move an image<T> into a shared image_ptr<T>
template <typename T>
typename image_ptr<T>::T make_shared(ts::image<T>& img)
{
   return std::make_shared<ts::image<T>>(std::move(img));
}

// Move an image<const T> into a shared image_ptr<const T>.
template <typename T>
typename image_ptr<const T>::T make_shared(ts::image<const T>& img)
{
   return make_shared(reinterpret_cast<ts::image<T>&>(img));
}

}