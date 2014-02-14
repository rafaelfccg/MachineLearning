#pragma once

/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 4 July 2011

   Examples:

      for each (const auto& i in ts::images_from_path<unsigned short>(argv[1]))
         std::wcout << i.width() << ", " << i.height() << std::endl;

      for (ts::images_from_path<unsigned short>::iterator i(argv[1]); i; ++i)
         std::wcout << i.path() << ": " << (*i).width() << ", " << (*i).height() << std::endl;

*/
#include "image.h"
#include "dir.h"
#include "WicIo.h"

namespace ts
{

template <typename T>
image<T> load(const wchar_t* path)
{
   ImageLoader loader(path);
   if (loader.Bpp() != 8 * sizeof(T))
      throw std::exception("Pixel format does not match data type.");
   image<T> image(loader.Width(), loader.Height());
   loader.Write(&image(0, 0), image.stride_bytes(), (unsigned int)image.image_bytes());
   return image;
}

template <typename T>
image<T> load(const std::wstring& path)
{
   return load<T>(path.c_str());
}

template <typename _Timg>
void save(const _Timg& imgOut, const wchar_t* szOutFile, bool bIndexed = false, bool bHalftone = true)
{
   ImageWriter::Save(szOutFile, &imgOut(0, 0), imgOut.width(), imgOut.height(), imgOut.pixel_bytes() * 8,
      imgOut.stride_bytes(), bIndexed, bHalftone);
}

template <typename _Timg>
void save(const _Timg& imgOut, const std::wstring& path, bool bIndexed = false, bool bHalftone = true)
{
   save(imgOut, path.c_str(), bIndexed, bHalftone);
}

template <typename T>
class images_from_path
{
public:
   class iterator
   {
   public:
      iterator() : m_ignore_fail(false) {}
      iterator(const std::wstring& path, bool ignore_fail = false)
         : m_path(path), m_ignore_fail(ignore_fail)
      {
         advance();
      }
      ts::image<T>& operator *()
      {
         return m_image;
      }
      ts::image<T>* operator ->()
      {
         return &m_image;
      }
      iterator& operator++()
      {
         ++m_path;
         advance();
         return *this;
      }
      operator bool() const
      {
         return !m_image.empty();
      }
      bool operator !=(const iterator& rhs) const
      {
         return m_path != rhs.m_path;
      }
      const std::wstring& path() const
      {
         return *m_path;
      }
   protected:
      void advance()
      {
         m_image.clear();
         for (; m_path; ++m_path)
         {
            try
            {
               m_image = ts::load<T>(*m_path);
               break;
            }
            catch (std::exception& e)
            {
               if (!m_ignore_fail)
                  throw e;
            }
         }
      }

      ts::image<T> m_image;
      path_list::iterator m_path;
      bool m_ignore_fail;
   };

   images_from_path(const std::wstring& path, bool ignore_fail = false) : m_path(path), m_ignore_fail(ignore_fail) {}
   iterator begin() const { return iterator(m_path, m_ignore_fail); }
   iterator end() const { return iterator(); }
private:
   std::wstring m_path;
   bool m_ignore_fail;
};

template <typename T>
class images_from_path_sorted
{
public:
   class iterator
   {
   public:
      iterator() : m_ignore_fail(false), m_it(m_paths.end()) {}
      iterator(const std::wstring& match, bool ignore_fail = false)
         : m_ignore_fail(ignore_fail)
      {
         for each (const std::wstring& path in path_list(match))
            m_paths.push_back(path);
         m_it = m_paths.begin();
         advance();
      }
      ts::image<T>& operator *()
      {
         return m_image;
      }
      ts::image<T>* operator ->()
      {
         return &m_image;
      }
      iterator& operator++()
      {
         ++m_it;
         advance();
         return *this;
      }
      operator bool() const
      {
         return !m_image.empty();
      }
      bool operator !=(const iterator& rhs) const
      {
         if (m_it == m_paths.end() && rhs.m_it == rhs.m_paths.end())
            return false;
         else if (m_it == m_paths.end() || rhs.m_it == rhs.m_paths.end())
            return true;
         else
            return *m_it != *rhs.m_it;
      }
      const std::wstring& path() const
      {
         return *m_it;
      }
   protected:
      void advance()
      {
         m_image.clear();
         for (; m_it != m_paths.end(); ++m_it)
         {
            try
            {
               m_image = ts::load<T>(*m_it);
               break;
            }
            catch (std::exception& e)
            {
               if (!m_ignore_fail)
                  throw e;
            }
         }
      }

      ts::image<T> m_image;
      std::vector<std::wstring> m_paths;
      std::vector<std::wstring>::const_iterator m_it;
      bool m_ignore_fail;
   };

   images_from_path_sorted(const std::wstring& path, bool ignore_fail = false) : m_path(path), m_ignore_fail(ignore_fail) {}
   iterator begin() const { return iterator(m_path, m_ignore_fail); }
   iterator end() const { return iterator(); }
private:
   std::wstring m_path;
   bool m_ignore_fail;
};
}