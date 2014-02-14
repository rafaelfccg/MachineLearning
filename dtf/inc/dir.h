#pragma once

/*
   Copyright Microsoft Corporation 2011

   Author: Toby Sharp (tsharp)

   Date: 8 July 2011

   Simple utility for iterating over the paths matching a given string.

   Examples:

      for each (auto path in ts::path_list(L"*.png"))
         std::wcout << path << std::endl;

      for (ts::path_list::iterator i(L"*.png"); i; ++i)
         std::wcout << *i << std::endl;
*/

#include <string>
#include <sstream>

#include "safewin.h"

namespace ts
{

class path_list
{
public:
   class iterator
   {
   public:
      iterator() : m_hFind(INVALID_HANDLE_VALUE) {}
      iterator(const std::wstring& match) : m_hFind(INVALID_HANDLE_VALUE)
      {
         m_hFind = ::FindFirstFile(match.c_str(), &m_findData);
         if (m_hFind != INVALID_HANDLE_VALUE)
         {
            size_t lastSlash = match.find_last_of(L"/\\");
            if (lastSlash != match.npos)
               m_folder = match.substr(0, lastSlash + 1);
            m_path = m_folder + m_findData.cFileName;
         }
         else
         {
            DWORD dwError = ::GetLastError();
            if (dwError != ERROR_FILE_NOT_FOUND)
            {
               std::ostringstream s;
               s << __FUNCTION__ << ": FindFirstFile failed with GetLastError=" << dwError;
               throw std::exception(s.str().c_str());
            }
         }
      }
      ~iterator()
      {
         if (m_hFind != INVALID_HANDLE_VALUE)
            ::FindClose(m_hFind);
      }
      operator bool() const
      {
         return !m_path.empty();
      }
      iterator& operator++()
      {
         m_path.clear();
         if (::FindNextFile(m_hFind, &m_findData) == TRUE)
            m_path = m_folder + m_findData.cFileName;
         return *this;
      }
      const std::wstring& operator*() const
      {
         return m_path;
      }
      const std::wstring* operator ->() const
      {
         return &m_path;
      }
      bool operator !=(const iterator& rhs) const
      {
         return m_path != rhs.m_path;
      }
   private:
      HANDLE m_hFind;
      WIN32_FIND_DATA m_findData;
      std::wstring m_folder;
      std::wstring m_path;
   };

   path_list(const std::wstring& match) : m_match(match) {}
   iterator begin() const { return iterator(m_match); }
   iterator end() const { return iterator(); }

protected:
   std::wstring m_match;
};

}
