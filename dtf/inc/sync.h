#pragma once

/*
   Copyright Microsoft Corporation 2012

   Date: 6 February 2012
   
   Author: Toby Sharp (tsharp)
*/

#include <functional>
#include <memory>
#include "safewin.h"

namespace ts
{

class waitable
{
   friend class waitables;
public:
   virtual bool wait(unsigned int ms) const
   {
      return ::WaitForSingleObject(handle(), ms) == WAIT_OBJECT_0;
   }
   virtual bool wait() const
   {
      return wait(-1);
   }
   waitables operator +(const waitable& rhs) const;
protected:
   virtual ~waitable() {}
   virtual HANDLE handle() const = 0;
};

class waitables
{
public:
   waitables(const waitable& w)
   {
      m_vec.push_back(w.handle());
   }
   waitables(waitables&& w) : m_vec(w.m_vec) {}
   waitables operator +(const waitable& rhs) const
   {
      std::vector<HANDLE> vec(m_vec);
      vec.push_back(rhs.handle());
      return waitables(vec);
   }
   //waitables operator +(const waitables& rhs) const
   //{
   //   std::vector<HANDLE> vec(m_vec);
   //   std::for_each(rhs.m_vec.begin(), rhs.m_vec.end(), [&](const waitable& w)
   //      { vec.push_back(w.handle()); });
   //   return waitables(vec);
   //}
   int wait_any(unsigned int ms) const
   {
      DWORD size = static_cast<DWORD>(m_vec.size());
      DWORD dw = ::WaitForMultipleObjects(size, &m_vec[0], FALSE, ms);
      if (dw >= WAIT_OBJECT_0 && dw < WAIT_OBJECT_0 + size)
         return static_cast<int>(dw - WAIT_OBJECT_0);
      return -1;
   }
   int wait_any() const
   {
      return wait_any(-1);
   }
   int wait_all(unsigned int ms) const
   {
      DWORD size = static_cast<DWORD>(m_vec.size());
      DWORD dw = ::WaitForMultipleObjects(size, &m_vec[0], FALSE, ms);
      if (dw >= WAIT_OBJECT_0 && dw < WAIT_OBJECT_0 + size)
         return static_cast<int>(dw - WAIT_OBJECT_0);
      return -1;
   }
   int wait_all() const
   {
      return wait_all(-1);
   }
protected:
   waitables(const std::vector<HANDLE>& vec) : m_vec(vec) {}
   waitables(std::vector<HANDLE>&& vec) : m_vec(vec) {}

   std::vector<HANDLE> m_vec;
};

inline waitables waitable::operator +(const waitable& rhs) const
{
   return waitables(*this) + rhs;
}

class event : public waitable
{
public:
   event(bool manual = true, bool initial = false)
   {
      m_hEvent = ::CreateEvent(NULL, manual, initial, NULL);
      if (m_hEvent == NULL)
      {
         DWORD dw = ::GetLastError();
         throw std::exception("CreateEvent failed");
      }
   }
   ~event()
   {
      ::CloseHandle(m_hEvent);
   }
   void signal(bool signalled = true)
   {
      if (signalled)
         ::SetEvent(m_hEvent);
      else
         ::ResetEvent(m_hEvent);
   }
   bool wait(unsigned int ms) const
   {
      return ::WaitForSingleObject(m_hEvent, ms) == WAIT_OBJECT_0;
   }
   bool wait() const
   {
      return wait(-1);
   }
   bool is_signalled() const
   {
      return wait(0);
   }
protected:
   virtual HANDLE handle() const { return m_hEvent; }
   event(const event&);
   HANDLE m_hEvent;
};

class critical_section
{
public:
   critical_section()
   {
      ::InitializeCriticalSection(&m_cs);
   }
   ~critical_section()
   {
      ::DeleteCriticalSection(&m_cs);
   }
   void lock()
   {
      ::EnterCriticalSection(&m_cs);
   }
   void unlock()
   {
       ::LeaveCriticalSection(&m_cs);
   }
   bool try_lock()
   {
      return ::TryEnterCriticalSection(&m_cs) != 0;
   }
protected:
   critical_section(const critical_section& cs);
   CRITICAL_SECTION m_cs;
};

class lock
{
public:
    lock(critical_section& cs) : m_cs(cs)
    {
       m_cs.lock();
    }
    ~lock()
    {
       m_cs.unlock();
    }
protected:
    critical_section& m_cs;    
};

class try_lock
{
public:
    try_lock(critical_section& cs) : m_cs(cs)
    {
       m_b = m_cs.try_lock();
    }
    ~try_lock()
    {
       if (m_b)
          m_cs.unlock();
    }
    operator bool() const
    {
       return m_b;
    }
protected:
    critical_section& m_cs;
    bool m_b;
};

class semaphore : public waitable
{
public:
   semaphore() : m_hSemaphore(NULL) {}
   semaphore(unsigned int count)
   {
      create(count);
   }
   ~semaphore()
   {
      if (m_hSemaphore != NULL)
         ::CloseHandle(m_hSemaphore);
   }
   void create(unsigned int count)
   {
      if (m_hSemaphore != NULL)
      {
         ::CloseHandle(m_hSemaphore);
         m_hSemaphore = NULL;
      }
      m_hSemaphore = ::CreateSemaphore(NULL, 0, count, NULL);
      if (m_hSemaphore == NULL)
      {
         DWORD dw = ::GetLastError();
         throw std::exception("CreateSemaphore failed");
      }
   }
   void release(unsigned int count)
   {
      if (::ReleaseSemaphore(m_hSemaphore, count, NULL) == 0)
         throw std::exception("ReleaseSemaphore failed");
   }
protected:
   virtual HANDLE handle() const { return m_hSemaphore; }
   semaphore(const semaphore&);
   HANDLE m_hSemaphore;
};

class thread : public waitable
{
public:
   thread(std::function<int()> func) : m_func(new std::function<int()>(func))
   {
      m_hThread = ::CreateThread(NULL, 0, &ThreadProc, reinterpret_cast<void*>(m_func.get()), 0, NULL);
      if (m_hThread == NULL)
      {
         DWORD dw = ::GetLastError();
         throw std::exception("CreateThread failed");
      }
   }
   thread(const thread& rhs) : m_func(rhs.m_func)
   {
      if (::DuplicateHandle(::GetCurrentProcess(), rhs.m_hThread, ::GetCurrentProcess(), &m_hThread, 0, TRUE, DUPLICATE_SAME_ACCESS) == 0)
         throw std::exception("DuplicateHandle failed");
   }
   thread(thread&& rhs) : m_hThread(NULL)
   {
      std::swap(m_hThread, rhs.m_hThread);
      std::swap(m_func, rhs.m_func);
   }
   ~thread()
   {
      if (m_hThread != NULL)
         ::CloseHandle(m_hThread);
   }
   unsigned int id() const
   {
      return ::GetThreadId(m_hThread);
   }
protected:
   static DWORD WINAPI ThreadProc(void* pv)
   {
      std::function<int()>* p = reinterpret_cast<std::function<int()>*>(pv);
      return (*p)();
   }
   virtual HANDLE handle() const { return m_hThread; }
   HANDLE m_hThread;
   std::shared_ptr<std::function<int()>> m_func;
};

}
