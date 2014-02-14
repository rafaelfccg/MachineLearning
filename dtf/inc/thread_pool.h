#pragma once

/*
   Copyright Microsoft Corporation 2011

   Date: 16 September 2011
   
   Author: Toby Sharp (tsharp)
*/

#include "sync.h"
#include <vector>

namespace ts
{

// A class to allow for for loops to be distributed over multiple threads.
// This provides for the same functionality as using "#pragma omp parallel for schedule(dynamic)",
// but without the run-time dependency on OpenMP.
class thread_pool
{
public:
   static void ParallelFor(int start, int stop, std::function<void(int)> func)
   {
      s_instance.ParallelFor_(start, stop, func);
   }

   static unsigned int GetThreadIndex()
   {
      DWORD id = ::GetCurrentThreadId();
      for (size_t i = 0; i < s_instance.m_ids.size(); i++)
      {
         if (s_instance.m_ids[i] == id)
            return static_cast<unsigned int>(i);
      }
      // NOTE: This case arises during multiple concurrent or nested parallel-for loops.
      return 0;
   }

   static unsigned int GetThreadCount()
   {
      return static_cast<unsigned int>(s_instance.m_ids.size());
   }

protected:
   static thread_pool s_instance;

   struct Item
   {
      Item() : m_Done(false) {}

      long m_stop;
      volatile long m_current;
      volatile long m_remaining;
      std::function<void(int)> m_func;
      event m_Done;
   };

   void ParallelFor_(int start, int stop, std::function<void(int)> func)
   {
      // Only one thread at a time can be the "master" thread.
      ts::try_lock lock(m_cs);      
      if (lock)
      {
         // Set start and stop pos
         m_item.m_current = start;
         m_item.m_stop = stop;
         m_item.m_func = func;
         int threadsToAwake = static_cast<unsigned int>(m_threads.size());
         m_item.m_remaining = threadsToAwake + 1;
         m_ids[0] = ::GetCurrentThreadId();
         // Use semaphore or event to wake threads
         try
         {
            m_Semaphore.release(threadsToAwake);
         }
         catch (...)
         {
            m_item.m_remaining = 1;
         }
         // Call worker loop in the master thread too
         DoWork(m_item);
         // Wait for other threads to finish
         m_item.m_Done.wait();
      }
      else
      {
         // If this object is already busy, just get on with running the loop in this thread
         for (int i = start; i < stop; i++)
            func(i);
      }
   }

   thread_pool() : m_Abort(true)
   {
      int totalThreads = (int)GetLogicalProcessorCount();
      int workerThreads = totalThreads - 1;
      m_ids.push_back(0);
      try
      {
         m_Semaphore.create(workerThreads);
         for (int i = 0; i < workerThreads; i++)
         {
            std::function<int()> func = std::bind(&thread_pool::ThreadProc, this);
            ts::thread thread(func);
            m_threads.push_back(thread);
            m_ids.push_back(thread.id());
         }
      }
      catch (...)
      {
         // Semaphore creation failed
      }
   }

   ~thread_pool()
   {
      m_Abort.signal();
   }

   static void DoWork(Item& item) 
   {
      //::InterlockedIncrement(&item.m_remaining);

      // The worker method called by each thread.
      // We can do this in one of (at least) two ways:
      // 1. Have an interlocked loop variable or
      //    Pros: Great for load balancing
      //    Cons: Per-core caching not so coherent
      for (;;)
      {
         long i = InterlockedIncrement(&item.m_current) - 1;
         if (i >= item.m_stop)
            break;
         try
         {
            item.m_func(static_cast<int>(i));
         }
         catch (...)
         {
            break;
         }
      }
      // or 2. Pre-assign a chunk of work to each thread.
      //    Pros: Per-core caching coherent
      //    Cons: Poor load balancing with non-uniform workloads

      // Signal when no more threads 
      if (InterlockedDecrement(&item.m_remaining) == 0)
         item.m_Done.signal();
   }
   
   int ThreadProc()
   {
      for (;;)
      {
         // Wait for abort event or do-work event
         if ((m_Abort + m_Semaphore).wait_any() == 0)
            return 0;
         DoWork(m_item);
      }
   }

   std::vector<ts::thread> m_threads;
   std::vector<DWORD> m_ids;
   semaphore m_Semaphore;
   event m_Abort;
   critical_section m_cs;
   Item m_item;

private:
   static unsigned int GetLogicalProcessorCount()
   {
      DWORD logicalProcessorCount = 0;
      DWORD returnLength = 0;

      GetLogicalProcessorInformation(NULL, &returnLength);
      if (returnLength > 0)
      {
         std::vector<BYTE> arr(returnLength);
         SYSTEM_LOGICAL_PROCESSOR_INFORMATION* ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)&arr[0];
         if (GetLogicalProcessorInformation(ptr, &returnLength) != 0)
         {
            const DWORD slpisize = sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            for (; returnLength >= slpisize; ++ptr, returnLength -= slpisize)
            {
               if (ptr->Relationship == RelationProcessorCore)
               {
                  while (ptr->ProcessorMask > 0)
                  {
                     logicalProcessorCount += ptr->ProcessorMask & 1;
                     ptr->ProcessorMask >>= 1;
                  }
               }
            }
         }
      }
      return std::max<unsigned int>(1, logicalProcessorCount);
   }
};

__declspec(selectany) /* static */ thread_pool thread_pool::s_instance;

}
