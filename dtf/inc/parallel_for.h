#pragma once

/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 13 September 2011
   
*/
#define TS_PARALLEL_FOR_SERIAL 0
#define TS_PARALLEL_FOR_OPENMP 1
#define TS_PARALLEL_FOR_HANDCODED 2
#define TS_PARALLEL_FOR_CONCRT 3

#ifndef TS_PARALLEL_FOR_IMPL
#ifdef __cplusplus_cli
#define TS_PARALLEL_FOR_IMPL TS_PARALLEL_FOR_SERIAL
#else
// Uncomment exactly one of these lines
//#define TS_PARALLEL_FOR_IMPL TS_PARALLEL_FOR_SERIAL
#define TS_PARALLEL_FOR_IMPL TS_PARALLEL_FOR_OPENMP
//#define TS_PARALLEL_FOR_IMPL TS_PARALLEL_FOR_HANDCODED
//#define TS_PARALLEL_FOR_IMPL TS_PARALLEL_FOR_CONCRT
#endif
#endif

#if TS_PARALLEL_FOR_IMPL == TS_PARALLEL_FOR_SERIAL // Serial

namespace ts
{

template <typename _Tidx, typename Func>
inline void parallel_for(_Tidx start, _Tidx stop, Func func)
{
   for (_Tidx i = start; i != stop; ++i)
      func(i);
}

inline unsigned int thread_get_index() { return 0; }
inline unsigned int thread_get_count() { return 1; }

}

#elif TS_PARALLEL_FOR_IMPL == TS_PARALLEL_FOR_OPENMP // OpenMP

#include <omp.h> // Forces run-time dependency (undesirable)

namespace ts
{

template <typename _Tidx, typename Func>
inline void parallel_for(_Tidx start, _Tidx stop, Func func)
{
   int count = static_cast<int>(stop - start);
   #pragma omp parallel for schedule(guided)
   for (int i = 0; i < count; i++)
      func(start + i);
}

inline unsigned int thread_get_index()
{
   return omp_get_thread_num();
}

inline unsigned int thread_get_count()
{
   return omp_get_max_threads();
}

}

#elif TS_PARALLEL_FOR_IMPL == TS_PARALLEL_FOR_HANDCODED // Hand-coded

#include "thread_pool.h"

namespace ts
{

template <typename _Tidx, typename Func>
inline void parallel_for(_Tidx start, _Tidx stop, Func func)
{
   int count = static_cast<int>(stop - start);
   thread_pool::ParallelFor(0, count, [&](int i) 
      { func(start + i); });
}

inline unsigned int thread_get_index()
{
   return thread_pool::GetThreadIndex();
}

inline unsigned int thread_get_count()
{
   return thread_pool::GetThreadCount();
}

}

#elif TS_PARALLEL_FOR_IMPL == TS_PARALLEL_FOR_CONCRT // ConcRT

#include <ppl.h> // Forces run-time dependency (undesirable)
#include "ppl_partitioner.h"

namespace ts
{

template <typename _Tidx, typename Func>
inline void parallel_for(_Tidx start, _Tidx stop, Func func)
{
   int count = static_cast<int>(stop - start);
   Concurrency::parallel_for(0, count, [&](int i) { func(start + i); } );
      // , Concurrency::fixed_partitioner(count));
}

inline unsigned int thread_get_count()
{
   return Concurrency::CurrentScheduler::GetPolicy().GetPolicyValue(Concurrency::MaxConcurrency) + 1;
}

inline unsigned int thread_get_index()
{
   int id = Concurrency::Context::CurrentContext()->GetVirtualProcessorId();
   return id < 0 ? 0 : static_cast<unsigned int>(id);
}

}

#endif

namespace ts
{
template <typename _Tidx, typename Func>
inline void serial_for(_Tidx start, _Tidx stop, Func func)
{
   for (_Tidx i = start; i != stop; ++i)
      func(i);
}

template <typename T, typename _TIt, typename _Func>
T reduce(_TIt start, _TIt stop, const T& init, _Func reducer)
{ 
   T rv(init);
   for (_TIt i = start; i != stop; ++i)
      reducer(rv, *i);
   return rv;
}

template <typename T>
class thread_locals
{
public:
   thread_locals() : m_locals(thread_get_count()) {}
   thread_locals(const T& t) : m_locals(thread_get_count(), t) {}
   T& local() { return m_locals[thread_get_index()]; }
   template <typename _Func> void for_each(_Func func)
   {
      for (auto i = m_locals.begin(); i != m_locals.end(); ++i)
         func(*i);
   }
   template <typename _Func> void parallel_for_each(_Func func)
   {
      parallel_for(m_locals.begin(), m_locals.end(), [&](std::vector<T>::iterator i)
      { func(*i); });
   }
   unsigned int size() const { return static_cast<unsigned int>(m_locals.size()); }
   T& operator[](unsigned int index) { return m_locals[index]; }
   const T& operator[](unsigned int index) const { return m_locals[index]; }
   T& operator*() { return local(); }
   T* operator ->() { return &local(); }

   template <typename _Func> T reduce(const T& start, _Func func) const
   {
      return ts::reduce<T>(m_locals.begin(), m_locals.end(), start, func);
   }
   typename std::vector<T>::const_iterator begin() const { return m_locals.begin(); }
   typename std::vector<T>::const_iterator end() const { return m_locals.end(); }

protected:
   std::vector<T> m_locals;
};

}
