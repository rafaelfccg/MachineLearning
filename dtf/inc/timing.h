#pragma once

/*
   Copyright Microsoft Corpoation 2011

   Date: 16 September 2011
   
   Author: Toby Sharp (tsharp)
*/

#include <vector>
#include <algorithm>
#include "safewin.h"

namespace ts
{

// Execute 'func' 'repeat' times, and return the median duration of the function calls, in ms.
template <typename Func> inline float timing_median_ms(unsigned int repeat, Func func)
{
   if (repeat > 0) // Timings for the function
   {
      std::vector<__int64> ticks(repeat);
      LARGE_INTEGER llFreq, llStart, llStop;
      ::QueryPerformanceFrequency(&llFreq);

      // Increase thread priority for timings
      HANDLE hThread = ::GetCurrentThread();
      int nPriority = ::GetThreadPriority(hThread);
      if (nPriority != THREAD_PRIORITY_ERROR_RETURN)
         ::SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL);

      // Do the timed runs
      for (unsigned int i = 0; i < repeat; i++)
      {
         ::QueryPerformanceCounter(&llStart);
         func();
         ::QueryPerformanceCounter(&llStop);
         ticks[i] = llStop.QuadPart - llStart.QuadPart;
      }

      // Revert thread priority
      if (nPriority != THREAD_PRIORITY_ERROR_RETURN)
         ::SetThreadPriority(hThread, nPriority);

      // Return median value in ms
      std::nth_element(ticks.begin(), ticks.begin() + ticks.size() / 2, ticks.end());
      return (float)(1000.0 * (double)(ticks[ticks.size()/2]) / (double)llFreq.QuadPart);
   }
   return 0.0f;
}

// Execute 'func', and return the duration of the function call, in ms.
template <typename Func> inline float timing_ms(Func func)
{
   return timing_median_ms(1, func);
}

}
