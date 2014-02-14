#pragma once

/*
   Copyright Microsoft Corpoation 2011

   Date: 16 September 2011
   
   Author: Toby Sharp (tsharp)
*/

#include <array>
#include <vector>
#include <cassert>

template <typename T, size_t N>
std::array<T, N> operator -(const std::array<T, N>& rhs)
{
   std::array<T, N> rv;
   for (size_t i = 0; i < N; i++)
      rv[i] = -rhs[i];
   return rv;
}

template <typename T, size_t N>
std::array<T, N> operator +(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
   std::array<T, N> rv;
   for (int i = 0; i < N; i++)
      rv[i] = lhs[i] + rhs[i];
   return rv;
}

template <typename T, size_t N>
std::array<T, N>& operator +=(std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
   std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<T>());
   return lhs;
}

template <typename T, size_t N>
std::array<T, N> operator -(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
   std::array<T, N> rv;
   for (int i = 0; i < N; i++)
      rv[i] = lhs[i] - rhs[i];
   return rv;
}

template <typename T, size_t N>
std::array<T, N>& operator -=(std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
   for (size_t i = 0; i < N; i++)
      lhs[i] -= rhs[i];
   return lhs;
}

template <typename T, size_t N>
std::array<T, N> operator *(const std::array<T, N>& lhs, T rhs)
{
   std::array<T, N> rv;
   for (int i = 0; i < N; i++)
      rv[i] = lhs[i] * rhs;
   return rv;
}

template <typename T, size_t N>
std::array<T, N> operator *(T lhs, const std::array<T, N>& rhs)
{
   return operator *(rhs, lhs);
}

template <typename T, size_t N>
std::array<T, N> operator /(const std::array<T, N>& lhs, T rhs)
{
   std::array<T, N> rv;
   for (int i = 0; i < N; i++)
      rv[i] = lhs[i] / rhs;
   return rv;
}

template <typename T>
std::vector<T> operator *(T lhs, const std::vector<T>& rhs)
{
   return operator *(rhs, lhs);
}

template <typename T>
std::vector<T> operator *(const std::vector<T>& lhs, T rhs)
{
   std::vector<T> rv(lhs.size());
   std::transform(lhs.begin(), lhs.end(), rv.begin(), [&](const T& val) { return val * rhs; });
   return rv;
}

template <typename T>
std::vector<T> operator +(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
   assert(lhs.size() == rhs.size());
   std::vector<T> rv(lhs.size());
   std::transform(lhs.begin(), lhs.end(), rhs.begin(), rv.begin(), [](const T& lhs, const T& rhs) { return lhs + rhs; });
   return rv;
}

template <typename T>
std::vector<T> operator -(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
   assert(lhs.size() == rhs.size());
   std::vector<T> rv(lhs.size());
   std::transform(lhs.begin(), lhs.end(), rhs.begin(), rv.begin(), [](const T& lhs, const T& rhs) { return lhs - rhs; });
   return rv;
}

template <typename T>
std::vector<T>& operator -=(std::vector<T>& lhs, const std::vector<T>& rhs)
{
   assert(lhs.size() == rhs.size());
   std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), [](const T& lhs, const T& rhs) { return lhs - rhs; });
   return lhs;
}

template <typename T>
std::vector<T>& operator +=(std::vector<T>& lhs, const std::vector<T>& rhs)
{
   assert(lhs.size() == rhs.size());
   std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<T>());
   return lhs;
}

template <typename T>
std::vector<T>& operator *=(std::vector<T>& lhs, T rhs)
{
   std::transform(lhs.begin(), lhs.end(), lhs.begin(), [&](const T& lhs) { return lhs * rhs; });
   return lhs;
}

template <typename T>
std::vector<T> operator /(const std::vector<T>& lhs, T rhs)
{
   std::vector<T> rv(lhs.size());
   std::transform(lhs.begin(), lhs.end(), rv.begin(), [&](const T& val) { return val / rhs; });
   return rv;
}

template <typename T>
std::vector<T> operator -(const std::vector<T>& rhs)
{
   std::vector<T> rv(rhs.size());
   std::transform(rhs.begin(), rhs.end(), rv.begin(), [](const T& val) { return -val; });
   return rv;
}

namespace ts
{

template <size_t N2, typename T, size_t N>
std::array<T, N2> truncate(const std::array<T, N>& lhs)
{
   std::array<T, N2> rv;
   for (size_t i = 0; i < N2; i++)
      rv[i] = lhs[i];
   return rv;
}

template <typename _It>
typename _It::value_type sqrlength(_It start, _It end)
{
   typename _It::value_type sum(0);
   for (_It i = start; i != end; ++i)
      sum += *i * *i;
   return sum;
}

template <typename _TVec>
typename _TVec::value_type sqrlength(const _TVec& lhs)
{
   return sqrlength(lhs.begin(), lhs.end());
}

template <typename _TVec>
typename _TVec::value_type length(const _TVec& lhs)
{
   return std::sqrt(sqrlength(lhs));
}

template <typename T, size_t N>
T sqrdist(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
   return sqrlength(lhs - rhs);
}

template <typename T, size_t N>
T distance(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
   return std::sqrt(sqrdist(lhs, rhs));
}

template <typename LeftIt, typename RightIt>
typename LeftIt::value_type dot(LeftIt lhs_start, LeftIt lhs_end, RightIt rhs_start)
{
   typename LeftIt::value_type rv(0);
   RightIt rhs(rhs_start);
   for (LeftIt lhs(lhs_start); lhs != lhs_end; ++lhs, ++rhs)
      rv += *lhs * *rhs;
   return rv;
}

template <typename T, size_t N>
T dot(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
   return dot(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
T dot(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
   assert(lhs.size() == rhs.size());
   return dot(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T, size_t N>
std::array<T, N> normalize_unchecked(const std::array<T, N>& rhs)
{
   T recip = 1 / length(rhs);
   return rhs * recip;
}

template <typename T>
inline T sigmoid(T x)
{
   return 1 / (1 + std::exp(-x));
}

typedef std::array<float, 4> float4;
typedef std::array<float, 3> float3;

}