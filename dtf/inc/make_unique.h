#pragma once

#include <memory>

namespace ts
{

template<typename _Ty, typename _Arg0> 
std::unique_ptr<_Ty> make_unique(_Arg0&& arg0) 
{ 
   return std::unique_ptr<_Ty>(new _Ty(std::forward<_Arg0>(arg0))); 
} 

template<typename _Ty, typename _Arg0, typename _Arg1> 
std::unique_ptr<_Ty> make_unique(_Arg0&& arg0, _Arg1&& arg1) 
{ 
   return std::unique_ptr<_Ty>(new _Ty(std::forward<_Arg0>(arg0), std::forward<_Arg1>(arg1))); 
} 

}