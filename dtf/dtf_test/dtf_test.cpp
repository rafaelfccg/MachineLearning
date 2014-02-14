/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 4 November 2011
   
*/
#include <iostream>

#include "imageio.h"

namespace SimpleTest { void Run(); }
namespace StandardTest { void Run(); }
namespace FunctionMinimizationTest { void Run(); }
namespace EntropyTest { void Run(); }
namespace PairwiseTest { void Run(); }
namespace EstimationTest { void Run(); }

int main()
{
   try
   {
      SimpleTest::Run();
      StandardTest::Run();
      PairwiseTest::Run();
      //EntropyTest::Run();
      EstimationTest::Run();

      std::cout << "All tests passed." << std::endl;
   }
   catch (std::exception& e)
   {
      std::cout << "Test failed: " << e.what() << std::endl;
   }
   catch (...)
   {
      std::cout << "Test failed." << std::endl;
   }
   return 0;
}