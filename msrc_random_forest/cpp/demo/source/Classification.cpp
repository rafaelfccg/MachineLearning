#include "Classification.h"

#include "FeatureResponseFunctions.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  AxisAlignedFeatureResponse AxisAlignedFeatureResponseFactory::CreateRandom(Random& random)
  {
    return AxisAlignedFeatureResponse::CreateRandom(random);
  }

  LinearFeatureResponse2d LinearFeatureFactory::CreateRandom(Random& random)
  {
    return LinearFeatureResponse2d::CreateRandom(random);
  }


#pragma region added by jie feng

  AxisAlignedFeatureResponseND AxisAlignedFeatureResponseFactoryND::CreateRandom(Random& random, int N)
  {
	  return AxisAlignedFeatureResponseND::CreateRandom(random, N);
  }

  LinearFeatureResponseND LinearFeatureFactoryND::CreateRandom(Random& random, int N)
  {
	  return LinearFeatureResponseND::CreateRandom(random, N);
  }

#pragma endregion added by jie feng


} } }

