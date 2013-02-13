#include "FeatureResponseFunctions.h"

#include <cmath>

#include <sstream>

#include "DataPointCollection.h"
#include "Random.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

	//////////////////////////////////////////////////////////////////////////
  AxisAlignedFeatureResponse AxisAlignedFeatureResponse ::CreateRandom(Random& random)
  {
    return AxisAlignedFeatureResponse(random.Next(0, 2));
  }

  float AxisAlignedFeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const
  {
    const DataPointCollection& concreteData = (DataPointCollection&)(data);
    return concreteData.GetDataPoint((int)sampleIndex)[axis_];
  }

  std::string AxisAlignedFeatureResponse::ToString() const
  {
    std::stringstream s;
    s << "AxisAlignedFeatureResponse(" << axis_ << ")";

    return s.str();
  }


#pragma region nD axis feature by jie feng

  AxisAlignedFeatureResponseND AxisAlignedFeatureResponseND ::CreateRandom(Random& random, int N)
  {
	  return AxisAlignedFeatureResponseND(random.Next(0, N));	// [0, N-1]
  }

  float AxisAlignedFeatureResponseND::GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const
  {
	  const DataPointCollection& concreteData = (DataPointCollection&)(data);
	  return concreteData.GetDataPoint((int)sampleIndex)[axis_];
  }

  std::string AxisAlignedFeatureResponseND::ToString() const
  {
	  std::stringstream s;
	  s << "GeneralAxisAlignedFeatureResponse(" << axis_ << ")";

	  return s.str();
  }

#pragma endregion nD axis feature by jie feng
  


  /// <returns>A new LinearFeatureResponse2d instance.</returns>
  LinearFeatureResponse2d LinearFeatureResponse2d::CreateRandom(Random& random)
  {
    double dx = 2.0 * random.NextDouble() - 1.0;
    double dy = 2.0 * random.NextDouble() - 1.0;

    double magnitude = sqrt(dx * dx + dy * dy);

    return LinearFeatureResponse2d((float)(dx / magnitude), (float)(dy / magnitude));
  }

  float LinearFeatureResponse2d::GetResponse(const IDataPointCollection& data, unsigned int index) const
  {
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    return dx_ * concreteData.GetDataPoint((int)index)[0] + dy_ * concreteData.GetDataPoint((int)index)[1];
  }

  std::string LinearFeatureResponse2d::ToString() const
  {
    std::stringstream s;
    s << "LinearFeatureResponse(" << dx_ << "," << dy_ << ")";

    return s.str();
  }


#pragma region nd linear feature by jie feng

  LinearFeatureResponseND LinearFeatureResponseND::CreateRandom(Random& random, int N)
  {
	  // create coefficients between [-1,1]
	  vector<float> dparams(N,0);
	  float sqsumd = 0;
	  for(size_t i=0; i<dparams.size(); i++)
	  {
		  dparams[i] = 2.0 * random.NextDouble() - 1.0;
		  sqsumd += dparams[i] * dparams[i];
	  }

	  float magnitude = sqrt(sqsumd);

	  // normalize
	  for(size_t i=0; i<dparams.size(); i++)
		  dparams[i] /= magnitude;

	  return LinearFeatureResponseND(dparams);
  }

  float LinearFeatureResponseND::GetResponse(const IDataPointCollection& data, unsigned int index) const
  {
	  const DataPointCollection& concreteData = (const DataPointCollection&)(data);
	  assert(concreteData.Count() == d_pos.size());

	  float res = 0;
	  for(size_t i=0; i<d_pos.size(); i++)
		  res += d_pos[i] * concreteData.GetDataPoint((int)index)[i];

	  return res;
  }

  std::string LinearFeatureResponseND::ToString() const
  {
	  std::stringstream s;
	  s << "LinearFeatureResponse(";
	  for(size_t i=0; i<d_pos.size(); i++)
		s << d_pos[i] << ",";
	  s<<")";

	  return s.str();
  }

#pragma endregion nd linear feature by jie feng

} } }
