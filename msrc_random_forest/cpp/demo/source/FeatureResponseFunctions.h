#pragma once

// This file defines some IFeatureResponse implementations used by the example code in
// Classification.h, DensityEstimation.h, etc. Note we represent IFeatureResponse
// instances using simple structs so that all tree data can be stored
// contiguously in a linear array.

#include <string>
#include <vector>
using namespace std;

#include "Sherwood.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  class Random;

  /// <summary>
  /// A feature that orders data points using one of their coordinates,
  /// i.e. by projecting them onto a coordinate axis.
  /// </summary>
  class AxisAlignedFeatureResponse
  {
    int axis_;	// specific axis

  public:
    AxisAlignedFeatureResponse()
    {
      axis_ = -1;
    }

    /// <summary>
    /// Create an AxisAlignedFeatureResponse instance for the specified axis.
    /// </summary>
    /// <param name="axis">The zero-based index of the axis.</param>
    AxisAlignedFeatureResponse(int axis)
    {
      axis_ = axis;
    }

    /// <summary>
    /// Create an AxisAlignedFeatureResponse instance with a random choice of axis.
    /// </summary>
    /// <param name="randomNumberGenerator">A random number generator.</param>
    /// <returns>A new AxisAlignedFeatureResponse instance.</returns>
    static AxisAlignedFeatureResponse CreateRandom(Random& random);

    int Axis() const
    {
      return axis_;
    }

    // IFeatureResponse implementation
    float GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const;

    std::string ToString() const;
  };

  //////////////////////////////////////////////////////////////////////////
  // added by Jie Feng
  //////////////////////////////////////////////////////////////////////////
  // a n dimensional axis feature
  class AxisAlignedFeatureResponseND
  {
	  int axis_;

  public:
	  AxisAlignedFeatureResponseND()
	  {
		  axis_ = -1;
	  }

	  /// <summary>
	  /// Create an AxisAlignedFeatureResponse instance for the specified axis.
	  /// </summary>
	  /// <param name="axis">The zero-based index of the axis.</param>
	  AxisAlignedFeatureResponseND(int axis)
	  {
		  axis_ = axis;
	  }

	  /// <summary>
	  /// Create an AxisAlignedFeatureResponse instance with a random choice of axis.
	  /// </summary>
	  /// <param name="randomNumberGenerator">A random number generator.</param>
	  /// <returns>A new AxisAlignedFeatureResponse instance.</returns>
	  static AxisAlignedFeatureResponseND CreateRandom(Random& random, int N);

	  int Axis() const
	  {
		  return axis_;
	  }

	  // IFeatureResponse implementation
	  float GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const;

	  std::string ToString() const;
  };



  /// <summary>
  /// A feature that orders data points using a linear combination of their
  /// coordinates, i.e. by projecting them onto a given direction vector.
  /// </summary>
  class LinearFeatureResponse2d
  {
    float dx_, dy_;

  public:
    LinearFeatureResponse2d()
    {
      dx_ = 0.0;
      dy_ = 0.0;
    }

    /// <summary>
    /// Create a LinearFeatureResponse2d instance for the specified direction vector.
    /// </summary>
    /// <param name="dx">The first element of the direction vector.</param>
    /// <param name="dx">The second element of the direction vector.</param> 
    LinearFeatureResponse2d(float dx, float dy)
    {
      dx_ = dx; dy_ = dy;
    }

    /// <summary>
    /// Create a LinearFeatureResponse2d instance with a random direction vector.
    /// </summary>
    /// <param name="randomNumberGenerator">A random number generator.</param>
    /// <returns>A new LinearFeatureResponse2d instance.</returns>
    static LinearFeatureResponse2d CreateRandom(Random& random);

    // IFeatureResponse implementation
    float GetResponse(const IDataPointCollection& data, unsigned int index) const;

    std::string ToString()  const;
  };	


  //////////////////////////////////////////////////////////////////////////
  // n dimensional linear feature by Jie Feng
  //////////////////////////////////////////////////////////////////////////
  class LinearFeatureResponseND
  {
	  float dx_, dy_;
	  float d_pos[256];	// params for each coordinate; vector will cause wrong for default serialization

	  int MAX_FEAT_DIM;

  public:
	  LinearFeatureResponseND(): MAX_FEAT_DIM(256)
	  {
		  dx_ = 0.0;
		  dy_ = 0.0;
	  }

	  /// <summary>
	  /// Create a LinearFeatureResponse2d instance for the specified direction vector.
	  /// </summary>
	  /// <param name="dx">The first element of the direction vector.</param>
	  /// <param name="dx">The second element of the direction vector.</param> 
	  LinearFeatureResponseND(vector<float>& d_data): MAX_FEAT_DIM(256)
	  {
		  if(d_data.size() > MAX_FEAT_DIM)
			  throw std::runtime_error("Linear feature nd dimension could not exceed 256.");

		  for(size_t i=0; i<d_data.size(); i++)
			  d_pos[i] = d_data[i];
	  }

	  /// <summary>
	  /// Create a LinearFeatureResponse2d instance with a random direction vector.
	  /// </summary>
	  /// <param name="randomNumberGenerator">A random number generator.</param>
	  /// <returns>A new LinearFeatureResponse2d instance.</returns>
	  static LinearFeatureResponseND CreateRandom(Random& random, int N);

	  // IFeatureResponse implementation
	  float GetResponse(const IDataPointCollection& data, unsigned int index) const;

	  std::string ToString()  const;
  };	


} } }
