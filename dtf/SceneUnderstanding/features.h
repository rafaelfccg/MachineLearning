#pragma once

#include <random>
#include <algorithm>
#include <limits>
#include <cmath>

#include "dtf.h"
#include "imageio.h"
#include "integralimage.h"
#include "gradientmap.h"

namespace SceneUnderstanding {

// A simple location feature independent of the image content.
// The feature encodes the pixel position as a 0.0-1.0 range of x and y position within image.
// For semantic segmentation tasks this is a useful location prior
// ("sky is at the top, grass is at the bottom, object is in the center").
class location_feature {
public:
   typedef ts::image<ts::bgr24> preprocess_type;

   location_feature(void) : axis(0), axis_fraction(0.0)
   {
   }

   location_feature(size_t axis, float axis_fraction)
      : axis(axis), axis_fraction(axis_fraction)
   {
   }

   static const preprocess_type& pre_process(const ts::image<ts::bgr24>& image)
   {
      return image;
   }

   template <typename _It>
   bool operator()(int x, int y, const preprocess_type& prep, _It offsets) const
   {
      if (axis >= 0 && axis <= 1) {
         // x-axis or y-axis position
         float axis_pos = static_cast<float>((axis == 0) ? x : y);
         float axis_len = static_cast<float>((axis == 0) ? prep.width() : prep.height());
         return (axis_pos / axis_len) <= axis_fraction;
      } else {
         // distance from image center
         float cx = static_cast<float>(2*x - prep.width());
         cx /= static_cast<float>(prep.width());   // cx is in [-1,1] now, where 0 is the center of the image

         float cy = static_cast<float>(2*y - prep.height());
         cy /= static_cast<float>(prep.height());

         float dist = 0.5f*(cy*cy + cx*cx);
         return dist <= axis_fraction;
      }
   }

private:
   size_t axis;
   float axis_fraction;
};

class location_feature_sampler {
public:
   typedef location_feature TFeature;

   location_feature_sampler(unsigned long seed = 1)
      : eng(seed), axis_dist(0, 2)
   {
   }

   TFeature operator()(void)
   {
      return location_feature(axis_dist(eng), fraction_dist(eng));
   }

   TFeature operator()(const TFeature::preprocess_type& prep)
   {
      return operator()();
   }

   template <typename PrepCache>
   std::vector<TFeature> operator()(PrepCache& cache, unsigned int count)
   {
      std::vector<TFeature> rv(count);
      for (unsigned int i = 0; i < count; ++i)
         rv[i] = operator()();
      return rv;
   }

private:
   std::mt19937 eng;
   std::uniform_int_distribution<size_t> axis_dist;
   std::uniform_real_distribution<float> fraction_dist;
};

// Feature: oriented gradient mean response in a rectangle (space-continuous Histogram-of-Gradients)
template<unsigned int direction_bin_count = 8>
class oriented_gradient_feature {
public:
   typedef gradientmap<ts::bgr24, direction_bin_count> preprocess_type;

   oriented_gradient_feature(void)
      : direction(0), off_x(0), off_y(0), size_x(0), size_y(0)
   {
   }

   oriented_gradient_feature(int direction, int off_x, int off_y, int size_x, int size_y, double thresh)
      : direction(direction), off_x(off_x), off_y(off_y), size_x(size_x), size_y(size_y), thresh(thresh)
   {
   }

   static preprocess_type pre_process(const ts::image<ts::bgr24>& image)
   {
      return preprocess_type(image);
   }

   double compute_response(int x, int y, const preprocess_type& prep) const
   {
      return prep.compute_mean_response(direction,
         x+off_x, y+off_y, x+off_x+size_x, y+off_y+size_y);
   }

   template <typename _It>
   bool operator()(int x, int y, const preprocess_type& prep, _It offsets) const
   {
      double response = compute_response(x, y, prep);

      return response >= thresh;
   }

   void set_thresh(double thresh)
   {
      this->thresh = thresh;
   }

private:
   int direction;
   int off_x;
   int off_y;
   int size_x;
   int size_y;
   double thresh;
};

template<unsigned int direction_bin_count = 8>
class oriented_gradient_feature_sampler {
public:
   typedef oriented_gradient_feature<direction_bin_count> TFeature;

   // The (x,y) offsets for the feature tests are sampled from a Student t distribution, as
   //     x,y ~ offset_sigma * f(dof),
   // where
   //     offset_sigma: The multiplier on the radius.
   //        If dof is very large this is just the Normal standard deviation.
   //     dof: degrees of freedom.
   // The box_max is the maximum integral box size along each dimension.
   oriented_gradient_feature_sampler(int box_max, double offset_sigma, double dof = 4.0, unsigned long seed = 1)
      : eng(seed), radius_dist(dof), box_dist(0, box_max), direction_dist(0, direction_bin_count-1),
         offset_sigma(offset_sigma)
   {
   }

   TFeature operator()(void)
   {
      return TFeature(direction_dist(eng), sample_offset(), sample_offset(),
         sample_box(), sample_box(), std::numeric_limits<double>::signaling_NaN());
   }

   TFeature operator()(typename const TFeature::preprocess_type& prep)
   {
      std::uniform_int_distribution<int> rx(0, prep.I.I.width()-1);
      std::uniform_int_distribution<int> ry(0, prep.I.I.height()-1);

      // Create random feature, but then set the threshold by sampling from the data
      auto feat = operator()();
      feat.set_thresh(feat.compute_response(rx(eng), ry(eng), prep));

      return feat;
   }

   template <typename PrepCache>
   std::vector<TFeature> operator()(PrepCache& cache, unsigned int count)
   {
      // Get a uniformly distributed set of image indices, one for each test to sample
      std::uniform_int_distribution<size_t> rimage(0, cache.database_size()-1);
      std::vector<size_t> image_idx(count);
      for (unsigned int i = 0; i < count; ++i)
         image_idx[i] = rimage(eng);
      std::sort(image_idx.begin(), image_idx.end());

      // Sample feature tests
      std::vector<TFeature> rv(count);
      for (unsigned int i = 0; i < count; ++i)
         rv[i] = operator()(cache.pre_processed<TFeature>(image_idx[i]));

      return rv;
   }

private:
   std::mt19937 eng;
   std::student_t_distribution<double> radius_dist;
   std::uniform_int_distribution<int> box_dist;
   std::uniform_int_distribution<unsigned int> direction_dist;
   double offset_sigma;

   int sample_offset(void)
   {
      return static_cast<int>(std::floor(offset_sigma*radius_dist(eng) + 0.5));
   }

   int sample_box(void)
   {
      return static_cast<int>(std::floor(box_dist(eng) + 0.5));
   }
};


// Simple mean and variance of RGB color channels in a given rectangle.
// Computation is made efficient using integral images.
class simple_pixel_feature {
public:
   simple_pixel_feature(void)
      : channel1(0), channel2(0), off1_x(0), off1_y(0), size1_x(0), size1_y(0),
         off2_x(0), off2_y(0), size2_x(0), size2_y(0), thresh(0)
   {
   }
   simple_pixel_feature(int channel1, int channel2,
      int off1_x, int off1_y, int size1_x, int size1_y,
      int off2_x, int off2_y, int size2_x, int size2_y, double thresh)
      : channel1(channel1), channel2(channel2),
         off1_x(off1_x), off1_y(off1_y), size1_x(size1_x), size1_y(size1_y),
         off2_x(off2_x), off2_y(off2_y), size2_x(size2_x), size2_y(size2_y), thresh(thresh)
   {
   }

   struct feat_prep
   {
   public:
      feat_prep(const ts::image<ts::bgr24>& image)
      {
         ts::image<ts::bgri64> image1(image.width(), image.height());
         ts::image<ts::bgri128> image2(image.width(), image.height());
         for (unsigned int y(0); y < image2.height(); ++y) {
            for (unsigned int x(0); x < image2.width(); ++x) {
               image1(x, y)[3] = 0;
               image2(x, y)[3] = 0;
               for (unsigned int ch(0); ch <= 2; ++ch) {
                  image1(x, y)[ch] = image(x, y)[ch];
                  image2(x, y)[ch] = static_cast<std::uint32_t>(image(x, y)[ch])
                     * static_cast<std::uint32_t>(image(x, y)[ch]);

                  image1(x, y)[3] += image(x, y)[ch];
               }
               image1(x, y)[3] /= 3;   // mean intensity
               image2(x, y)[3] = image1(x, y)[3] * image1(x, y)[3];  // product of intensity
            }
         }
         I = integralimage<ts::bgri64, ts::bgri256>(image1);
         I2 = integralimage<ts::bgri128, ts::bgri256>(image2);
      }
      integralimage<ts::bgri64, ts::bgri256> I;	// raw integral image of RGBI channels
      integralimage<ts::bgri128, ts::bgri256> I2;	// integral image of RGBI^2 channels
   };
   typedef feat_prep preprocess_type;

   static preprocess_type pre_process(const ts::image<ts::bgr24>& image)
   {
      return feat_prep(image);
   }

   double compute_response(int x, int y, const feat_prep& prep) const
   {
      // Channel 0-3: raw RGBI channels
      // Channel 4-7: RGB variances in rectangle
      if (channel1 >= 4) {
         int ch = channel1 - 4;	// get original channel, apply to both variances
         double var1 = compute_variance(prep, ch, x+off1_x, y+off1_y, x+off1_x+size1_x, y+off1_y+size1_y);
         double var2 = compute_variance(prep, ch, x+off2_x, y+off2_y, x+off2_x+size2_x, y+off2_y+size2_y);
         return var1 - var2;
      }

      // Compare plain RGB channels instead
      double v1 = prep.I.compute_mean_response(channel1, x+off1_x, y+off1_y, x+off1_x+size1_x, y+off1_y+size1_y);
      double v2 = 0;
      if (channel2 < 4)
         v2 = prep.I.compute_mean_response(channel2, x+off2_x, y+off2_y, x+off2_x+size2_x, y+off2_y+size2_y);

      return v1 - v2;
   }

   // A simple integral feature test ala TextonBoost
   template <typename _It>
   bool operator()(int x, int y, const feat_prep& prep, _It offsets) const
   {
      double resp = compute_response(x, y, prep);
      return resp >= thresh;
   }

   void set_thresh(double thresh)
   {
      this->thresh = thresh;
   }

private:
   int channel1;
   int channel2;
   int off1_x, off1_y;
   int size1_x, size1_y;
   int off2_x, off2_y;
   int size2_x, size2_y;
   double thresh;

   static double compute_variance(const feat_prep& prep, int ch,
      int x1, int y1, int x2, int y2)
   {
      double E_x2 = prep.I2.compute_mean_response(ch, x1, y1, x2, y2);
      double sq_E_x = prep.I.compute_mean_response(ch, x1, y1, x2, y2);
      sq_E_x *= sq_E_x;

      return E_x2 - sq_E_x;	// variance decomposition
   }
};

class simple_pixel_feature_sampler {
public:
   typedef simple_pixel_feature TFeature;

   // The (x,y) offsets for the feature tests are sampled from a Student t distribution, as
   //     x,y ~ offset_sigma * f(dof),
   // where
   //     offset_sigma: The multiplier on the radius.
   //         If dof is very large this is just the Normal standard deviation.
   //     dof: degrees of freedom.
   // The box_max is the maximum integral box size along each dimension.
   simple_pixel_feature_sampler(int box_max, double offset_sigma, double dof = 4.0, unsigned long seed = 1)
      : eng(seed), radius_dist(dof), box_dist(0, box_max), channel_dist(0, 7), offset_sigma(offset_sigma)
   {
   }

   simple_pixel_feature operator()(void)
   {
      return simple_pixel_feature(channel_dist(eng), channel_dist(eng),
         sample_offset(), sample_offset(), sample_box(), sample_box(),
         sample_offset(), sample_offset(), sample_box(), sample_box(), 0);
   }
   simple_pixel_feature operator()(const TFeature::preprocess_type& prep)
   {
      std::uniform_int_distribution<int> rx(0, prep.I.I.width()-1);
      std::uniform_int_distribution<int> ry(0, prep.I.I.height()-1);

      // Create random feature, but then set the threshold by sampling from the data
      auto feat = operator()();
      feat.set_thresh(feat.compute_response(rx(eng), ry(eng), prep));

      return feat;
   }

   template <typename PrepCache>
   std::vector<TFeature> operator()(PrepCache& cache, unsigned int count)
   {
      // Get a uniformly distributed set of image indices, one for each test to sample
      std::uniform_int_distribution<size_t> rimage(0, cache.database_size()-1);
      std::vector<size_t> image_idx(count);
      for (unsigned int i = 0; i < count; ++i)
         image_idx[i] = rimage(eng);
      std::sort(image_idx.begin(), image_idx.end());

      // Sample feature tests
      std::vector<TFeature> rv(count);
      for (unsigned int i = 0; i < count; ++i)
         rv[i] = operator()(cache.pre_processed<TFeature>(image_idx[i]));

      return rv;
   }

private:
   std::mt19937 eng;
   std::student_t_distribution<double> radius_dist;
   std::uniform_int_distribution<int> box_dist;
   std::uniform_int_distribution<int> channel_dist;
   double offset_sigma;

   int sample_offset(void)
   {
      return static_cast<int>(std::floor(offset_sigma*radius_dist(eng) + 0.5));
   }
   int sample_box(void)
   {
      return static_cast<int>(std::floor(box_dist(eng) + 0.5));
   }
};

// A linear hyperplane on the RGB vector
class color_feature {
public:
   typedef ts::image<ts::bgr24> preprocess_type;
   typedef ts::bgr24 pixel_type;

   color_feature(void)
      : thresh(0)
   {
   }
   color_feature(const pixel_type& vcolor, double thresh)
      : vcolor(vcolor), thresh(thresh)
   {
   }

   static const preprocess_type& pre_process(const ts::image<ts::bgr24>& image)
   {
      return image;
   }

   double compute_response(int x, int y, const preprocess_type& prep) const
   {
      return std::inner_product(prep(x, y).cbegin(), prep(x, y).cend(), vcolor.cbegin(), 0.0);
   }

   template <typename _It>
   bool operator()(int x, int y, const preprocess_type& prep, _It offsets) const
   {
      return compute_response(x, y, prep) >= thresh;
   }

   void set_vector_thresh(const pixel_type& vec, double thresh)
   {
      this->vcolor = vec;
      this->thresh = thresh;
   }

private:
   pixel_type vcolor;
   double thresh;
};

class color_feature_sampler {
public:
   typedef color_feature TFeature;

   color_feature_sampler(unsigned long seed = 1)
      : eng(seed)
   {
   }

   color_feature operator()(void)
   {
      color_feature::pixel_type pix;
      return color_feature(pix, 0);
   }
   color_feature operator()(const TFeature::preprocess_type& prep)
   {
      std::uniform_int_distribution<int> rx(0, prep.width()-1);
      std::uniform_int_distribution<int> ry(0, prep.height()-1);

      auto feat = operator()();

      // Sample two pixels at random, use difference vector as norm
      color_feature::pixel_type vec;
      double thresh = 0;
      const color_feature::pixel_type& p1 = prep(rx(eng),ry(eng));
      const color_feature::pixel_type& p2 = prep(rx(eng),ry(eng));
      for (size_t ci = 0; ci < vec.size(); ++ci) {
         vec[ci] = p2[ci] - p1[ci]; // vec = v2 - v1
         thresh += p2[ci]*p2[ci] - p1[ci]*p1[ci];
      }
      thresh *= 0.5;
      feat.set_vector_thresh(vec, thresh);

      return feat;
   }

   template <typename PrepCache>
   std::vector<TFeature> operator()(PrepCache& cache, unsigned int count)
   {
      // Get a uniformly distributed set of image indices, one for each test to sample
      std::uniform_int_distribution<size_t> rimage(0, cache.database_size()-1);
      std::vector<size_t> image_idx(count);
      for (unsigned int i = 0; i < count; ++i)
         image_idx[i] = rimage(eng);
      std::sort(image_idx.begin(), image_idx.end());

      // Sample feature tests
      std::vector<TFeature> rv(count);
      for (unsigned int i = 0; i < count; ++i)
         rv[i] = operator()(cache.pre_processed<TFeature>(image_idx[i]));

      return rv;
   }

private:
   std::mt19937 eng;
};

// A simple contrast-sensitive feature
// Note: only applicable to pairwise factors!
class contrast_feature {
public:
   typedef ts::image<ts::bgr24> preprocess_type;
   typedef ts::bgr24 pixel_type;

   contrast_feature(void)
      : alpha(0)
   {
   }
   contrast_feature(double alpha)
      : alpha(alpha)
   {
   }

   static const preprocess_type& pre_process(const ts::image<ts::bgr24>& image)
   {
      return image;
   }

   double compute_response(int x, int y, const preprocess_type& prep, int xoff, int yoff) const
   {
      int i1 = compute_intensity(x, y, prep);
      int i2 = compute_intensity(x + xoff, y + yoff, prep);

      double sq_diff = static_cast<double>((i1 - i2) * (i1 - i2));

      return alpha * sq_diff;
   }

   template <typename _It>
   bool operator()(int x, int y, const preprocess_type& prep, _It offsets) const
   {
      return compute_response(x, y, prep, offsets[1].x, offsets[1].y) >= 1.0;
   }

   void set_alpha(double alpha)
   {
      this->alpha = alpha;
   }

private:
   double alpha;
   double thresh;

   int clamp(int val, int min_val, int max_val) const
   {
      return std::max(min_val, std::min(max_val, val));
   }

   int compute_intensity(int x, int y, const preprocess_type& prep) const
   {
      x = clamp(x, 0, prep.width());
      y = clamp(y, 0, prep.height());

      return prep(x, y)[0] + prep(x, y)[1] + prep(x, y)[2];
   }
};

class contrast_feature_sampler {
public:
   typedef contrast_feature TFeature;

   contrast_feature_sampler(unsigned long seed = 1)
      : eng(seed)
   {
   }

   contrast_feature operator()(void)
   {
      return contrast_feature(1.0);
   }
   contrast_feature operator()(const TFeature::preprocess_type& prep)
   {
      std::uniform_int_distribution<int> rx(0, prep.width()-2);   // we use +1 x offsets
      std::uniform_int_distribution<int> ry(0, prep.height()-1);

      auto feat = operator()();

      // Sample a pair of adjacent pixels at random to determine alpha
      feat.set_alpha(1.0 / feat.compute_response(rx(eng), ry(eng), prep, 1, 0)); // make this instance 1.0

      return feat;
   }

   template <typename PrepCache>
   std::vector<TFeature> operator()(PrepCache& cache, unsigned int count)
   {
      // Sample feature tests
      std::vector<TFeature> rv(count);
      std::uniform_int_distribution<size_t> rimage(0, cache.database_size()-1);
      for (unsigned int i = 0; i < count; ++i)
         rv[i] = operator()(cache.pre_processed<TFeature>(rimage(eng)));

      return rv;
   }

private:
   std::mt19937 eng;
};


}
