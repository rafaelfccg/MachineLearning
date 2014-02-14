#ifndef GRADIENTMAP_H
#define GRADIENTMAP_H

/*
   Copyright Microsoft Corporation 2012
   
   Author: Sebastian Nowozin (senowozi)
   
   Date: 2 May 2012
*/

#include <array>
#include <algorithm>
#include <cmath>

#include "dtf.h"
#include "image.h"
#include "integralimage.h"

namespace SceneUnderstanding {

template<typename TPixelType, unsigned int direction_bin_count>
class gradientmap {
private:
   typedef std::array<float, direction_bin_count> bin_type;
   typedef std::array<double, direction_bin_count> integral_bin_type;

   float gradient(const TPixelType& pix1, const TPixelType& pix2) const {
      int v1 = 0;
      int v2 = 0;
      for (size_t channel = 0; channel < pix1.size(); ++channel) {
         v1 += pix1[channel];
         v2 += pix2[channel];
      }
      return static_cast<float>(v1-v2) / static_cast<float>(pix1.size());
   }

   ts::image<bin_type> compute_orientation_maps(const ts::image<TPixelType>& image) {
      ts::image<bin_type> oriented_gradients(image.width(), image.height());

      // Bin and weight gradient magnitudes and directions
      dtf::dense_sampling samples = dtf::all_pixels(image);
      ts::parallel_for(samples.begin(), samples.end(), [&](dtf::dense_pixel_iterator sample)
      {
         dtf::offset_t pos = *sample;
         dtf::offset_t left = dtf::offset_t(pos.x == 0 ? 0 : (pos.x-1), pos.y);
         dtf::offset_t top = dtf::offset_t(pos.x, pos.y == 0 ? 0 : (pos.y-1));
         auto pix_center = image(pos.x, pos.y);
         auto pix_left = image(left.x, left.y);
         auto pix_top = image(top.x, top.y);

         float gx = gradient(pix_center, pix_left);
         float gy = gradient(pix_center, pix_top);
         float gmag = std::sqrt(gx*gx + gy*gy);
         float gdir = std::atan2f(gy, gx);

         // Bin
         const float pi = 3.14159265358979323f;
         size_t bin = static_cast<size_t>((static_cast<float>(direction_bin_count) * (gdir+pi)) / (2.0*pi));
         bin = std::min(bin, static_cast<size_t>(direction_bin_count)-1);
         assert(bin < direction_bin_count);
         oriented_gradients(pos.x, pos.y)[bin] += gmag;
      });
      return oriented_gradients;
   }

public:
   // I(x,y)[di] is the direction bin 'di' integral from (0,0) to (x,y).
   integralimage<bin_type, integral_bin_type> I;

   gradientmap(void) {
   }

   gradientmap(const ts::image<TPixelType>& image) {
      // Compute orientation maps and their integral images
      auto oriented_gradients = compute_orientation_maps(image);
      I = integralimage<bin_type, integral_bin_type>(oriented_gradients);
   }

   unsigned int number_of_directions(void) const {
      return direction_bin_count;
   }

   double compute_mean_response(unsigned int direction, int x1, int y1, int x2, int y2) const {
      return I.compute_mean_response(direction, x1, y1, x2, y2);
   }

   static_assert(direction_bin_count >= 1, "Number of gradient direction bins must be >= 1.");
};

}

#endif
