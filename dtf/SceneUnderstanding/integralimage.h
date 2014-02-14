#pragma once

#include <array>
#include <cstdint>

#include "image.h"
#include "parallel_for.h"

namespace ts {
   typedef std::array<std::uint16_t, 4> bgri64;
   typedef std::array<std::uint32_t, 4> bgri128;
   typedef std::array<std::uint64_t, 4> bgri256;
}

namespace SceneUnderstanding {

// TPixelType: the type of the input image pixel.
// TIntegralPixelType: the pixel type we use for the integral image.  Needs to have as many channels as TPixelType.
template<typename TPixelType, typename TIntegralPixelType>
class integralimage {
public:
   integralimage(void)
   {
   }

   integralimage(const ts::image<TPixelType>& image)
   {
      compute_integral_image(image);
   }

   // Compute the average value in the given channel over the rectangle from (x1,y1) to (x2,y2), inclusive.
   double compute_mean_response(int channel, int x1, int y1, int x2, int y2) const
   {
      x1 = clamp(x1, 0, I.width()-1);
      y1 = clamp(y1, 0, I.height()-1);
      x2 = clamp(x2, 0, I.width()-1);
      y2 = clamp(y2, 0, I.height()-1);
      assert(y2 >= y1);
      assert(x2 >= x1);

      size_t pixel_count = count_rectangle(x1, y1, x2, y2);
      auto rect_sum = sum_rectangle(channel, x1, y2, x2, y2);

      return static_cast<double>(rect_sum) / static_cast<double>(pixel_count);
   }

   // Integral image.  Semantics: I(x,y) = sum_{c <= x, r <= y} image(c,r).
   ts::image<TIntegralPixelType> I;

private:
   void compute_integral_image(const ts::image<TPixelType>& image)
   {
      I.resize(image.width(), image.height());

      // Parallelize over channels
      typedef typename TIntegralPixelType::value_type pix_sum_type;
      ts::parallel_for(0, std::tuple_size<TPixelType>::value, [&](int ci) -> void
      {
         for (unsigned int y = 0; y < image.height(); ++y)
         {
            pix_sum_type rs = 0; // work around MSVC 2010 bug (using TIntegralPixelType::value_type does not work at this place inside parallel_for)
            for (unsigned int x = 0; x < image.width(); ++x)
            {
               rs += image(x,y)[ci];
               I(x,y)[ci] = (y > 0 ? I(x,y-1)[ci] : 0) + rs;
            }
         }
      });
   }

   static int clamp(int value, int min, int max)
   {
      return std::min(std::max(min, value), max);
   }

   static size_t count_rectangle(int x1, int y1, int x2, int y2)
   {
      size_t width(x2 - x1 + 1);
      size_t height(y2 - y1 + 1);

      return width * height;
   }

   // Assumption: the coordinates are valid to access I with and y2 >= y1, x2 >= x1.
   typename TIntegralPixelType::value_type sum_rectangle(int ch, int x1, int y1, int x2, int y2) const
   {
      typename TIntegralPixelType::value_type v_x2y2 = I(x2, y2)[ch];
      typename TIntegralPixelType::value_type v_x1y1 = (x1 > 0 && y1 > 0) ? I(x1-1, y1-1)[ch] : 0;
      typename TIntegralPixelType::value_type v_x2y1 = (y1 > 0) ? I(x2, y1-1)[ch] : 0;
      typename TIntegralPixelType::value_type v_x1y2 = (x1 > 0) ? I(x1-1, y2)[ch] : 0;

      return v_x2y2 + v_x1y1 - v_x2y1 - v_x1y2;
   }
};

}
