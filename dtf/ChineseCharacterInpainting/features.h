#pragma once

#include <algorithm>
#include <random>
#include <cassert>

#include "dtf.h"
#include "imageio.h"

namespace ChineseCharacterInpainting {

// Simple observation feature: at a relative offset to the current factor location,
// test whether the pixel is black, white, or unobserved
class simple_feature {
public:
   typedef ts::image<ts::bgr24> preprocess_type;
   typedef ts::bgr24 pixel_type;
   enum ObservationType {
      Black = 0,
      White = 1,
      Unobserved = 2,
   };

   simple_feature(void)
      : off1_x(0), off1_y(0), off2_x(0), off2_y(0), var_count(0)
   {
   }
   simple_feature(int off1_x, int off1_y, ObservationType obs1)
      : off1_x(off1_x), off1_y(off1_y), var_count(1), obs1(obs1)
   {
   }
   simple_feature(int off1_x, int off1_y, int off2_x, int off2_y, ObservationType obs1, ObservationType obs2)
      : off1_x(off1_x), off1_y(off1_y), off2_x(off2_x), off2_y(off2_y), var_count(2), obs1(obs1), obs2(obs2)
   {
   }

   static const preprocess_type& pre_process(const ts::image<ts::bgr24>& image)
   {
      return image;
   }

   template <typename _It>
   bool operator()(int x, int y, const preprocess_type& prep, _It offsets) const
   {
      if (var_count == 1) {
         // Feature test at a unary factor type
         int pos_x = clamp(x + offsets[0].x + off1_x, prep.width()-1);
         int pos_y = clamp(y + offsets[0].y + off1_y, prep.height()-1);
         return (obs1 == pix_to_obs(prep(pos_x, pos_y)));
      } else {
         // Feature test at a pairwise or higher-order factor type
         int pos1_x = clamp(x + offsets[0].x + off1_x, prep.width()-1);
         int pos1_y = clamp(y + offsets[0].y + off1_y, prep.height()-1);
         ObservationType pix_obs1 = pix_to_obs(prep(pos1_x, pos1_y));

         int pos2_x = clamp(x + offsets[1].x + off2_x, prep.width()-1);
         int pos2_y = clamp(y + offsets[1].y + off2_y, prep.height()-1);
         ObservationType pix_obs2 = pix_to_obs(prep(pos2_x, pos2_y));

         return (pix_obs1 == obs1) && (pix_obs2 == obs2);
      }
   }

private:
   int off1_x, off1_y;
   int off2_x, off2_y;
   int var_count;
   ObservationType obs1;
   ObservationType obs2;

   int clamp(int val, int max_val) const
   {
      return std::max(0, std::min(val, max_val));
   }

   static ObservationType pix_to_obs(const ts::bgr24& pix)
   {
      if (pix[0] <= 64)
      {
         return Black;
      }
      else if (pix[1] >= 192)
      {
         return White;
      }
      return Unobserved;
   }
};

class simple_feature_sampler {
public:
   typedef simple_feature TFeature;

   simple_feature_sampler(int offset_dist, unsigned int factor_var_count, unsigned long seed = 1)
      : eng(seed), rcentered(0.5), roffset(-offset_dist, offset_dist),
         rtype(0, 2), factor_var_count(factor_var_count)
   {
      assert(offset_dist >= 0);
   }

   simple_feature operator()(void)
   {
      if (factor_var_count == 1) {
         return simple_feature(get_offset(), get_offset(), simple_feature::ObservationType(rtype(eng)));
      } else {
         return simple_feature(get_offset(), get_offset(), get_offset(), get_offset(),
            simple_feature::ObservationType(rtype(eng)), simple_feature::ObservationType(rtype(eng)));
      }
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
   std::bernoulli_distribution rcentered;
   std::uniform_int_distribution<int> roffset;
   std::uniform_int_distribution<int> rtype;
   unsigned int factor_var_count;

   int get_offset(void)
   {
      if (rcentered(eng))
         return 0;

      return roffset(eng);
   }
};

}
