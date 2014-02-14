
#pragma once

#include <random>
#include <cassert>

// A helper class to combine multiple features into one
template<typename TFeature1, typename TFeature2>
class combined_feature
{
public:
   combined_feature(void)
   {
   }

   combined_feature(const TFeature1& feat1)
      : feat1(feat1), select_first(true)
   {
   }
   combined_feature(const TFeature2& feat2)
      : feat2(feat2), select_first(false)
   {
   }

   struct combined_preprocess_type {
      typename TFeature1::preprocess_type prep1;
      typename TFeature2::preprocess_type prep2;

      template<typename TImage>
      combined_preprocess_type(const TImage& image)
         : prep1(TFeature1::pre_process(image)), prep2(TFeature2::pre_process(image))
      {
      }
   };

   typedef combined_preprocess_type preprocess_type;

   template<typename TImage>
   static preprocess_type pre_process(const TImage& image)
   {
      return combined_preprocess_type(image);
   }

   template<typename _It>
   bool operator()(int x, int y, const preprocess_type& prep, _It offsets) const
   {
      if (select_first)
      {
         return feat1(x, y, prep.prep1, offsets);
      }
      else
      {
         return feat2(x, y, prep.prep2, offsets);
      }
   }

private:
   TFeature1 feat1;
   TFeature2 feat2;
   bool select_first;
};

// Corresponding feature sampler, initialized with the two feature samplers and
// the fraction of feature1 types sampled.
template<typename TFeatureSampler1, typename TFeatureSampler2>
class combined_feature_sampler
{
public:
   typedef typename TFeatureSampler1::TFeature TFeature1;
   typedef typename TFeatureSampler2::TFeature TFeature2;
   typedef combined_feature<TFeature1, TFeature2> TFeature;

   combined_feature_sampler(const TFeatureSampler1& sampler1, const TFeatureSampler2& sampler2,
      double frac_first = 0.5, unsigned long seed = 29001)
      : sampler1(sampler1), sampler2(sampler2), frac_first(frac_first), eng(seed)
   {
      assert(frac_first >= 0.0 && frac_first <= 1.0);
   }

   TFeature operator()(typename const TFeature::preprocess_type& prep)
   {
      std::uniform_real_distribution<double> runi;
      if (runi(eng) <= frac_first)
         return TFeature(sampler1(prep.prep1));
      else
         return TFeature(sampler2(prep.prep2));
   }

   template <typename PrepCache>
   std::vector<TFeature> operator()(PrepCache& cache, unsigned int count)
   {
      // Split the number of features to generate between the types
      std::binomial_distribution<unsigned int> rfeat1(count, frac_first);
      unsigned int count1 = rfeat1(eng);
      unsigned int count2 = count - count1;

      // Generate features of corresponding sub types
      std::vector<TFeature1> rv1 = sampler1(cache, count1);
      std::vector<TFeature2> rv2 = sampler2(cache, count2);

      // Concatenate generated features and wrap using a combined_feature
      std::vector<TFeature> rv;
      rv.reserve(count);
      for (unsigned int i = 0; i < count1; ++i)
         rv.push_back(TFeature(rv1[i]));
      for (unsigned int i = 0; i < count2; ++i)
         rv.push_back(TFeature(rv2[i]));

      return rv;
   }

private:
   std::mt19937 eng;
   std::uniform_real_distribution<double> rfirst;

   TFeatureSampler1 sampler1;
   TFeatureSampler2 sampler2;
   double frac_first;
};
