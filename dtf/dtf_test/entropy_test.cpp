#include "dtf.h"
#include "imageio.h"

namespace EntropyTest
{

/*
   Author your training database here.
*/
class SplatDatabase 
{
public:
   static const dtf::label_t label_count = 2;
   static const unsigned int image_count = 3;

   SplatDatabase() : m_train(image_count), m_gt(image_count)
   {
      for (unsigned int i = 0; i < image_count; i++)
      {
         std::wstringstream s;
         s << "splat" << i;
         m_train[i] = ts::load<ts::bgr24>(s.str() + L".png");
         m_gt[i] = ts::load<dtf::label_t>(s.str() + L"_gt.png");
      }
   }
   typedef ts::image<ts::bgr24> training_image;
   typedef ts::image<dtf::label_t> gt_image;
   size_t size() const { return image_count; }
   training_image get_training(size_t index) const { return m_train[index]; }
   gt_image get_ground_truth(size_t index) const { return m_gt[index]; }
   dtf::dense_sampling get_samples(size_t index) const { return dtf::all_pixels(m_gt[index]); }
private:
   mutable std::vector<training_image> m_train;
   mutable std::vector<gt_image> m_gt;
};

/*
   Author your features here.
*/
class RgbValueFeature
{
public:
   static const ts::image<ts::bgr24>& pre_process(const ts::image<ts::bgr24>& image)
   {
      return image;
   }
   template <typename _It>
   bool operator()(int x, int y, const ts::image<ts::bgr24>& image, _It offsets) const
   {
      unsigned char a = image(x, y)[c];
      return a >= lo && a < hi;
   }
   int c;
   int lo, hi;
};

int irand(int imin, int inext)
{
   if (imin >= inext)
      throw std::invalid_argument("irand: imin >= inext");
   int range = inext - imin;
   int bits = 0;
   int rangeCopy = range;
   while (rangeCopy > 0)
   {
      rangeCopy >>= 1;
      ++bits;
   }
   if (bits > 15)
      throw std::invalid_argument("irand: bits > 15");
   int mask = (1 << bits) - 1;
   int r;
   do
   {
      r = rand() & mask;
   } while (r >= range);
   return r + imin;
}

class RgbValueFeatureSampler
{
public:
   RgbValueFeatureSampler(int width) : a(0), b(0), c(0), cycle(0)
   {
      m_width = width;
      m_bins = (256 + width - 1) / width;
      cycle = 3 * (m_bins + m_bins - 1);
   }

   template <typename PrepCache>
   std::vector<RgbValueFeature> operator()(PrepCache&, unsigned int count)
   {
      std::vector<RgbValueFeature> rv(count);
      for (unsigned int i = 0; i < count; ++i)
      {
         RgbValueFeature feature;
         feature.c = c;
         feature.lo = (int)(m_width * a);
         feature.hi = (int)(m_width * (b + 1));

         if (b == m_bins - 1)
         {
            if (a == m_bins - 1)
            {
               c = (c + 1) % 3;
               a = b = 0;
            }
            else
               a++;
         }
         else
            b++;
         rv[i] = feature;
      }
      return rv;
   }

   int cycle;
private:
   int m_bins;
   double m_width;
   int a, b, c;
};

void Run()
{
   RgbValueFeatureSampler featureSampler(1);
   SplatDatabase trainingSet;

   auto tree = dtf::training::LearnDecisionTreeUnary(
                           trainingSet,
                           featureSampler,
                           featureSampler.cycle,
                           30);

   dtf::classify::decision_tree_classifier<trainingSet.label_count, RgbValueFeature> classifier(tree);

   auto labelling = classifier(trainingSet.get_training(0));
   ts::save(labelling, L"classify0.png", true);

   double error = dtf::classify::classification_accuracy(trainingSet, classifier);
   std::cout << "Training accuracy " << 100.0 * error << "%" << std::endl;
}

}