
#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cassert>

#include "imageio.h"
#include "dtf.h"

namespace ts {

// General dataset class for the DTF standard dataset format
template <unsigned char _LabelCount, typename _Pixel>
class dataset {
public:
   static const dtf::label_t label_count = _LabelCount;

   // Generic data set class
   //
   // base_path: The directory where there are subdirectories "labels" and "images".
   // setname: The file base_path\setname.txt lists the image base names in this image set.
   dataset(const std::wstring& base_path, const std::wstring& setname, double subsampling) :
      m_base_path(base_path), m_setname(setname), m_subsampling(subsampling)
   {
      read_basenames();
      m_cache.resize(size());
   }

   size_t size() const { return basenames.size(); }

   ts::image<const _Pixel> get_training(size_t i) const
   {
      assert(i < basenames.size());
      #pragma omp critical
      {
         if (m_cache[i].input.empty())
            m_cache[i].input = ts::load<_Pixel>(get_image_filepath(i));
      }

      return m_cache[i].input;
   }
   const std::wstring& get_basename(size_t i) const
   {
      return basenames[i];
   }

   ts::image<const unsigned char> get_ground_truth(size_t i) const
   {
      assert(i < basenames.size());
      #pragma omp critical
      {
         if (m_cache[i].gt.empty())
            m_cache[i].gt = ts::load<unsigned char>(get_gt_filepath(i));
      }

      return m_cache[i].gt;
   }

   dtf::sparse_sampling get_samples(size_t i) const
   {
      return dtf::sparse_subset_pixels(get_ground_truth(i), m_subsampling, static_cast<unsigned long>(i));
   }

protected:
   void read_basenames(void)
   {
      std::wstringstream setfilepath;
      setfilepath << m_base_path << "\\" << m_setname << ".txt";
      std::wifstream ifs(setfilepath.str());

      if (ifs.fail())
         throw std::runtime_error("Failed to open image basename set file.");

      while (ifs.eof() == false)
      {
         std::wstring img_basename;
         ifs >> img_basename;
         if (img_basename.empty() == false)
            basenames.push_back(img_basename);
      }
      ifs.close();
   }

   std::wstring get_filepath(const std::wstring& sub_folder,
      const std::wstring& extension, size_t i) const
   {
      assert(i < basenames.size());
      std::wstringstream fpath;
      fpath << m_base_path << "\\" << sub_folder << "\\" << basenames[i] << "." << extension;

      return fpath.str();
   }

   std::wstring get_image_filepath(size_t i) const
   {
      return get_filepath(L"images", L"jpg", i);
   }

   std::wstring get_gt_filepath(size_t i) const
   {
      return get_filepath(L"labels", L"png", i);
   }

   std::wstring m_base_path;	// Data set directory
   std::wstring m_setname;	// Name of data set subset, e.g. "train"
   double m_subsampling;
   std::vector<std::wstring> basenames;	// Set of image basenames

   struct element {
      ts::image<_Pixel> input;
      ts::image<unsigned char> gt;
   };
   mutable std::vector<element> m_cache;
};

}
