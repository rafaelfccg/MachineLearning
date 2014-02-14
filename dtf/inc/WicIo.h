#pragma once

/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 23 August 2011

*/

#pragma push_macro("min")
#pragma push_macro("max")

#include <atlbase.h>
#include <atlstr.h>
#include <wincodec.h>
#include <stdexcept>

struct CoUninitCaller
{
   bool bUninit;
   CoUninitCaller() : bUninit(false) {}
   ~CoUninitCaller()
   {
      if (bUninit)
         ::CoUninitialize();
   }
};

class ImageLoader
{
public:
   ImageLoader(LPCWSTR path) : m_width(0), m_height(0), m_bpp(0)
   {
      CComPtr<IWICImagingFactory> pFactory;
      HRESULT hr = pFactory.CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER);
      if (hr == CO_E_NOTINITIALIZED)
      {
         m_uninit.bUninit = SUCCEEDED(::CoInitializeEx(NULL, COINIT_MULTITHREADED));
         if (FAILED(pFactory.CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER)))
            throw std::exception("Has this thread called CoInitializeEx?");
      }
      else if (FAILED(hr))
         throw std::exception("Failed to create WIC imaging factory.");

      CComPtr<IWICBitmapDecoder> pDecoder;
      hr = pFactory->CreateDecoderFromFilename(path, NULL, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &pDecoder);
      if (FAILED(hr))
         throw std::exception("Failed to open or parse file.");

      if (FAILED(pDecoder->GetFrame(0, &m_pFrame)))
         throw std::exception("Failed to decode frame.");

      WICPixelFormatGUID format;
      m_pFrame->GetSize(&m_width, &m_height);
      m_pFrame->GetPixelFormat(&format);

      if (format == GUID_WICPixelFormat8bppIndexed || format == GUID_WICPixelFormat8bppGray)
         m_bpp = 8;
      else if (format == GUID_WICPixelFormat16bppGray)
         m_bpp = 16;
      else if (format == GUID_WICPixelFormat24bppRGB || format == GUID_WICPixelFormat24bppBGR)
         m_bpp = 24;
      else if (format == GUID_WICPixelFormat32bppBGR || format == GUID_WICPixelFormat32bppRGBA || format == GUID_WICPixelFormat32bppBGRA)
         m_bpp = 32;
      else if (format == GUID_WICPixelFormat48bppRGB || format == GUID_WICPixelFormat48bppBGR)
         m_bpp = 48;
      else
         throw std::exception("Unsupported image format.");
   }

   unsigned int Width() { return m_width; }
   unsigned int Height() { return m_height; }
   unsigned int Bpp() { return m_bpp; }

   bool Write(void* pBits, unsigned int stride, unsigned int size)
   {
      if (m_bpp > 0)
      {
         if (SUCCEEDED(m_pFrame->CopyPixels(NULL, stride, size, (unsigned char*)pBits)))
            return true;
      }
      return false;
   }

protected:
   CoUninitCaller m_uninit;
   CComPtr<IWICBitmapFrameDecode> m_pFrame;
   unsigned int m_width, m_height, m_bpp;
};

class ImageWriter
{
public:
   static void Save(LPCWSTR path, const void* pBits,
                     int width, int height, int bpp, int stride,
                     bool bIndexed = false, bool bHalftone = true)
   {
      CoUninitCaller uninit;
      {
         CComPtr<IWICImagingFactory> pFactory;
         HRESULT hr = pFactory.CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER);
         if (hr == CO_E_NOTINITIALIZED)
         {
            uninit.bUninit = SUCCEEDED(::CoInitializeEx(NULL, COINIT_MULTITHREADED));
            if (FAILED(pFactory.CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER)))
               throw std::exception("Has this thread called CoInitializeEx?");
         }
         else if (FAILED(hr))
            throw std::exception("Failed to create WIC imaging factory.");

         {
            CComPtr<IWICStream> pStream;
            pFactory->CreateStream(&pStream);
            if (FAILED(pStream->InitializeFromFilename(path, GENERIC_WRITE)))
               throw std::exception("Invalid path");

            CComPtr<IWICBitmapEncoder> pEncoder;
            pFactory->CreateEncoder(GetEncoderFormatFromPath(path), NULL, &pEncoder);
            HRESULT hr = pEncoder->Initialize(pStream, WICBitmapEncoderNoCache);

            CComPtr<IWICBitmapFrameEncode> pBitmapFrame;
            CComPtr<IPropertyBag2> pPropertyBag;
            hr = pEncoder->CreateNewFrame(&pBitmapFrame, &pPropertyBag);
            hr = pBitmapFrame->Initialize(pPropertyBag);
            hr = pBitmapFrame->SetSize(width, height);
            WICPixelFormatGUID guidPixelFormat = GetPixelFormat(bpp, bIndexed);
            hr = pBitmapFrame->SetPixelFormat(&guidPixelFormat);
            if (bpp == 8 && bIndexed)
            {
               unsigned char maxValue = 0;
               for (int y = 0; y < height; y++)
               {
                  const unsigned char* p = (const unsigned char*)pBits + y * stride;
                  for (int x = 0; x < width; x++)
                  {
                     if (p[x] > maxValue)
                        maxValue = p[x];
                  }
               }

               CComPtr<IWICPalette> pPalette;
               pFactory->CreatePalette(&pPalette);
               WICBitmapPaletteType paletteType = GetPaletteType(maxValue, bHalftone);
               pPalette->InitializePredefined(paletteType, FALSE);
               pBitmapFrame->SetPalette(pPalette);
            }
            hr = pBitmapFrame->WritePixels(height, stride, height * stride, (unsigned char*)pBits);
            hr = pBitmapFrame->Commit();
            hr = pEncoder->Commit();
         }
      }
   }
protected:
   static WICBitmapPaletteType GetPaletteType(unsigned char maxValue, bool bHalftone)
   {
      int count = (int)maxValue + 1;
      if (bHalftone)
      {
         if (count <= 2)
            return WICBitmapPaletteTypeFixedBW;
         else if (count <= 8)
            return WICBitmapPaletteTypeFixedHalftone8;
         else if (count <= 27)
            return WICBitmapPaletteTypeFixedHalftone27;
         else if (count <= 64)
            return WICBitmapPaletteTypeFixedHalftone64;
         else if (count <= 125)
            return WICBitmapPaletteTypeFixedHalftone125;
         else if (count <= 216)
            return WICBitmapPaletteTypeFixedHalftone216;
         else if (count <= 252)
            return WICBitmapPaletteTypeFixedHalftone252;
         else
            return WICBitmapPaletteTypeFixedHalftone256;
      }
      else
      {
         if (count <= 2)
            return WICBitmapPaletteTypeFixedBW;
         else if (count <= 4)
            return WICBitmapPaletteTypeFixedGray4;
         else if (count <= 16)
            return WICBitmapPaletteTypeFixedGray16;
         else
            return WICBitmapPaletteTypeFixedGray256;
      }
   }

   static WICPixelFormatGUID GetPixelFormat(int bpp, bool bIndexed)
   {
      if (bpp == 8 && bIndexed)
         return GUID_WICPixelFormat8bppIndexed;
      else if (bpp == 8)
         return GUID_WICPixelFormat8bppGray;
      else if (bpp == 16)
         return GUID_WICPixelFormat16bppGray;
      else if (bpp == 24)
         return GUID_WICPixelFormat24bppBGR;
      else if (bpp == 48)
         return GUID_WICPixelFormat48bppRGB;
      else if (bpp == 32)
         return GUID_WICPixelFormat32bppBGRA;
      else
         throw std::exception("Bit depth not handled.");
   }

   static const GUID& GetEncoderFormatFromPath(LPCTSTR path)
   {
      CString strpath(path);
      int dotpos = strpath.ReverseFind(_T('.'));
      if (dotpos < 0)
         throw std::exception("No file extension supplied.");
      CString ext = strpath.Mid(dotpos + 1);

      if (Match(ext, _T("bmp")))
         return GUID_ContainerFormatBmp;
      else if (Match(ext, _T("jpg")) || Match(ext, _T("jpeg")))
         return GUID_ContainerFormatJpeg;
      else if (Match(ext, _T("png")))
         return GUID_ContainerFormatPng;
      else if (Match(ext, _T("gif")))
         return GUID_ContainerFormatGif;
      else if (Match(ext, _T("tif")) || Match(ext, _T("tiff")))
         return GUID_ContainerFormatTiff;
      else if (Match(ext, _T("wmp")))
         return GUID_ContainerFormatWmp;
      else
         throw std::exception("File extension unrecognized.");
   }

   static bool Match(const CString& ext, LPCTSTR match)
   {
      return ext.CompareNoCase(match) == 0;
   }
};

#pragma pop_macro("min")
#pragma pop_macro("max")

