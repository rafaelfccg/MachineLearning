#pragma once

/*
   Copyright Microsoft Corporation 2011
   
   Author: Toby Sharp (tsharp)
   
   Date: 23 August 2011
*/

#include <istream>
#include <ostream>
#include <map>
#include <functional>

namespace ts
{

template <typename T> 
inline T read(std::istream& in)
{
   T t;
   in.read(reinterpret_cast<char*>(&t), sizeof(t));
   if (in.fail())
      throw std::exception("Input stream read failed");
   return t;
}

template <typename T> 
inline void read(std::istream& in, T& t)
{
   in.read(reinterpret_cast<char*>(&t), sizeof(t));
   if (in.fail())
      throw std::exception("Input stream read failed");
}

template <typename T>
inline void write(std::ostream& out, const T& t)
{
   out.write(reinterpret_cast<const char*>(&t), sizeof(t));
   if (out.fail())
      throw std::exception("Output stream write failed");
}

template <>
inline std::string read(std::istream& in)
{
   unsigned int length = 0;
   unsigned char byte = 0x80; 
   for (int i = 0; byte & 0x80; i += 7)
   {
      byte = read<unsigned char>(in);
      length |= (byte & 0x7F) << i;
   }
   
   std::string str(length, '\0');
   in.read(&str[0], length);
   return str;
}

template <>
inline void write(std::ostream& out, const std::string& str)
{
   unsigned int length = static_cast<unsigned int>(str.size());
   unsigned char byte = 0x80;
   for (int i = 0; byte & 0x80; i += 7)
   {
      byte = ((length >> i) & 0x7F) | ((length >> i) > 0x7F ? 0x80 : 0x00);
      write(out, byte);
   }

   out.write(&str[0], length);
}

template <typename T>
inline std::ostream& operator <<(std::ostream& os, const std::vector<T>& vec)
{
   write(os, static_cast<unsigned __int64>(vec.size()));
   for (auto i = vec.begin(); i != vec.end(); ++i)
      os << *i;
   return os;
}

template <typename T>
inline std::istream& operator >>(std::istream& is, std::vector<T>& vec)
{
   size_t size = static_cast<size_t>(read<unsigned __int64>(is));
   vec.resize(size);
   for (auto i = vec.begin(); i != vec.end(); ++i)
      is >> *i;
   return is;
}

inline void skip(std::istream& in, int bytes)
{
   in.seekg(bytes, std::ios_base::cur);
}

template <typename T> 
inline std::ostream& write_raw(std::ostream& lhs, const std::vector<T>& vec)
{
   write<unsigned __int64>(lhs, vec.size());
   write<unsigned int>(lhs, sizeof(T));
   if (!vec.empty())
      lhs.write(reinterpret_cast<const char*>(&vec[0]), vec.size() * sizeof(T));
   return lhs;
}

template <typename T> 
inline std::istream& read_raw(std::istream& lhs, std::vector<T>& vec)
{
   unsigned __int64 size = read<unsigned __int64>(lhs);
   unsigned int a = read<unsigned int>(lhs);
   if (a != sizeof(T))
      throw std::exception("Vector element of unexpected size");
   vec.resize(static_cast<size_t>(size));
   if (!vec.empty())
      lhs.read(reinterpret_cast<char*>(&vec[0]), vec.size() * sizeof(T));
   return lhs;
}

template <typename Base>
class factory
{
public:
   template <typename T>
   void add_type()
   {
      m_map.insert(std::make_pair(typeid(T).name(), []() { return std::unique_ptr<T>(new T()); }));
   }

   std::unique_ptr<Base> operator()(const std::string& name) const
   {
      auto i = m_map.find(name);
      if (i == m_map.end())
         throw std::exception("Detected type does not exist in the factory. Use factory::add_type.");
      return (*i).second();
   }

protected:
   typedef std::function<std::unique_ptr<Base>()> create_fn;
   std::map<std::string, create_fn> m_map;
};

template <typename T>
void save_object(std::ostream& os, const T& t)
{
   write(os, std::string(typeid(t).name()));
   os << t;
}

template <typename Base>
std::unique_ptr<Base> load_object(std::istream& is, const factory<Base>& factory)
{
   std::string name = read<std::string>(is);
   std::unique_ptr<Base> ptr = factory(name);
   is >> *ptr;
   return ptr;
}

}