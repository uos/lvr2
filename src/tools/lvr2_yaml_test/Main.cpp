#include "lvr2/io/PlutoMetaDataIO.hpp"
#include "lvr2/io/DataStruct.hpp"

using namespace lvr2;
using Vec = lvr2::BaseVector<float>;

int main(int argc, char** argv)
{
    boost::filesystem::path p("/home/lennart/spectral.yaml");
    floatArr ang;
  size_t size = PlutoMetaDataIO::readSpectralMetaData(p, ang);
  for(size_t i = 0; i < size; ++i)
      std::cout << ang[i] << std::endl;
  return 0;
}

