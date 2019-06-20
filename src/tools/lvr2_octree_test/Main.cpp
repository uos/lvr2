#include "lvr2/display/PointOctree.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/IOUtils.hpp"


using namespace lvr2;
using Vec = lvr2::BaseVector<float>;

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    return 0;
  }

  std::cout << argv[1] << std::endl;
  ModelPtr model = ModelFactory::readModel(std::string(argv[1]));

  // Parse loaded data
  if (!model)
  {
    std::cout << "IO Error: Unable to parse " << std::endl;
    return 0;
  }
  
  PointOctree<Vec> oct = PointOctree<Vec>(model->m_pointCloud, 5);

  return 0;
}

