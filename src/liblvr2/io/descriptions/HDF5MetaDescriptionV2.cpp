#include "lvr2/io/descriptions/HDF5MetaDescriptionV2.hpp"
#include "lvr2/io/yaml/MetaNodeDescriptions.hpp"

namespace lvr2
{

YAML::Node HDF5MetaDescriptionV2::scanPosition() const
{
    return defaultNode<ScanPosition>();
}

YAML::Node HDF5MetaDescriptionV2::scan()  const
{
    return defaultNode<Scan>();
}

YAML::Node HDF5MetaDescriptionV2::scanCamera()  const
{
    return defaultNode<ScanCamera>();
}

YAML::Node HDF5MetaDescriptionV2::scanProject() const
{
    return defaultNode<ScanProject>();
}

YAML::Node HDF5MetaDescriptionV2::scanImage() const
{
    return defaultNode<ScanImage>();
}

YAML::Node HDF5MetaDescriptionV2::hyperspectralCamera()  const
{
     return defaultNode<HyperspectralCamera>();
}

YAML::Node HDF5MetaDescriptionV2::hyperspectralPanoramaChannel()  const
{
     return defaultNode<HyperspectralPanoramaChannel>();
}

} // namespace lvr2