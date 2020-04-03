#ifndef HDF5_META_DESCRIPTION
#define HDF5_META_DESCRIPTION

#include <yaml-cpp/yaml.h>

namespace lvr2
{

class HDF5MetaDescriptionBase
{
public:
    HDF5MetaDescriptionBase() = default;
    virtual ~HDF5MetaDescriptionBase() = default;

    virtual YAML::Node scanPosition() const = 0;
    virtual YAML::Node scan() const = 0;
    virtual YAML::Node scanCamera() const = 0;
    virtual YAML::Node scanProject() const = 0;
    virtual YAML::Node scanImage() const = 0;
    virtual YAML::Node hyperspectralCamera() const = 0;
    virtual YAML::Node hyperspectralPanoramaChannel() const = 0;
};

} // namespace lvr2

#endif