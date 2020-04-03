#ifndef HDF5METADESCRIPTION_V2
#define HDF5METADESCRIPTION_V2

#include "lvr2/io/descriptions/HDF5MetaDescriptionBase.hpp"

namespace lvr2
{

class HDF5MetaDescriptionV2 : public HDF5MetaDescriptionBase
{
public:
    virtual YAML::Node scanPosition() const override;
    virtual YAML::Node scan() const override;
    virtual YAML::Node scanCamera() const override;
    virtual YAML::Node scanProject() const override;
    virtual YAML::Node scanImage() const override;
    virtual YAML::Node hyperspectralCamera() const override;
    virtual YAML::Node hyperspectralPanoramaChannel() const override;
protected:
    template<typename T> 
    YAML::Node defaultNode() const
    {
        // Setup default object
        T t;

        // Build and return corresponding node
        YAML::Node node;
        node = t;
        return node;
    }
};

} // namespace lvr2

#endif