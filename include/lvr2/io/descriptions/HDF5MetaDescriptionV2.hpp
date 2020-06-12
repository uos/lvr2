#ifndef HDF5METADESCRIPTION_V2
#define HDF5METADESCRIPTION_V2

#include "lvr2/io/descriptions/HDF5MetaDescriptionBase.hpp"

namespace lvr2
{

class HDF5MetaDescriptionV2 : public HDF5MetaDescriptionBase
{
public:
    virtual YAML::Node hyperspectralCamera(const HighFive::Group& g) const override;
    virtual YAML::Node hyperspectralPanoramaChannel(const HighFive::Group& g) const override;
    virtual YAML::Node scan(const HighFive::Group& g) const override;
    virtual YAML::Node scanPosition(const HighFive::Group& g) const override;
    virtual YAML::Node scanProject(const HighFive::Group& g) const override;
    virtual YAML::Node scanCamera(const HighFive::Group& g) const override;
    virtual YAML::Node scanImage(const HighFive::Group& g) const override;

    virtual void saveHyperspectralCamera(
      HighFive::Group& g, 
      const YAML::Node& n) const override;

    virtual void saveHyperspectralPanoramaChannel(
      HighFive::Group& g, 
      const YAML::Node& n) const override;

    virtual void saveScan(
      HighFive::Group& g, 
      const YAML::Node& n) const override;

    virtual void saveScanPosition(
      HighFive::Group& g, 
      const YAML::Node& n) const override;

    virtual void saveScanProject(
      HighFive::Group& g, 
      const YAML::Node& n) const override;

    virtual void saveScanCamera(
      HighFive::Group& g, 
      const YAML::Node& n) const override;

    virtual void saveScanImage(
      HighFive::Group& g, 
      const YAML::Node& n) const override;
protected:
    
};

} // namespace lvr2

#endif