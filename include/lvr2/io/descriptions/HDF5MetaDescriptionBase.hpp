#ifndef HDF5_META_DESCRIPTION
#define HDF5_META_DESCRIPTION

#include <yaml-cpp/yaml.h>
#include <hdf5_hl.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include <tuple>


namespace lvr2
{

class HDF5MetaDescriptionBase
{
public:
    HDF5MetaDescriptionBase() = default;
    virtual ~HDF5MetaDescriptionBase() = default;

    // virtual YAML::Node hyperspectralCamera(const HighFive::Group& g) const = 0;
    // virtual YAML::Node hyperspectralPanoramaChannel(const HighFive::Group& g) const = 0;
    virtual YAML::Node scan(const HighFive::Group& g) const = 0;
    virtual YAML::Node scanPosition(const HighFive::Group& g) const = 0;
    virtual YAML::Node scanProject(const HighFive::Group& g) const = 0;
    virtual YAML::Node lidar(const HighFive::Group& g) const = 0;
    virtual YAML::Node camera(const HighFive::Group& g) const = 0;
    virtual YAML::Node cameraImage(const HighFive::Group& g) const = 0;

    virtual YAML::Node channel(const HighFive::DataSet& d) const = 0;

    // virtual void saveHyperspectralCamera(
    //   HighFive::Group& g, 
    //   const YAML::Node& n) const = 0;

    // virtual void saveHyperspectralPanoramaChannel(
    //   HighFive::Group& g, 
    //   const YAML::Node& n) const = 0;

    virtual void saveScanPosition(
      HighFive::Group& g, 
      const YAML::Node& n) const = 0;

    virtual void saveScanProject(
      HighFive::Group& g, 
      const YAML::Node& n) const = 0;

    virtual void saveLIDAR(
      HighFive::Group& g,
      const YAML::Node& n
    ) const = 0;

    virtual void saveCamera(
      HighFive::Group& g, 
      const YAML::Node& n) const = 0;

    virtual void saveScan(
      HighFive::Group& g, 
      const YAML::Node& h) const = 0;

    virtual void saveCameraImage(
      HighFive::Group& g, 
      const YAML::Node& n) const = 0;

    virtual void saveChannel(
      HighFive::DataSet &d,
      const YAML::Node &n) const = 0;

};

} // namespace lvr2

#endif // LVR2_IO_DESCRIPTIONS_HDF5METADESCRIPTION_BASE_HPP