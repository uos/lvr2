#pragma once

#ifndef SCANPROJECTIO
#define SCANPROJECTIO

#include "lvr2/io/scanio/yaml/ScanProject.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"

// Dependencies
#include "MetaIO.hpp"
#include "ScanPositionIO.hpp"

namespace lvr2
{

namespace scanio
{

/**
 * @class ScanProjectIO
 * @brief Hdf5IO Feature for handling ScanProject related IO
 *
 * This Feature of the Hdf5IO handles the IO of a ScanProject object.
 *
 * Example:
 * @code
 * MyHdf5IO io;
 * PointBufferPtr pointcloud, pointcloud_in;
 *
 * // writing
 * io.open("test.h5");
 * io.save("apointcloud", pointcloud);
 *
 * // reading
 * pointcloud_in = io.loadPointCloud("apointcloud");
 *
 * @endcode
 *
 * Generates attributes at hdf5 group:
 * - IO: ScanProjectIO
 * - CLASS: ScanProject
 *
 * Dependencies:
 * - ScanPositionIO
 *
 */
template <typename FeatureBase>
class ScanProjectIO
{
  public:
    void save(ScanProjectPtr scanProject) const;
    void saveScanProject(ScanProjectPtr scanProject) const;
   
    ScanProjectPtr load() const;
    ScanProjectPtr loadScanProject() const;
    ScanProjectPtr loadScanProject(ReductionAlgorithmPtr reduction) const;

    boost::optional<YAML::Node> loadMeta() const;

    
  protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    
    // dependencies
    MetaIO<FeatureBase>* m_metaIO = 
        static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    ScanPositionIO<FeatureBase>* m_scanPositionIO =
        static_cast<ScanPositionIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanProjectIO";
    static constexpr const char* OBJID = "ScanProject";
};

} // namespace scanio

template <typename FeatureBase>
struct FeatureConstruct<lvr2::scanio::ScanProjectIO, FeatureBase>
{
    // DEPS
    //
    using dep1 = typename FeatureConstruct<lvr2::scanio::MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::ScanPositionIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;


    // add the feature itself
    using type = typename deps::template add_features<lvr2::scanio::ScanProjectIO>::type;
};

} // namespace lvr2

#include "ScanProjectIO.tcc"

#endif // SCANPROJECTIO
