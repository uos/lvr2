#pragma once

#ifndef POINTCLOUDIO
#define POINTCLOUDIO

#include <boost/optional.hpp>

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/baseio/VariantChannelIO.hpp"
#include "lvr2/io/baseio/ChannelIO.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"

using lvr2::baseio::VariantChannelIO;

namespace lvr2 
{

namespace scanio
{

/**
 * @class PointCloudIO 
 * @brief Hdf5IO Feature for handling PointBuffer related IO
 * 
 * This Feature of the Hdf5IO handles the IO of a PointBuffer object.
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
 * Generates attributes:
 * - IO: PointCloudIO
 * - CLASS: PointBuffer
 * 
 * Dependencies:
 * - VariantChannelIO
 * 
 */
template<typename BaseIO>
class PointCloudIO 
{
public:
    void save(
        const std::string& group, 
        const std::string& name, 
        PointBufferPtr pcl
        ) const;

    void save(
        const std::string& groupandname, 
        PointBufferPtr pcl) const;

    PointBufferPtr load(
        const std::string& group, 
        const std::string& container) const;

    PointBufferPtr load(
        const std::string& group) const;

    PointBufferPtr load(
        const std::string& group,
        const std::string& container, 
        ReductionAlgorithmPtr reduction) const;

    /**
     * @brief Save a point buffer at the position defined by \ref group and \ref container
     * 
     * @param group             Group with the point cloud data 
     * @param container         Container of the point cloud data
     * @param buffer            Point cloud data
     */
    void savePointCloud(
        const std::string& group, 
        const std::string& name, 
        PointBufferPtr pcl) const;

    void savePointCloud(
        const std::string& groupandname,
        PointBufferPtr pcl) const;

    /**
     * @brief  Loads a point cloud
     * 
     * @param group             Group with the point cloud data 
     * @param container         Container of the point cloud data
     * @return PointBufferPtr   A point buffer containing the point 
     *                          cloud data stored at the position 
     *                          defined by \ref group and \ref container
     */
    PointBufferPtr loadPointCloud(
        const std::string& group, 
        const std::string& container) const;

    PointBufferPtr loadPointCloud(
        const std::string& group) const;

    /**
     * @brief Loads a reduced version of a point cloud
     * 
     * @param group             Group with the point cloud data 
     * @param container         Container of the point cloud data
     * @param reduction         A reduction object that is used to generate the reduced data
     * @return PointBufferPtr   A point buffer containing a reduced version of the point 
     *                          cloud data stored at the position 
     *                          defined by \ref group and \ref container
     */
    PointBufferPtr loadPointCloud(
        const std::string& group,
        const std::string& container, 
        ReductionAlgorithmPtr reduction) const;
    
protected:

    /// Add access to feature base
    BaseIO* m_baseIO = static_cast<BaseIO*>(this);

    /// Dependencies
    VariantChannelIO<BaseIO>* m_vchannel_io = static_cast<VariantChannelIO<BaseIO>*>(m_baseIO);

    /// Class ID
    static constexpr const char* ID = "PointCloudIO";

    /// Object ID
    static constexpr const char* OBJID = "PointBuffer";
};

} // namespace scanio

template <typename T>
struct FeatureConstruct<lvr2::scanio::PointCloudIO, T>
{
    // DEPS
    using deps = typename FeatureConstruct<lvr2::baseio::VariantChannelIO, T>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::PointCloudIO>::type;
};

} // namespace lvr2 

#include "PointCloudIO.tcc"

#endif // POINTCLOUDIO
