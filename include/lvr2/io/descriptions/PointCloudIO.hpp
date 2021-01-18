#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_POINTBUFFERIO_HPP
#define LVR2_IO_DESCRIPTIONS_POINTBUFFERIO_HPP

#include <boost/optional.hpp>

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"

// Dependencies
#include "ChannelIO.hpp"
#include "VariantChannelIO.hpp"

namespace lvr2 {

#define CONST_FUNC_ALIAS(X,Y) template<typename... Ts> \
    auto X(Ts&&... ts) const -> decltype(Y(std::forward<Ts>(ts)...)) \
    { \
        return Y(std::forward<Ts>(ts)...); \
    }

#define FUNC_ALIAS(X,Y) template<typename... Ts> \
    auto X(Ts&&... ts) -> decltype(Y(std::forward<Ts>(ts)...)) \
    { \
        return Y(std::forward<Ts>(ts)...); \
    }

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
template<typename FeatureBase>
class PointCloudIO 
{
public:

    void save(
        const std::string& group, 
        const std::string& name, 
        PointBufferPtr pcl
        ) const;


    void save(const std::string& groupandname, 
        PointBufferPtr pcl) const;

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
    PointBufferPtr loadPointCloud(const std::string& group, const std::string& container);

    PointBufferPtr loadPointCloud(const std::string& group);

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
        ReductionAlgorithmPtr reduction);
    
protected:

    /// Checks whether the indicated group contains point cloud data
    // How can we decide if no meta data is available?
    // We cannot
    bool isPointCloud(const std::string& group, 
        const std::string& name);

    /// Add access to feature base
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    /// Dependencies
    VariantChannelIO<FeatureBase>* m_vchannel_io = static_cast<VariantChannelIO<FeatureBase>*>(m_featureBase);

    /// Class ID
    static constexpr const char* ID = "PointCloudIO";

    /// Object ID
    static constexpr const char* OBJID = "PointBuffer";
};

template <typename FeatureBase>
struct FeatureConstruct<PointCloudIO, FeatureBase >
{
    // DEPS
    using deps = typename FeatureConstruct<VariantChannelIO, FeatureBase >::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<PointCloudIO>::type;
};

} // namespace lvr2 

#include "PointCloudIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_POINTBUFFERIO_HPP