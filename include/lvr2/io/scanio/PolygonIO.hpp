#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_POLYGONIO_HPP
#define LVR2_IO_DESCRIPTIONS_POLYGONIO_HPP

#include <boost/optional.hpp>

#include "lvr2/io/Polygon.hpp"

// Dependencies
#include "ChannelIO.hpp"
#include "VariantChannelIO.hpp"

namespace lvr2 {

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
class PolygonIO
{
public:
    void savePolygon(const std::string& group, const std::string& name, const PolygonPtr& buffer);
    PolygonPtr loadPolygon(const std::string& group, const std::string& container);
    
protected:

    bool isPolygon(const std::string& group);

    FeatureBase* m_FeatureBase = static_cast<FeatureBase*>(this);
    // dependencies
    VariantChannelIO<FeatureBase>* m_vchannel_io = static_cast<VariantChannelIO<FeatureBase>*>(m_file_access);

    static constexpr const char* ID = "PolygonIO";
    static constexpr const char* OBJID = "Polygon";
};

template<typename FeatureBase>
struct FeatureConstruct<PolygonIO, FeatureBase> {
    
    // DEPS
    using deps = typename FeatureConstruct<VariantChannelIO, FeatureBase>::type;

    // add actual feature
    using type = typename deps::template add_features<PolygonIO>::type;
     
};

} // namespace lvr2 

#include "PolygonIO.tcc"


#endif // LVR2_IO_DESCRIPTIONS_POLYGONIO_HPP