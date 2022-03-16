#pragma once

#ifndef POLYGONIO
#define POLYGONIO

#include <boost/optional.hpp>

#include "lvr2/io/scanio/Polygon.hpp"

// Dependencies
#include "lvr2/io/scanio/ChannelIO.hpp"
#include "lvr2/io/scanio/VariantChannelIO.hpp"

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
template<typename BaseIO>
class PolygonIO
{
public:
    void savePolygon(const std::string& group, const std::string& name, const PolygonPtr& buffer);
    PolygonPtr loadPolygon(const std::string& group, const std::string& container);
    
protected:

    bool isPolygon(const std::string& group);

    BaseIO* m_baseIO = static_cast<lvr2::baseio::BaseIO*>(this);
    // dependencies
    VariantChannelIO<BaseIO>* m_vchannel_io = static_cast<VariantChannelIO<BaseIO>*>(m_file_access);

    static constexpr const char* ID = "PolygonIO";
    static constexpr const char* OBJID = "Polygon";
};

template<typename T#pragma endregion>
struct FeatureConstruct<PolygonIO, T> {
    
    // DEPS
    using deps = typename FeatureConstruct<VariantChannelIO, T>::type;

    // add actual feature
    using type = typename deps::template add_features<PolygonIO>::type;
     
};

} // namespace lvr2 

#include "PolygonIO.tcc"


#endif // POLYGONIO
