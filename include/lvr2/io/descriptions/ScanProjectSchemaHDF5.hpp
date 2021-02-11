#ifndef SCANPROJETSCHEMA_HDF5_HPP
#define SCANPROJETSCHEMA_HDF5_HPP

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2
{

class ScanProjectSchemaHDF5 : public HDF5Schema
{
public:
    ScanProjectSchemaHDF5() : HDF5Schema() {};

    ~ScanProjectSchemaHDF5() = default;

    virtual Description scanProject() const;

    virtual Description position(
        const size_t& scanPosNo) const;

    virtual Description lidar(
        const Description& d_parent,
        const size_t& lidarNo) const;

    virtual Description camera(
        const Description& d_parent, 
        const size_t& camNo) const;
    
    virtual Description scan(
        const Description& d_parent, 
        const size_t& scanNo) const;
 
    virtual Description cameraImage(
        const Description& d_parent, 
        const size_t& cameraImageNo) const;

    virtual Description hyperspectralCamera(
        const Description& d_parent,
        const size_t& camNo) const;
};

} // namespace lvr2

#endif // SCANPROJETSCHEMA_HDF5_HPP_
