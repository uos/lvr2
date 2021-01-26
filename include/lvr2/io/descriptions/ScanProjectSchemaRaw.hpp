#ifndef SCANPROJETSCHEMA_RAW_HPP_
#define SCANPROJETSCHEMA_RAW_HPP_

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2
{

class ScanProjectSchemaRaw : public DirectorySchema
{
public:
    ScanProjectSchemaRaw(std::string& rootDir) : DirectorySchema(rootDir) {};

    ~ScanProjectSchemaRaw() = default;

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
};

} // namespace lvr2

#endif // SCANPROJETSCHEMA_RAW_HPP_