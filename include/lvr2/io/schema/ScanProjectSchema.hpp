#ifndef SCANPROJECTSCHEMA
#define SCANPROJECTSCHEMA

#include <string>
#include <tuple>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include <yaml-cpp/yaml.h>

#include "lvr2/util/Logging.hpp"

namespace lvr2
{

using StringOptional = boost::optional<std::string>;

// struct Description
// {
//     // path to relative to project root: is built recursively
//     StringOptional groupName;
//     // if group contains data (relative to group)
//     StringOptional dataSetName;
//     // if group contains meta (relative to group)
//     StringOptional metaName;
// };

struct Description
{
    // data or group location
    StringOptional dataRoot;
    StringOptional data;

    StringOptional metaRoot;
    StringOptional meta;

};

std::ostream& operator<<(std::ostream& os, const Description& desc);

inline lvr2::Logger &operator<<(lvr2::Logger &log, const Description &desc)
{
    if(desc.dataRoot)
    {
        log << "[LVR Description] DataRoot: " << *desc.dataRoot << lvr2::endl;
    }

    if(desc.data)
    {
        log << "[LVR Description] Data: " << *desc.data << lvr2::endl;
    }

    if(desc.metaRoot)
    {
        log << "[LVR Description] Meta Root: " << *desc.metaRoot << lvr2::endl;
    }

    if(desc.meta)
    {
        log << "[LVR Description] Meta: " << *desc.meta << lvr2::endl;
    }
    
    return log;
}

class ScanProjectSchema
{
public:
    ScanProjectSchema() = default;

    virtual ~ScanProjectSchema() = default;

    virtual Description scanProject() const = 0;

    virtual Description position(
        const size_t& scanPosNo) const = 0;

    virtual Description lidar(
        const size_t& scanPosNo,
        const size_t& lidarNo) const = 0;

    virtual Description scan(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo) const = 0;

    virtual Description scanChannel(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo,
        const std::string& channelName) const = 0;

    virtual Description camera(
        const size_t& scanPosNo,
        const size_t& camNo) const = 0;

    virtual Description cameraImage(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& GroupNo,
        const size_t& imgNo) const = 0;

    virtual Description cameraImageGroup(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& GroupNo) const = 0;

    virtual Description hyperspectralCamera(
        const size_t& scanPosNo,
        const size_t& camNo) const = 0;

    virtual Description hyperspectralPanorama(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const = 0;

    virtual Description hyperspectralPanoramaPreview(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const = 0;

    virtual Description hyperspectralPanoramaChannel(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo,
        const size_t& channelNo) const = 0;
};

/// Marker interface for HDF5 schemas
class HDF5Schema : public ScanProjectSchema 
{
public:
    HDF5Schema() {}
};

/// Marker interface for HDF5 schemas
class LabelHDF5Schema : public HDF5Schema
{
public:
    LabelHDF5Schema() {}
};

/// Marker interface for directory schemas
class DirectorySchema : public ScanProjectSchema
{
public:
    DirectorySchema(const std::string& root) : m_rootPath(root) {}

protected:
    boost::filesystem::path m_rootPath;
};

using ScanProjectSchemaPtr = std::shared_ptr<ScanProjectSchema>;
using DirectorySchemaPtr = std::shared_ptr<DirectorySchema>;
using HDF5SchemaPtr = std::shared_ptr<HDF5Schema>;
using LabelHDF5SchemaPtr = std::shared_ptr<LabelHDF5Schema>;

} // namespace lvr2

#endif // SCANPROJECTSCHEMA
