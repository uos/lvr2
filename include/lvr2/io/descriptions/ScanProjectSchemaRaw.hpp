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
    virtual Description position(const size_t &scanPosNo) const;
    virtual Description scan(const size_t &scanPosNo, const size_t &scanNo) const;
    virtual Description scan(const std::string &scanPositionPath, const size_t &scanNo) const;
    virtual Description waveform(const size_t &scanPosNo, const size_t &scanNo) const;
    virtual Description waveform(const std::string &scanPositionPath, const size_t &scanNo) const;

    virtual Description scanCamera(const size_t &scanPositionNo, const size_t &camNo) const;
    virtual Description scanCamera(const std::string &scanPositionPath, const size_t &camNo) const;

    virtual Description scanImage(
        const size_t &scanPosNo, const size_t &scanNo,
        const size_t &scanCameraNo, const size_t &scanImageNo) const;

    virtual Description scanImage(
        const std::string &scanImagePath, const size_t &scanImageNo) const;

private:
    //boost::filesystem::path m_rootPath;
};

} // namespace lvr2

#endif // SCANPROJETSCHEMA_RAW_HPP_