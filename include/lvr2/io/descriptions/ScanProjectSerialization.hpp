#ifndef SCANPROJECTSERIALIZATION_HPP
#define SCANPROJECTSERIALIZATION_HPP

#include <string>

#include "lvr2/io/descriptions/ScanProjectStructure.hpp"
#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/descriptions/FeatureBase.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

void saveScanProject(const scanProjectStructure& structure, const FileKernel& kernel, const ScanProjectPtr project)
{
    using ScanProjectIO = FeatureBase::AddFeatures<lvr2::ScanProjectIO>;
    ScanProjectIO io(kernel, description);
    io.save(project);
}

ScanProjectPtr loadScanProject(const scanProjectStructure& structure, const FileKernel& kernel)
{
    using ScanProjectIO = FeatureBase::AddFeatures<lvr2::ScanProjectIO>;
    ScanProjectIO io(kernel, description);
    return io.load(project);
}


// class ScanProjectWriter
// {
// public:
//     ScanProjectWriter() = delete;
//     ScanProjectWriter(const ScanProjectStructure& structure, const FileKernel& kernel);

//     virtual void saveScanProject(const ScanProjectPtr& ptr);

//     virtual void saveScanPosition(
//         const size_t& positionNo, const ScanPositionPtr& pos);

//     virtual void saveScan(
//         const size_t& positionNo, const size_t& scanNo, const ScanPtr& ptr);
    
//     virtual void saveScanCamera(
//         const size_t& positionNo, 
//         const size_t& camNo, const 
//         ScanCameraPtr ptr);

//     virtual void saveScanImage(
//         const size_t& positionNo, 
//         const size_t& camNo, const size_t imageNo, const ScanImagePtr ptr);

// protected:
//     ScanProjectStructure    m_root;
//     FileKernel              m_kernel;
// };

// class ScanProjectReader
// {
// public:
//     ScanProjectReader() = delete;
//     ScanProjectReader(const std::string& root, const FileKernel& kernel);
    
// }

} // namespace lvr2

#endif