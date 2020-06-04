#ifndef SCANPROJECTSERIALIZATION_HPP
#define SCANPROJECTSERIALIZATION_HPP

#include <string>

#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/descriptions/FeatureBase.hpp"
#include "lvr2/io/descriptions/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

void saveScanProject(const ScanProjectSchema& structure, const FileKernel& kernel, const ScanProjectPtr project)
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::ScanProjectIO>;

    MyScanProjectIO io(kernel, structure);
    io.saveScanProject(project);
}


ScanProjectPtr loadScanProject(const ScanProjectSchema& structure, const FileKernel& kernel)
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::ScanProjectIO>;

    MyScanProjectIO io(kernel, structure);
    ScanProjectPtr ptr = io.loadScanProject();
    return ptr;
}


// class ScanProjectWriter
// {
// public:
//     ScanProjectWriter() = delete;
//     ScanProjectWriter(const ScanProjectSchema& structure, const FileKernel& kernel);

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
//     ScanProjectSchema    m_root;
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