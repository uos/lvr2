#ifndef DIRECTORYIO_HPP
#define DIRECTORYIO_HPP

#include "lvr2/io/descriptions/DirectoryKernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/FeatureBase.hpp"
#include "lvr2/io/descriptions/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

class DirectoryIO
{
public:
    DirectoryIO() = delete;
    DirectoryIO(DirectoryKernelPtr kernel, DirectorySchemaPtr schema) : m_kernel(kernel), m_schema(schema) {}

    void saveScanProject(ScanProjectPtr project);
    ScanProjectPtr loadScanProject();

private:
    DirectoryKernelPtr m_kernel;
    DirectorySchemaPtr m_schema;
};

using DirectoryIOPtr = std::shared_ptr<DirectoryIO>;

} // namespace lvr2

#endif