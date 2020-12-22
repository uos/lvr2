#ifndef DIRECTORYIO_HPP
#define DIRECTORYIO_HPP

#include "lvr2/io/descriptions/DirectoryKernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/FeatureBase.hpp"
#include "lvr2/io/descriptions/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include <memory> // shared_ptr

namespace lvr2
{

using DirectoryIOBase = FeatureBuild<ScanProjectIO>;

class DirectoryIO : public DirectoryIOBase
{
public:
<<<<<<< HEAD
    
    /**
     * @brief Construct a new Directory IO object.
     * DirectoryIO allows only combinations of DirectoryKernels and DirectorySchemas in constructor
     * 
     * @param kernel 
     * @param schema 
     */
    DirectoryIO(DirectoryKernelPtr kernel, DirectorySchemaPtr schema) 
    : DirectoryIOBase(kernel, schema) 
    {}
=======
    DirectoryIO() = delete;
    DirectoryIO(DirectoryKernelPtr kernel, DirectorySchemaPtr schema) : m_kernel(kernel), m_schema(schema) {}

    void saveScanProject(ScanProjectPtr project);
    ScanProjectPtr loadScanProject(bool lazy = false);

private:
    DirectoryKernelPtr m_kernel;
    DirectorySchemaPtr m_schema;
>>>>>>> feature/Lazy-Loading
};

using DirectoryIOPtr = std::shared_ptr<DirectoryIO>;

} // namespace lvr2

#endif // DIRECTORYIO_HPP