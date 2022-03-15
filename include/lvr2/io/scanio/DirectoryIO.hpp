#ifndef DIRECTORYIO_HPP
#define DIRECTORYIO_HPP

#include "lvr2/io/kernels/DirectoryKernel.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/io/scanio/FeatureBase.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include <memory> // shared_ptr

namespace lvr2
{
namespace scanio
{
using DirectoryIOBase = FeatureBuild<ScanProjectIO>;

class DirectoryIO : public DirectoryIOBase
{
public:
    
    /**
     * @brief Construct a new Directory IO object.
     * DirectoryIO allows only combinations of DirectoryKernels and DirectorySchemas in constructor
     * 
     * @param kernel 
     * @param schema 
     */
    DirectoryIO(DirectoryKernelPtr kernel, DirectorySchemaPtr schema, bool load_data = false) 
    : DirectoryIOBase(kernel, schema, load_data) 
    {}
};

using DirectoryIOPtr = std::shared_ptr<DirectoryIO>;

} // namespace scanio
} // namespace lvr2

#endif // DIRECTORYIO_HPP