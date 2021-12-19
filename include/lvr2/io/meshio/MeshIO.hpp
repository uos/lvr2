#pragma once

#include "MeshSchema.hpp"
#include <lvr2/io/scanio/FileKernel.hpp>
#include <lvr2/io/meshio/MeshSchemaDirectory.hpp>
#include <lvr2/io/meshio/MeshSchemaHDF5.hpp>

namespace lvr2
{

class MeshIO
{
private:
    FileKernelPtr   m_kernel;
    MeshSchemaPtr   m_schema;

public:
    MeshIO(FileKernelPtr kernel, MeshSchemaPtr schema): 
        m_kernel(kernel), 
        m_schema(schema)
        {};

    void saveMesh(
        std::string mesh_name, 
        MeshBufferPtr mesh
        );

}; 

using MeshIOPtr = std::shared_ptr<MeshIO>;

} // namespace lvr2

#include "MeshIO.tcc"