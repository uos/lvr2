#pragma once

#include "MeshSchema.hpp"
#include <lvr2/io/scanio/FileKernel.hpp>

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

    template <typename ClusterHandleT, typename MaterialHandleT>
    void MeshIO::saveMesh(
        std::string mesh_name, 
        MeshBufferPtr mesh, 
        ClusterBiMap<ClusterHandleT> clusters,
        MaterializerResult<MaterialHandleT> materials);

}; 

#include "MeshIO.tcc"

} // namespace lvr2