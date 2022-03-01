#pragma once

#include "lvr2/io/meshio/MeshSchema.hpp"

namespace lvr2
{

class MeshSchemaHDF5 : public MeshSchema
{
    public:
    MeshSchemaHDF5() = default;

    virtual ~MeshSchemaHDF5() = default;

    virtual Description mesh(std::string name) const;

    virtual Description vertices(std::string name) const;

    virtual Description vertexChannel(std::string mesh_name, std::string channel_name) const;

    virtual Description surfaces(std::string name) const;

    virtual Description faces(std::string name) const;

    virtual Description faceIndices(std::string name) const;

    virtual Description faceNormals(std::string name) const;

    virtual Description faceColors(std::string name) const;

    virtual Description faceMaterialIndices(std::string name) const;

    virtual Description surfaceCombinedFaceIndices(std::string name) const;

    virtual Description surfaceFaceIndexRanges(std::string name) const;

    virtual Description surfaceMaterialIndices(std::string name) const;

    virtual Description material(std::string name, size_t material_index) const;

    virtual Description texture(std::string name, size_t material_index, std::string layer_name) const;
};

using MeshSchemaHDF5Ptr = std::shared_ptr<MeshSchemaHDF5>;

} // namespace lvr2