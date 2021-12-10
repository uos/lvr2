#pragma once

#include "lvr2/io/meshio/MeshSchema.hpp"

namespace lvr2
{

class MeshSchemaDirectory : public MeshSchema
{
    public:
    MeshSchemaDirectory() = default;

    virtual ~MeshSchemaDirectory() = default;

    virtual Description mesh(std::string name) const;

    virtual Description vertices(std::string name) const;

    virtual Description vertexChannel(std::string mesh_name, std::string channel_name) const;

    virtual Description surface(std::string name, size_t surface_index) const;

    virtual Description surfaceIndices(std::string name, size_t surface_index) const;

    virtual Description textureCoordinates(std::string name, size_t surface_index) const;

    virtual Description material(std::string name, size_t material_index) const;

    virtual Description texture(std::string name, size_t material_index, std::string layer_name) const;
};


} // namespace lvr2