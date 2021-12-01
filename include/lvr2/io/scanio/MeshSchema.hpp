#pragma once
#include <lvr2/io/scanio/ScanProjectSchema.hpp>

namespace lvr2
{
class MeshSchema
{
public:
    MeshSchema() = default;

    virtual ~MeshSchema() = default;

    virtual Description mesh(std::string name) const = 0;

    virtual Description vertices(std::string name) const = 0;

    virtual Description vertexChannel(std::string mesh_name, std::string channel_name) const = 0;

    virtual Description surface(std::string name, size_t surface_index) const = 0;

    virtual Description surfaceIndices(std::string name, size_t surface_index) const = 0;

    virtual Description textureCoordinates(std::string name, size_t surface_index) const = 0;

    virtual Description material(std::string name, size_t material_index) const = 0;

    virtual Description texture(std::string name, size_t material_index, std::string layer_name) const = 0;

};
} // namespace lvr2