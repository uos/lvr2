#pragma once
#include <lvr2/io/scanio/ScanProjectSchema.hpp> // include to get Description

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

    virtual Description surfaces(std::string name) const = 0;

    virtual Description faces(std::string name) const = 0;

    virtual Description faceIndices(std::string name) const = 0;

    virtual Description faceNormals(std::string name) const = 0;

    virtual Description faceColors(std::string name) const = 0;

    virtual Description faceMaterialIndices(std::string name) const = 0;

    virtual Description surfaceCombinedFaceIndices(std::string name) const = 0;

    virtual Description surfaceFaceIndexRanges(std::string name) const = 0;

    virtual Description surfaceMaterialIndices(std::string name) const = 0;

    virtual Description material(std::string name, size_t material_index) const = 0;

    virtual Description texture(std::string name, size_t material_index, std::string layer_name) const = 0;

};

using MeshSchemaPtr = std::shared_ptr<MeshSchema>;

} // namespace lvr2