#ifndef MESHSCHEMADIRECTORY
#define MESHSCHEMADIRECTORY

#include "lvr2/io/schema/MeshSchema.hpp"

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

using MeshSchemaDirectoryPtr = std::shared_ptr<MeshSchemaDirectory>;

} // namespace lvr2


#endif // MESHSCHEMADIRECTORY
