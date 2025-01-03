#include "lvr2/io/schema/MeshSchemaDirectory.hpp"
#include <iomanip>

namespace lvr2
{

Description MeshSchemaDirectory::mesh(std::string name) const
{
    Description d;
    d.dataRoot = "meshes/" + name;
    d.data = "";
    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";

    return d;
}

Description MeshSchemaDirectory::vertices(std::string name) const
{
    auto d = mesh(name);
    d.dataRoot = *d.dataRoot + "/vertices";
    d.metaRoot = d.dataRoot;
    return d;
}

Description MeshSchemaDirectory::vertexChannel(std::string mesh_name, std::string channel_name) const
{
    auto d = vertices(mesh_name);
    d.data = channel_name + ".data";
    d.meta = channel_name + ".yaml";
    return d;
}
Description MeshSchemaDirectory::faces(std::string name) const
{
    auto d = mesh(name);

    d.dataRoot = *d.dataRoot + "/faces";
    d.metaRoot = d.dataRoot;

    return d;
}

Description MeshSchemaDirectory::faceIndices(std::string name) const
{
    auto d = faces(name);

    d.data = "indices.data";
    d.meta = "indices.yaml";

    return d;
}

Description MeshSchemaDirectory::faceNormals(std::string name) const
{
    auto d = faces(name);

    d.data = "normals.data";
    d.meta = "normals.yaml";

    return d;
}

Description MeshSchemaDirectory::faceColors(std::string name) const
{
    auto d = faces(name);

    d.data = "colors.data";
    d.meta = "colors.yaml";

    return d;
}

Description MeshSchemaDirectory::faceMaterialIndices(std::string name) const
{
    auto d = faces(name);

    d.data = "materials.data";
    d.meta = "materials.yaml";

    return d;
}

Description MeshSchemaDirectory::surfaces(std::string name) const
{
    auto d = mesh(name);

    d.dataRoot = *d.dataRoot + "/surfaces";
    d.metaRoot = d.dataRoot;

    return d;
}

Description MeshSchemaDirectory::surfaceCombinedFaceIndices(std::string name) const
{
    auto d = surfaces(name);
    d.data = "combined_face_indices.data";
    d.meta = "combined_face_indices.yaml";
    return d;
}

Description MeshSchemaDirectory::surfaceFaceIndexRanges(std::string name) const
{
    auto d = surfaces(name);
    d.data = "face_index_ranges.data";
    d.meta = "face_index_ranges.yaml";
    return d;
}

Description MeshSchemaDirectory::surfaceMaterialIndices(std::string name) const
{
    auto d = surfaces(name);
    d.data = "material_indices.data";
    d.meta = "material_indices.yaml";
    return d;
}

Description MeshSchemaDirectory::material(std::string name, size_t material_index) const
{
    auto d = mesh(name);
    auto sstr = std::stringstream();
    sstr << std::setw(8) << std::setfill('0') << material_index;

    d.dataRoot  = *d.dataRoot + "/materials/" + sstr.str();
    d.metaRoot  = d.dataRoot;
    d.meta      = "meta.yaml";

    return d;
}

Description MeshSchemaDirectory::texture(std::string name, size_t material_index, std::string layer_name) const
{
    auto d = material(name, material_index);
    d.dataRoot = *d.dataRoot + "/textures";
    d.data = layer_name + ".data"; // TODO: Change to .jpg when kernel->saveImage is used
    d.metaRoot = d.dataRoot;
    d.meta = layer_name + ".yaml";
    return d;
}

} // namespace lvr2