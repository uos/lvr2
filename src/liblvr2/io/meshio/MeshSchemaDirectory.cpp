#include "lvr2/io/meshio/MeshSchemaDirectory.hpp"
#include <iomanip>

namespace lvr2
{

Description MeshSchemaDirectory::mesh(std::string name) const
{
    Description d;
    d.dataRoot = "meshes/" + name;
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
    d.data = channel_name + ".ply";
    d.meta = channel_name + ".yaml";
    return d;
}

Description MeshSchemaDirectory::surface(std::string name, size_t surface_index) const
{
    auto d = mesh(name);
    auto sstr = std::stringstream();
    sstr << std::setw(8) << std::setfill('0') << surface_index;

    d.dataRoot = *d.dataRoot + "/surfaces/" + sstr.str();
    d.metaRoot = d.dataRoot;

    return d;
}

Description MeshSchemaDirectory::surfaceIndices(std::string name, size_t surface_index) const
{
    auto d = surface(name, surface_index);
    d.data = "indices";
    d.meta = "indices.yaml";
    return d;
}

Description MeshSchemaDirectory::textureCoordinates(std::string name, size_t surface_index) const
{
    auto d = surface(name, surface_index);
    d.data = "texture_coordinates";
    d.meta = "texture_coordinates.yaml";
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
    d.data = layer_name + ".jpg";
    d.metaRoot = d.dataRoot;
    d.meta = layer_name + ".yaml";
    return d;
}

} // namespace lvr2