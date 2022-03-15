#include "lvr2/io/schema/MeshSchemaHDF5.hpp"
#include <iomanip>

namespace lvr2
{

Description MeshSchemaHDF5::mesh(std::string name) const
{
    Description d;
    d.dataRoot = "meshes/" + name;
    d.data = "";
    d.metaRoot = d.dataRoot;
    d.meta = "";

    return d;
}

Description MeshSchemaHDF5::vertices(std::string name) const
{
    auto d = mesh(name);
    d.dataRoot = *d.dataRoot + "/vertices";
    d.metaRoot = d.dataRoot;
    return d;
}

Description MeshSchemaHDF5::vertexChannel(std::string mesh_name, std::string channel_name) const
{
    auto d = vertices(mesh_name);
    d.data = channel_name;
    d.meta = d.data;
    return d;
}

Description MeshSchemaHDF5::faces(std::string name) const
{
    auto d = mesh(name);

    d.dataRoot = *d.dataRoot + "/faces";
    d.metaRoot = d.dataRoot;

    return d;
}

Description MeshSchemaHDF5::faceIndices(std::string name) const
{
    auto d = faces(name);

    d.data = "indices";
    d.meta = d.data;

    return d;
}

Description MeshSchemaHDF5::faceNormals(std::string name) const
{
    auto d = faces(name);

    d.data = "normals";
    d.meta = d.data;

    return d;
}

Description MeshSchemaHDF5::faceColors(std::string name) const
{
    auto d = faces(name);

    d.data = "colors";
    d.meta = d.data;

    return d;
}

Description MeshSchemaHDF5::faceMaterialIndices(std::string name) const
{
    auto d = faces(name);

    d.data = "materials";
    d.meta = d.data;

    return d;
}

Description MeshSchemaHDF5::surfaces(std::string name) const
{
    auto d = mesh(name);

    d.dataRoot = *d.dataRoot + "/surfaces";
    d.metaRoot = d.dataRoot;

    return d;
}

Description MeshSchemaHDF5::surfaceCombinedFaceIndices(std::string name) const
{
    auto d = surfaces(name);
    d.data = "combined_face_indices";
    d.meta = d.data;
    return d;
}

Description MeshSchemaHDF5::surfaceFaceIndexRanges(std::string name) const
{
    auto d = surfaces(name);
    d.data = "face_index_ranges";
    d.meta = d.data;
    return d;
}

Description MeshSchemaHDF5::surfaceMaterialIndices(std::string name) const
{
    auto d = surfaces(name);
    d.data = "material_indices";
    d.meta = d.data;
    return d;
}

Description MeshSchemaHDF5::material(std::string name, size_t material_index) const
{
    auto d = mesh(name);
    auto sstr = std::stringstream();
    sstr << std::setw(8) << std::setfill('0') << material_index;

    d.dataRoot  = *d.dataRoot + "/materials/" + sstr.str();
    d.metaRoot  = d.dataRoot;
    d.meta      = "";

    return d;
}

Description MeshSchemaHDF5::texture(std::string name, size_t material_index, std::string layer_name) const
{
    auto d = material(name, material_index);
    d.dataRoot = *d.dataRoot + "/textures";
    d.data = layer_name;
    d.metaRoot = d.dataRoot;
    d.meta = d.data;
    return d;
}

} // namespace lvr2