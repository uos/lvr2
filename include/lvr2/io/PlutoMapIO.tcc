/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/**
* @file      PlutoMapIO.tcc
**/



#include <hdf5_hl.h>

namespace lvr2
{

inline PlutoMapIO::PlutoMapIO(string filename)
    : m_file(filename, hf::File::ReadOnly | hf::File::Excl)
{

}

inline PlutoMapIO::PlutoMapIO(
    string filename,
    const vector<float>& vertices,
    const vector<uint32_t>& face_ids
)
    : m_file(filename, hf::File::ReadWrite | hf::File::Create | hf::File::Truncate)
{
    // Create top level groups
    m_geometryGroup = m_file.createGroup("/geometry");
    m_attributesGroup = m_file.createGroup("/attributes");
    m_clusterSetsGroup = m_file.createGroup("/clustersets");
    m_texturesGroup = m_file.createGroup("/textures");
    m_labelsGroup = m_file.createGroup("/labels");

    // Create geometry data sets
    m_geometryGroup
        .createDataSet<float>("vertices", hf::DataSpace::From(vertices))
        .write(vertices);
    m_geometryGroup
        .createDataSet<uint32_t>("faces", hf::DataSpace::From(face_ids))
        .write(face_ids);
}

inline PlutoMapIO::~PlutoMapIO()
{
    H5Gclose(m_geometryGroup.getId());
    H5Gclose(m_attributesGroup.getId());
    H5Gclose(m_clusterSetsGroup.getId());
    H5Gclose(m_texturesGroup.getId());
    H5Gclose(m_labelsGroup.getId());

    H5Fclose(m_file.getId());
}


inline vector<float> PlutoMapIO::getVertices()
{
    vector<float> vertices;
    m_geometryGroup.getDataSet("vertices")
        .read(vertices);

    return vertices;
}

inline vector<uint32_t> PlutoMapIO::getFaceIds()
{
    vector<uint32_t> faceIds;
    m_geometryGroup.getDataSet("faces")
        .read(faceIds);

    return faceIds;
}

inline vector<float> PlutoMapIO::getVertexNormals()
{
    vector<float> normals;
    m_attributesGroup.getDataSet("normals")
        .read(normals);

    return normals;
}

inline vector<uint8_t> PlutoMapIO::getVertexColors()
{
    vector<uint8_t> rgbColors;
    m_attributesGroup.getDataSet("rgb_colors")
        .read(rgbColors);

    return rgbColors;
}

inline vector<PlutoMapImage> PlutoMapIO::getTextures()
{
    vector<PlutoMapImage> textures;

    const hf::Group& imagesGroup = m_texturesGroup.getGroup("images");
    for (auto setName: imagesGroup.listObjectNames())
    {
        textures.push_back(getImage(imagesGroup, setName));
    }

    return textures;
}

inline vector<PlutoMapMaterial> PlutoMapIO::getMaterials()
{
    vector<PlutoMapMaterial> materials;

    m_texturesGroup.getDataSet("materials")
        .read(materials);

    return materials;
}

inline vector<uint32_t> PlutoMapIO::getMaterialFaceIndices()
{
    vector<uint32_t> matFaceIndices;

    m_texturesGroup.getDataSet("mat_face_indices")
        .read(matFaceIndices);

    return matFaceIndices;
}

inline vector<float> PlutoMapIO::getVertexTextureCoords()
{
    vector<float> coords;

    m_texturesGroup.getDataSet("coords")
        .read(coords);

    return coords;
}

inline vector<string> PlutoMapIO::getLabelGroups()
{
    return m_labelsGroup.listObjectNames();
}

inline vector<string> PlutoMapIO::getAllLabelsOfGroup(string groupName)
{
    if (!m_labelsGroup.exist(groupName))
    {
        return vector<string>();
    }

    return m_labelsGroup.getGroup(groupName).listObjectNames();
}

inline vector<uint32_t> PlutoMapIO::getFaceIdsOfLabel(string groupName, string labelName)
{
    vector<uint32_t> faceIds;

    if (!m_labelsGroup.exist(groupName))
    {
        return faceIds;
    }

    auto lg = m_labelsGroup.getGroup(groupName);

    if (!lg.exist(labelName))
    {
        return faceIds;
    }

    lg.getDataSet(labelName).read(faceIds);

    return faceIds;
}

inline PlutoMapImage PlutoMapIO::getImage(hf::Group group, string name)
{
    PlutoMapImage t;

    if (!group.exist(name))
    {
        return t;
    }

    hsize_t width;
    hsize_t height;
    hsize_t pixel_size;
    char interlace[20];
    hssize_t npals;

    H5IMget_image_info(group.getId(), name.c_str(), &width, &height, &pixel_size, interlace, &npals);

    auto bufSize = width * height * pixel_size;
    unsigned char buf[bufSize];
    H5IMread_image(group.getId(), name.c_str(), buf);

    t.name = name;
    t.width = width;
    t.height = height;
    t.channels = pixel_size;
    t.data = buf;

    return t;
}

// hsize_t zero[] = { 0 };
// hsize_t unlimited[] = { H5S_UNLIMITED };
// auto vertices_ds = H5Screate_simple(1, zero, unlimited);
// auto hid = H5Dcreate2(
//     geometry_group.getId(),
//     "vertices",
//     hf::AtomicType<PlutoMapVector>().getId(),
//     vertices_ds,
//     H5P_DEFAULT,
//     H5P_DEFAULT,
//     H5P_DEFAULT
// );

// geometry_group.createDataSet<PlutoMapVector>("vertices", hf::DataSpace(1));
// geometry_group.createDataSet<PlutoMapFace>("faces", hf::DataSpace(0));

inline hf::DataSet PlutoMapIO::addVertexNormals(vector<float>& normals)
{
    // TODO make more versatile to add and/or overwrite normals in file
    auto dataSet = m_attributesGroup.createDataSet<float>("normals", hf::DataSpace::From(normals));
    dataSet.write(normals);

    return dataSet;
}

inline hf::DataSet PlutoMapIO::addVertexColors(vector<uint8_t>& colors)
{
    auto dataSet = m_attributesGroup.createDataSet<uint8_t>("rgb_colors", hf::DataSpace::From(colors));
    dataSet.write(colors);

    return dataSet;
}

inline void PlutoMapIO::addTexture(int index, uint32_t width, uint32_t height, uint8_t *data)
{
    if (!m_texturesGroup.exist("images"))
    {
        m_texturesGroup.createGroup("images");
    }

    auto imagesGroup = m_texturesGroup.getGroup("images");
    const string& name = std::to_string(index);

    if (imagesGroup.exist(name))
    {
        return;
    }

    addImage(imagesGroup, name, width, height, data);
}

inline void PlutoMapIO::addMaterials(vector<PlutoMapMaterial>& materials, vector<uint32_t>& matFaceIndices)
{
    m_texturesGroup
        .createDataSet<PlutoMapMaterial>("materials", hf::DataSpace::From(materials))
        .write(materials);

    m_texturesGroup
        .createDataSet<uint32_t>("mat_face_indices", hf::DataSpace::From(matFaceIndices))
        .write(matFaceIndices);
}

inline void PlutoMapIO::addVertexTextureCoords(vector<float>& coords)
{
    m_texturesGroup
        .createDataSet<float>("coords", hf::DataSpace::From(coords))
        .write(coords);
}

inline void PlutoMapIO::addLabel(string groupName, string labelName, vector<uint32_t>& faceIds)
{
    if (!m_labelsGroup.exist(groupName))
    {
        m_labelsGroup.createGroup(groupName);
    }

    m_labelsGroup.getGroup(groupName)
        .createDataSet<uint32_t>(labelName, hf::DataSpace::From(faceIds))
        .write(faceIds);
}

inline void PlutoMapIO::addImage(hf::Group group, string name, const uint32_t width, const uint32_t height,
                                 const uint8_t *pixelBuffer)
{
    H5IMmake_image_24bit(group.getId(), name.c_str(), width, height, "INTERLACE_PIXEL", pixelBuffer);
}

} // namespace lvr2
