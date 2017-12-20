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
 * @file PlutoMapIO.hpp
 */


#ifndef __LVR2_PLUTOMAPIO_HPP_
#define __LVR2_PLUTOMAPIO_HPP_


#include <string>
#include <highfive/H5File.hpp>

#include <H5Tpublic.h>

namespace hf = HighFive;

using std::string;

namespace lvr2
{

struct PlutoMapImage {
    string name;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t* data;
};

struct PlutoMapMaterial {
    int32_t textureIndex;
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

/**
 *
 */
class PlutoMapIO
{
public:
    /**
     * @brief Opens a Pluto map file for reading.
     */
    PlutoMapIO(string filename);

    /**
     * @brief Creates a Pluto map file (or truncates if the file alrady exists).
     */
    PlutoMapIO(
        string filename,
        const vector<float>& vertices,
        const vector<uint32_t>& face_ids
    );

    /**
     * @brief Closes main groups and makes sure all buffers are flushed to the file on disc.
     */
    ~PlutoMapIO();

    /**
     * @brief Returns vertices vector
     */
    vector<float> getVertices();

    /**
     * @brief Returns face ids vector
     */
    vector<uint32_t> getFaceIds();

    /**
     * @brief Returns vertex normals vector
     */
    vector<float> getVertexNormals();

    /**
     * @brief Returns vertex colors vector
     */
    vector<uint8_t> getVertexColors();

    /**
     * @brief Returns textures vector
     */
    // TODO: replace PlutoMapImage with lvr2::Texture?
    vector<PlutoMapImage> getTextures();

    /**
     * @breif Returns materials as PlutoMapMaterial
     */
    vector<PlutoMapMaterial> getMaterials();

    /**
     * @brief Returns material <-> face indices
     */
    vector<uint32_t> getMaterialFaceIndices();

    /**
     * @brief Returns vertex texture coordinates
     */
    vector<float> getVertexTextureCoords();

    /**
     * @brief Returns the image in the group, if it exists. If not an empty struct is returned
     */
    PlutoMapImage getImage(hf::Group group, string name);

    /**
     * @brief Add normals to the attributes group.
     */
    hf::DataSet addVertexNormals(vector<float>& normals);

    /**
     * @brief Add vertex colors to the attributes group.
     */
    hf::DataSet addVertexColors(vector<uint8_t>& colors);

    /**
     * Add texture img with given index to the textures group. Texture CAN NOT be overridden
     */
    void addTexture(int index, uint32_t width, uint32_t height, uint8_t* data);

    /**
     * @brief Add materials as PlutoMapMaterial and the corresponding material <-> face indices
     */
    void addMaterials(vector<PlutoMapMaterial>& materials, vector<uint32_t>& matFaceIndices);

    /**
     * @brief Add vertex texture coordinates to the textures group.
     */
    void addVertexTextureCoords(vector<float>& coords);

    /**
     * @brief Adds the label (labelName) to the label group with the given faces.
     * E.g.: tree_1 -> groupName=tree; labelName=1; separated by the '_'
     */
    void addLabel(string groupName, string labelName, vector<uint32_t>& faceIds);

    /**
     * @brief Adds an image with given data set name to the given group
     */
    void addImage(hf::Group group, string name, const uint32_t width, const uint32_t height, const uint8_t* pixelBuffer);

private:
    hf::File m_file;

    // main groups for reference
    hf::Group m_geometryGroup;
    hf::Group m_attributesGroup;
    hf::Group m_clusterSetsGroup;
    hf::Group m_texturesGroup;
    hf::Group m_labelsGroup;
};


struct PlutoMapVector {
    PlutoMapVector(float x, float y, float z) : x(x), y(y), z(z) {}

    float x;
    float y;
    float z;
};
struct PlutoMapFace {};

} // namespace lvr2




namespace HighFive {

template <>
inline AtomicType<lvr2::PlutoMapVector>::AtomicType()
{
    hid_t vector_hid = H5Tcreate(H5T_COMPOUND, sizeof(float) * 3);

    H5Tinsert(vector_hid, "x", sizeof(float) * 0 , H5T_NATIVE_FLOAT);
    H5Tinsert(vector_hid, "y", sizeof(float) * 1 , H5T_NATIVE_FLOAT);
    H5Tinsert(vector_hid, "z", sizeof(float) * 2 , H5T_NATIVE_FLOAT);

    _hid = H5Tcopy(vector_hid);
}

template <>
inline AtomicType<lvr2::PlutoMapFace>::AtomicType()
{
    hid_t face_hid = H5Tcreate(H5T_COMPOUND, sizeof(uint32_t) * 3);

    H5Tinsert(face_hid, "a", sizeof(uint32_t) * 0 , H5T_NATIVE_UINT);
    H5Tinsert(face_hid, "b", sizeof(uint32_t) * 1 , H5T_NATIVE_UINT);
    H5Tinsert(face_hid, "c", sizeof(uint32_t) * 2 , H5T_NATIVE_UINT);

    _hid = H5Tcopy(face_hid);
}
template <>
inline AtomicType<lvr2::PlutoMapMaterial>::AtomicType()
{
    hid_t materialHid = H5Tcreate(H5T_COMPOUND, sizeof(lvr2::PlutoMapMaterial));

    H5Tinsert(materialHid, "textureIndex", offsetof(lvr2::PlutoMapMaterial, textureIndex), H5T_NATIVE_INT);
    H5Tinsert(materialHid, "r", offsetof(lvr2::PlutoMapMaterial, r), H5T_NATIVE_UCHAR);
    H5Tinsert(materialHid, "g", offsetof(lvr2::PlutoMapMaterial, g), H5T_NATIVE_UCHAR);
    H5Tinsert(materialHid, "b", offsetof(lvr2::PlutoMapMaterial, b), H5T_NATIVE_UCHAR);

    _hid = H5Tcopy(materialHid);
}

}

#include "PlutoMapIO.tcc"

#endif // __LVR2_PLUTOMAPIO_HPP_
