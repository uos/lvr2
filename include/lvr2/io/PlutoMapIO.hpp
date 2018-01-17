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
#include <vector>
#include <iostream>

#include <H5Tpublic.h>
#include <highfive/H5File.hpp>

namespace hf = HighFive;

using std::string;
using std::vector;

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
     * @brief Opens a Pluto map file for reading and writing.
     */
    PlutoMapIO(string filename);

    /**
     * @brief Creates a Pluto map file (or truncates if the file already exists).
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
     * @brief Returns all available label groups
     */
    vector<string> getLabelGroups();

    /**
     * @brief  Returns all labels inside the given group
     */
    vector<string> getAllLabelsOfGroup(string groupName);

    /**
     * @brief Returns face ids for the given label inside the group.
     * E.g: label=tree_1 -> groupName=tree; labelName=1
     */
    vector<uint32_t> getFaceIdsOfLabel(string groupName, string labelName);

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

    /**
     * Removes all labels from the file.
     * <br>
     * Be careful, this does not clear up the space of the labels. Use the cli tool h5repack manually to clear up
     * all wasted space if this method was used multiple times.
     *
     * @return true if removing all labels successfully.
     */
    bool removeAllLabels();

private:
    hf::File m_file;

    // main groups for reference
    hf::Group m_geometryGroup;
    hf::Group m_attributesGroup;
    hf::Group m_clusterSetsGroup;
    hf::Group m_texturesGroup;
    hf::Group m_labelsGroup;
};
} // namespace lvr2




namespace HighFive {

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
