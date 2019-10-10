/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /**
 * @file PlutoMapIO.hpp
 */


#ifndef __LVR2_PLUTOMAPIO_HPP_
#define __LVR2_PLUTOMAPIO_HPP_


#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>

#include <H5Tpublic.h>
#include <highfive/H5File.hpp>

#include "lvr2/geometry/BaseVector.hpp"

namespace hf = HighFive;

using std::string;
using std::vector;
using std::unordered_map;

namespace lvr2
{

using Vec = BaseVector<float>;

/**
 * Helper struct to save textures / images to the map.
 */
struct PlutoMapImage {
    string name;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    vector<uint8_t> data;
};

/**
 * Helper struct for saving material data to the map.
 *
 * This struct is defined as an HDF compound data type.
 */
struct PlutoMapMaterial {
    int32_t textureIndex;
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

/**
 * This class if responsible for the Pluto map format. It tries to abstract most if not all calls to the
 * underlying HDF5 API and the HighFive wrapper. Furthermore it ensures the defined map format is always
 * in place and not tinkered with.
 *
 * NOTE: the map file is held open for the whole live time of this object. Thus it is possible that some data is
 * only written to disc if the destructor is called or the program has ended. Also make sure the map file is not opened
 * in any other way. (i.e. with the HDF5 Viewer). This will always lead to errors trying to access the file.
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
     * @brief Returns an map which keys are representing the features point in space and the values
     * are an vector of floats representing the keypoints.
     */
    unordered_map<Vec, vector<float>> getFeatures();

    /**
     * @brief Returns materials as PlutoMapMaterial
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
     * @brief Returns the roughness as float vector.
     */
    vector<float> getRoughness();

    /**
     * @brief Returns the height difference as float vector.
     */
    vector<float> getHeightDifference();

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
     * @brief Adds the keypoints with their corresponding positions to the attributes_group. The position
     * is saved to the entry via an attribute called 'vector'.
     */
    template<typename BaseVecT>
    void addTextureKeypointsMap(unordered_map<BaseVecT, std::vector<float>>& keypoints_map);

    /**
     * @brief Adds the roughness to the attributes group.
     */
    void addRoughness(vector<float>& roughness);

    /**
     * @brief Adds the height difference to the attributes group.
     */
    void addHeightDifference(vector<float>& diff);

    /**
     * @brief Adds an image with given data set name to the given group
     */
    void addImage(hf::Group group,
                  string name,
                  const uint32_t width,
                  const uint32_t height,
                  const uint8_t* pixelBuffer
    );

    /**
     * Removes all labels from the file.
     * <br>
     * Be careful, this does not clear up the space of the labels. Use the cli tool 'h5repack' manually to clear up
     * all wasted space if this method was used multiple times.
     *
     * @return true if removing all labels successfully.
     */
    bool removeAllLabels();

    /**
     * @brief Flushes the file. All opened buffers are saved to disc.
     */
    void flush();

private:
    hf::File m_file;

    // group names

    static constexpr const char* GEOMETRY_GROUP = "/geometry";
    static constexpr const char* ATTRIBUTES_GROUP = "/attributes";
    static constexpr const char* CLUSTERSETS_GROUP = "/clustersets";
    static constexpr const char* TEXTURES_GROUP = "/textures";
    static constexpr const char* LABELS_GROUP = "/labels";

    // main groups for reference
    hf::Group m_geometryGroup;
    hf::Group m_attributesGroup;
    hf::Group m_clusterSetsGroup;
    hf::Group m_texturesGroup;
    hf::Group m_labelsGroup;
};
} // namespace lvr2


namespace HighFive {

/**
 * Define the PlutoMapMaterial as an HDF5 compound data type.
 */
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
