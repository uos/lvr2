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

}

#include "PlutoMapIO.tcc"

#endif // __LVR2_PLUTOMAPIO_HPP_
