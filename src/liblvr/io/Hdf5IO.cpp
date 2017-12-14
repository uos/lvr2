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


#include <lvr/io/Hdf5IO.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr2/io/PlutoMapIO.hpp>

#include <iomanip>

namespace lvr
{

void Hdf5IO::save(string filename)
{
    using namespace std;

    if (!m_model)
    {
        cerr << timestamp << "No data to save." << endl;
        return;
    }

    if (m_model->m_pointCloud || !m_model->m_mesh)
    {
        cerr << timestamp << "Only supporting saving mesh right now." << endl;
        return;
    }

    size_t numVertices = 0;
    size_t numFaceIds = 0;
    size_t numNormals = 0;
    size_t numColors = 0;
    size_t numTextures = 0;
    size_t numMaterials = 0;
    size_t numCoords = 0;
    size_t numMatFaceIndices = 0;
    floatArr vertices;
    uintArr faceIndices;
    floatArr normals;
    ucharArr colors;
    textureArr textures;
    materialArr materials;
    floatArr coords;
    uintArr matFaceIndices;

    vertices = m_model->m_mesh->getVertexArray(numVertices);
    faceIndices = m_model->m_mesh->getFaceArray(numFaceIds);
    normals = m_model->m_mesh->getVertexNormalArray(numNormals);
    colors = m_model->m_mesh->getVertexColorArray(numColors);
    textures = m_model->m_mesh->getTextureArray(numTextures);
    materials = m_model->m_mesh->getMaterialArray(numMaterials);
    coords = m_model->m_mesh->getVertexTextureCoordinateArray(numCoords);
    matFaceIndices = m_model->m_mesh->getFaceMaterialIndexArray(numMatFaceIndices);

    auto verts = std::vector<float>(vertices.get(), vertices.get() + numVertices * 3);
    auto indis = std::vector<uint32_t>(faceIndices.get(), faceIndices.get() + numFaceIds * 3);
    auto normalsVector = std::vector<float>(normals.get(), normals.get() + numNormals * 3);
    auto colorsVector = std::vector<uint8_t>(colors.get(), colors.get() + numColors * 3);
    auto coordsVector = std::vector<float>(coords.get(), coords.get() + numCoords);
    auto matFaceIndicesVector = std::vector<uint32_t>(matFaceIndices.get(), matFaceIndices.get() + numMatFaceIndices * 3);

    // Save old error handler
    H5E_auto2_t  oldfunc;
    void *old_client_data;
    H5Eget_auto(NULL, &oldfunc, &old_client_data);

    // Turn off error handling
    H5Eset_auto(NULL, NULL, NULL);

    auto pm = unique_ptr<lvr2::PlutoMapIO>{};
    try {
        pm = make_unique<lvr2::PlutoMapIO>(filename, verts, indis);
    } catch(exception& e) {
        // assume HDF5 could not open file and throws an exception
        cerr << timestamp << "File writing error: " << e.what() << endl;

        // prefix filename with current time (YearMonthDateHourMinuteSecond_) , which should be unique enough to save
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        ostringstream newFilename;
        newFilename << std::put_time(&tm, "%Y%m%d%H%M%S") << "_" << filename;

        cout << timestamp << "Saving now to '" << newFilename.str() << "'" << endl;

        pm = make_unique<lvr2::PlutoMapIO>(newFilename.str(), verts, indis);
    }

    // Restore previous error handler
    H5Eset_auto(NULL, oldfunc, old_client_data);

    pm->addVertexNormals(normalsVector);
    pm->addVertexColors(colorsVector);
    pm->addVertexTextureCoords(coordsVector);

    // add texture images
    for (size_t i = 0; i < numTextures; i++)
    {
        auto t = textures[i];

        pm->addTexture(t->m_texIndex, t->m_width, t->m_height, t->m_pixels);
    }

    // add materials
    std::vector<lvr2::PlutoMapMaterial> matVector;
    for (size_t i = 0; i < numMaterials; i++)
    {
        auto m = materials[i];
        lvr2::PlutoMapMaterial material{};

        material.textureIndex = m->texture_index;
        material.r = m->r;
        material.g = m->g;
        material.b = m->b;

        matVector.push_back(material);
    }

    pm->addMaterials(matVector, matFaceIndicesVector);
}


ModelPtr Hdf5IO::read(string filename)
{
    throw "Reading not yet implemented.";
}



} // namespace lvr
