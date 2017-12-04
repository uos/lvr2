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
#include <lvr2/io/PlutoMapIO.hpp>

#include <cstdint>
#include <iostream>

namespace lvr
{

void Hdf5IO::save(string filename)
{
    using namespace std;

    size_t numVertices = 0;
    size_t numFaceIds = 0;
    size_t numNormals = 0;
    size_t numColors = 0;
    floatArr vertices;
    uintArr faceIndices;
    floatArr normals;
    ucharArr colors;

    // TODO: check whether or not we have a mesh...
    vertices = m_model->m_mesh->getVertexArray(numVertices);
    faceIndices = m_model->m_mesh->getFaceArray(numFaceIds);
    normals = m_model->m_mesh->getVertexNormalArray(numNormals);
    colors = m_model->m_mesh->getVertexColorArray(numColors);

    cout << "num vertices: " << numVertices << endl;

    auto verts = std::vector<float>(vertices.get(), vertices.get() + numVertices * 3);
    auto indis = std::vector<uint32_t>(faceIndices.get(), faceIndices.get() + numFaceIds * 3);
    auto normalsVector = std::vector<float>(normals.get(), normals.get() + numNormals * 3);
    auto colorsVector = std::vector<uint8_t>(colors.get(), colors.get() + numColors * 3);

    cout << "verts size: " << verts.size() << endl;

    lvr2::PlutoMapIO pm(filename, verts, indis);
    pm.addNormals(normalsVector);
    pm.addVertexColors(colorsVector);
}


ModelPtr Hdf5IO::read(string filename)
{
    throw "yolo";
}



} // namespace lvr
