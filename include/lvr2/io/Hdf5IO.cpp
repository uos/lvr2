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

#include <lvr2/io/Hdf5IO.hpp>

namespace lvr2
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

    size_t numVertices = m_model->m_mesh->numVertices();
    size_t numFaceIds = m_model->m_mesh->numFaces();
    
    // Is this assumption ok?
    size_t numNormals = numVertices;
    size_t numColors = numVertices;
    size_t numCoords = numVertices;
    size_t numMatFaceIndices = numFaceIds;

    unsigned widthColor = 0;
    unsigned dummy;

    floatArr vertices = m_model->m_mesh->getVertices();
    indexArray faceIndices = m_model->m_mesh->getFaceIndices();
    floatArr normals = m_model->m_mesh->getVertexNormals();
    ucharArr colors = m_model->m_mesh->getVertexColors(widthColor);
    vector<Texture>& textures = m_model->m_mesh->getTextures();
    vector<Material>& materials = m_model->m_mesh->getMaterials();
    floatArr coords = m_model->m_mesh->getTextureCoordinates();
    indexArray matFaceIndices = m_model->m_mesh->getFaceMaterialIndices();


    auto verts = std::vector<float>(vertices.get(), vertices.get() + numVertices * 3);
    auto indis = std::vector<uint32_t>(faceIndices.get(), faceIndices.get() + numFaceIds * 3);
    auto normalsVector = std::vector<float>(normals.get(), normals.get() + numNormals * 3);
    auto colorsVector = std::vector<uint8_t>(colors.get(), colors.get() + numColors * widthColor);
    auto coordsVector = std::vector<float>(coords.get(), coords.get() + numCoords * 3);
    auto matFaceIndicesVector = std::vector<uint32_t>(matFaceIndices.get(), matFaceIndices.get() + numMatFaceIndices);

    // Save old error handler
    H5E_auto2_t  oldfunc;
    void *old_client_data;
    H5Eget_auto(0, &oldfunc, &old_client_data);

    // Turn off error handling
    // We turn it of here, to avoid an error message from HDF5 because of already opened file.
    // The first error message can be safely ignored since we only try to save to the default filename
    // and change it if it fails.
    // The error handler is turned on again later to make sure we have an proper error handler and message again
    // while using / opening an new file.
    H5Eset_auto(0, NULL, NULL);

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
    H5Eset_auto(0, oldfunc, old_client_data);

    pm->addVertexNormals(normalsVector);
    pm->addVertexColors(colorsVector);
    pm->addVertexTextureCoords(coordsVector);

    // add texture images

    for (const Texture& t : textures)
    {
        pm->addTexture(t.m_index, t.m_width, t.m_height, t.m_data);
    }

    // add materials
    std::vector<lvr2::PlutoMapMaterial> matVector;
    for (const Material& m : materials)
    {
        lvr2::PlutoMapMaterial material{};

        material.textureIndex = m.m_texture->idx();
        material.r = m.m_color->at(0);
        material.g = m.m_color->at(1);
        material.b = m.m_color->at(2);
    }

    pm->addMaterials(matVector, matFaceIndicesVector);
}

using HighFive::File;
using HighFive::Group;
using HighFive::Exception;

ModelPtr Hdf5IO::read(string filename)
{
    PointBuffer2Ptr pc;
    MeshBuffer2Ptr mesh;

    int numPoints = 0;
    int numNormals = 0;
    int numConfidences = 0;
    int numIntensities = 0;
    int numColors = 0;
    int numSpectralChannels = 0;
    int numChannels = 0;
    int minSpectral = 0;
    int maxSpectral = 0;

    floatArr points;
    floatArr normals;
    floatArr confidences;
    floatArr intensities;
    ucharArr colors;
    floatArr spectralChannels;

    try
    {
        File file(filename);

        Group clouds = file.getGroup("/pointclouds");
        std::vector<std::string> cloudNames = clouds.listObjectNames();

        if (cloudNames.size() == 0)
        {
            throw Exception("pointclouds Group does not contain clouds");
        }

        Group cloud = clouds.getGroup(cloudNames[0]);

        if (!cloud.exist("points"))
        {
            throw Exception("pointcloud does not contain points");
        }
        Group pointGroup = cloud.getGroup("points");

        pointGroup.getDataSet("numPoints").read(numPoints);

        if (!numPoints)
        {
            throw Exception("pointcloud does not contain points");
        }

        points = floatArr(new float[numPoints * 3]);
        pointGroup.getDataSet("points").read(points.get());

        if (cloud.exist("colors"))
        {
            Group colorGroup = cloud.getGroup("colors");

            colorGroup.getDataSet("numPoints").read(numColors);

            if (numColors)
            {
                colors = ucharArr(new unsigned char[numColors * 3]);
                colorGroup.getDataSet("colors").read(colors.get());
            }
        }

        if (cloud.exist("confidences"))
        {
            Group confidenceGroup = cloud.getGroup("confidences");

            confidenceGroup.getDataSet("numPoints").read(numConfidences);

            if (numConfidences)
            {
                confidences = floatArr(new float[numConfidences]);
                confidenceGroup.getDataSet("confidences").read(confidences.get());
            }
        }

        if (cloud.exist("intensities"))
        {
            Group intensityGroup = cloud.getGroup("intensities");

            intensityGroup.getDataSet("numPoints").read(numIntensities);

            if (numIntensities)
            {
                intensities = floatArr(new float[numIntensities]);
                intensityGroup.getDataSet("intensities").read(intensities.get());
            }
        }

        if (cloud.exist("spectralChannels"))
        {
            Group spectralChannelGroup = cloud.getGroup("spectralChannels");

            spectralChannelGroup.getDataSet("numPoints").read(numSpectralChannels);
            spectralChannelGroup.getDataSet("numChannels").read(numChannels);
            spectralChannelGroup.getDataSet("minSpectral").read(minSpectral);
            spectralChannelGroup.getDataSet("maxSpectral").read(maxSpectral);

            if (numSpectralChannels)
            {
                spectralChannels = floatArr(new float[numSpectralChannels * numChannels]);
                spectralChannelGroup.getDataSet("spectralChannels").read(spectralChannels.get());
            }
        }

        if(numPoints)
        {
            pc = PointBuffer2Ptr(new PointBuffer2);
            pc->setPointArray(points, numPoints);
            pc->setColorArray(colors, numColors);
            pc->addFloatChannel(intensities, "intensities", numIntensities, 1);
            pc->addFloatChannel(confidences, "confidences", numConfidences, 1);
            pc->setNormalArray(normals, numNormals);

            // add spectral channels to Pointbuffer 
            pc->addIntAttribute(minSpectral, "spectral_wavelength_min");
            pc->addIntAttribute(maxSpectral, "spectral_wavelength_max");
            pc->addFloatChannel(spectralChannels, "spectral_channels", numSpectralChannels, numChannels);
        }
    }
    catch(HighFive::Exception& err)
    {
        std::cerr << "Unable to read File: " << err.what() << std::endl;
    }

    ModelPtr m(new Model(mesh, pc));
    m_model = m;
    return m;
}



} // namespace lvr2
