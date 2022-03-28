/* Copyright (C) 2018 Uni Osnabrück
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
 *
 * @file      DracoDecoder.cpp
 * @brief     Decodes a draco comptressed file into a lvr model
 * @details   Supports geometrys compressed using draco https://github.com/google/draco
 *
 * @author    Steffen Schupp (sschupp), sschupp@uos.de
 * @author	  Malte kl. Piening (mklpiening), mklpiening@uos.de
 *
 **/

#include "lvr2/io/DracoDecoder.hpp"
#include <draco/metadata/geometry_metadata.h>

namespace lvr2
{

/**
 * @brief loads any GeometryAttribute from draco pointcloud and returns its values
 *
 * @tparam LvrArrType Array type in Lvr (something like floatArr, or uintArr)
 * @tparam DataType Type of data that gets stored in the array that gets returned
 * @tparam numComponents num of the components of the dtaco attribute that gets loaded
 *
 * @param attribute Pointer to PointAttribute which is used to load data into lvr structure
 * @return LvrType data from Attribute in lvr structure
 *
 * @author Malte kl. Piening
 **/
template <typename LvrArrType, typename DataType, int numComponents>
LvrArrType loadAttributeFromDraco(const draco::PointAttribute* attribute)
{
    if (!attribute)
    {
        throw std::invalid_argument("attribute is not existing");
    }

    LvrArrType data(new DataType[attribute->size() * numComponents]);

    std::array<DataType, numComponents> tmp;
    for (draco::AttributeValueIndex i(0); i < attribute->size(); ++i)
    {
        if (!attribute->ConvertValue<DataType, numComponents>(i, &tmp[0]))
        {
            throw std::invalid_argument("attribute seems to have an incorrect structur");
        }

        for (int j = 0; j < numComponents; j++)
        {
            data[i.value() * numComponents + j] = tmp[j];
        }
    }

    return data;
}

/**
 *
 * @brief transforms the int32_t vector with texture data into actual GlTextures
 *
 * @param drcTextures the int32_t vector holding the image data
 * @param lvrTextures a GlTexture vector with the created textures
 */
void createTextures(std::vector<int32_t>& drcTextures, std::vector<GlTexture*>& lvrTextures)
{
    unsigned long size  = drcTextures.size();
    unsigned long index = 0;

    while (index < size)
    {
        // set texture attributes
        int id     = drcTextures[index++];
        int height = drcTextures[index++];
        int width  = drcTextures[index++];

        int            pixelDat = height * width * 3;
        unsigned char* pixel    = new unsigned char[pixelDat];

        // create pixel array
        for (int i = 0; i < pixelDat; ++i)
        {
            pixel[i] = (unsigned char)drcTextures[index++];
        }

        GlTexture* glTexture  = new GlTexture(pixel, width, height);
        glTexture->m_texIndex = (GLuint)id;
        lvrTextures.push_back(glTexture);
    }
}

/**
 *
 * @brief transforms the int32_t vector with texture data into actual GlTextures
 *
 * @param drcTextures the int32_t vector holding the image data
 * @param lvrTextures a GlTexture vector with the created textures
 */
void createTextures(std::vector<int32_t>& drcTextures, std::vector<Texture>& lvrTextures)
{
    unsigned long size  = drcTextures.size();
    unsigned long index = 0;

    while (index < size)
    {
        // set texture attributes
        int id     = drcTextures[index++];
        int height = drcTextures[index++];
        int width  = drcTextures[index++];

        int            pixelDat = height * width * 3;
        unsigned char* pixel    = new unsigned char[pixelDat];

        // create pixel array
        for (int i = 0; i < pixelDat; ++i)
        {
            pixel[i] = (unsigned char)drcTextures[index++];
        }

        Texture texture(id, width, height, 3, 1, 1, pixel);
        lvrTextures.push_back(texture);
    }
}

/**
 * @brief delivers PointAttribute by searching for given Attribute Metadata Entries
 *
 * @param geometry pointcloud that contains the attribute and its metadata
 * @param key String key of key value pair to be searched for
 * @param value String value of key value pair to be searched for
 * @return Pointer to PointAttribute that has the given key value pair in its AttributeMetadata;
 *nullptr if the Attribute Metadata could not be found
 *
 * @author Malte kl. Piening
 **/
const draco::PointAttribute* getDracoAttributeByAttributeMetadata(draco::PointCloud* geometry,
                                                                  std::string        key,
                                                                  std::string        value)
{
    if (!geometry->GetMetadata())
    {
        return nullptr;
    }

    const draco::AttributeMetadata* attributeMeta =
        geometry->GetMetadata()->GetAttributeMetadataByStringEntry(key, value);

    if (!attributeMeta)
    {
        return nullptr;
    }

    return geometry->GetAttributeByUniqueId(attributeMeta->att_unique_id());
}

/**
 * @brief loads a pointcloud from the decoderBuffer and converts it into the lvr structure
 *
 * @param buffer DecoderBuffer that contains the encoded data from the draco file
 * @param decoder Decoder that is used to decode the buffer
 * @return ModelPtr to the lvr model that got loaded from the draco file
 **/
ModelPtr readPointCloud(draco::DecoderBuffer& buffer, draco::Decoder& decoder)
{
    auto status = decoder.DecodePointCloudFromBuffer(&buffer);

    if (!status.ok())
    {
        std::cerr << "An error occurred while decoding file:"
                  << " " << status.status() << std::endl;
        return ModelPtr(new Model());
    }

    ModelPtr modelPtr(new Model(PointBufferPtr(new PointBuffer)));

    std::unique_ptr<draco::PointCloud> dracoPointCloud = std::move(status).value();

    // get coordinates
    try
    {
        const draco::PointAttribute* attribute =
            dracoPointCloud->GetNamedAttribute(draco::GeometryAttribute::POSITION);
        floatArr data = loadAttributeFromDraco<floatArr, float, 3>(attribute);
        modelPtr->m_pointCloud->setPointArray(data, attribute->size());
    }
    catch (const std::invalid_argument& ia)
    {
        // no points, no pointcloud
        std::cerr << "Error loading positions: " << ia.what() << std::endl;
        return ModelPtr(new Model());
    }

    // get normals
    try
    {
        const draco::PointAttribute* attribute =
            dracoPointCloud->GetNamedAttribute(draco::GeometryAttribute::NORMAL);
        floatArr data = loadAttributeFromDraco<floatArr, float, 3>(attribute);
        modelPtr->m_pointCloud->setNormalArray(data, attribute->size());
    }
    catch (const std::invalid_argument& ia)
    {
    }

    // get colors
    try
    {
        const draco::PointAttribute* attribute =
            dracoPointCloud->GetNamedAttribute(draco::GeometryAttribute::COLOR);
        ucharArr data = loadAttributeFromDraco<ucharArr, unsigned char, 3>(attribute);
        modelPtr->m_pointCloud->setColorArray(data, attribute->size());
    }
    catch (const std::invalid_argument& ia)
    {
    }

    // // get confidences
    // try
    // {
    //     const draco::PointAttribute* attribute =
    //         getDracoAttributeByAttributeMetadata(dracoPointCloud.get(), "name", "confidence");
    //     floatArr data = loadAttributeFromDraco<floatArr, float, 1>(attribute);
    //     modelPtr->m_pointCloud->setPointConfidenceArray(data, attribute->size());
    // }
    // catch (const std::invalid_argument& ia)
    // {
    // }

    // // get intensities
    // try
    // {
    //     const draco::PointAttribute* attribute =
    //         getDracoAttributeByAttributeMetadata(dracoPointCloud.get(), "name", "intensity");
    //     floatArr data = loadAttributeFromDraco<floatArr, float, 1>(attribute);
    //     modelPtr->m_pointCloud->setPointIntensityArray(data, attribute->size());
    // }
    // catch (const std::invalid_argument& ia)
    // {
    // }

    return modelPtr;
}

/**
 * @brief loads a mesh from the decoderBuffer and converts it into the lvr structure
 *
 * @param buffer DecoderBuffer that contains the encoded data from the draco file
 * @param decoder Decoder that is used to decode the buffer
 * @return ModelPtr to the lvr model that got loaded from the draco file
 **/
ModelPtr readMesh(draco::DecoderBuffer& buffer, draco::Decoder& decoder)
{
    auto status = decoder.DecodeMeshFromBuffer(&buffer);

    if (!status.ok())
    {
        std::cerr << "An error occurred while decoding file:"
                  << " " << status.status() << std::endl;
        return ModelPtr(new Model());
    }

    ModelPtr modelPtr(new Model(MeshBufferPtr(new MeshBuffer)));

    std::unique_ptr<draco::Mesh> dracoMesh = std::move(status).value();

    auto metadata = dracoMesh->GetMetadata();

    // get textures
    if (metadata)
    {
        std::vector<int32_t> texture;

        if (metadata->GetEntryIntArray("texture", &texture))
        {
            std::vector<Texture> textures;

            createTextures(texture, textures);

            modelPtr->m_mesh->setTextures(textures);
        }
    }

    // get materials
    if (metadata)
    {
        std::vector<int32_t> dat;

        if (metadata->GetEntryIntArray("material", &dat))
        {
            vector<Material> materials;
            for (int i = 0; i < dat.size() / 4; i++)
            {
                Material m;
                Rgb8Color rgb;
                rgb[0] = static_cast<unsigned char>(dat[4 * i + 0]);
                rgb[0] = static_cast<unsigned char>(dat[4 * i + 1]);
                rgb[0] = static_cast<unsigned char>(dat[4 * i + 2]);

                TextureHandle h(dat[4 * i + 3]);

                m.m_texture = h;
                m.m_color = rgb;

                materials.push_back(m);
            }

            modelPtr->m_mesh->setMaterials(materials);
        }
    }

    // get coordinates
    try
    {
        const draco::PointAttribute* attribute =
            dracoMesh->GetNamedAttribute(draco::GeometryAttribute::POSITION);
        floatArr data = loadAttributeFromDraco<floatArr, float, 3>(attribute);
        modelPtr->m_mesh->setVertices(data, attribute->size());
    }
    catch (const std::invalid_argument& ia)
    {
        // no vertices, no mesh
        std::cerr << "Error loading positions: " << ia.what() << std::endl;
        return ModelPtr(new Model());
    }

    // get normals
    try
    {
        const draco::PointAttribute* attribute =
            dracoMesh->GetNamedAttribute(draco::GeometryAttribute::NORMAL);
        floatArr data = loadAttributeFromDraco<floatArr, float, 3>(attribute);
        modelPtr->m_mesh->setVertexNormals(data);
    }
    catch (const std::invalid_argument& ia)
    {
    }

    // get colors
    try
    {
        const draco::PointAttribute* attribute =
            dracoMesh->GetNamedAttribute(draco::GeometryAttribute::COLOR);
        ucharArr data = loadAttributeFromDraco<ucharArr, unsigned char, 3>(attribute);
        modelPtr->m_mesh->setVertexColors(data);
    }
    catch (const std::invalid_argument& ia)
    {
    }

    // // get confidences
    // try
    // {
    //     const draco::PointAttribute* attribute =
    //         getDracoAttributeByAttributeMetadata(dracoMesh.get(), "name", "confidence");
    //     floatArr data = loadAttributeFromDraco<floatArr, float, 1>(attribute);
    //     modelPtr->m_mesh->setVertexConfidenceArray(data, attribute->size());
    // }
    // catch (const std::invalid_argument& ia)
    // {
    // }

    // // get intensities
    // try
    // {
    //     const draco::PointAttribute* attribute =
    //         getDracoAttributeByAttributeMetadata(dracoMesh.get(), "name", "intensity");
    //     floatArr data = loadAttributeFromDraco<floatArr, float, 1>(attribute);
    //     modelPtr->m_mesh->setVertexIntensityArray(data, attribute->size());
    // }
    // catch (const std::invalid_argument& ia)
    // {
    // }

    // get texture coordinates
    try
    {
        const draco::PointAttribute* attribute =
            dracoMesh->GetNamedAttribute(draco::GeometryAttribute::TEX_COORD);
        floatArr data = loadAttributeFromDraco<floatArr, float, 3>(attribute);
        modelPtr->m_mesh->setTextureCoordinates(data);
    }
    catch (const std::invalid_argument& ia)
    {
    }

    // get materialindices
    const draco::PointAttribute* matIndexAttribute =
        getDracoAttributeByAttributeMetadata(dracoMesh.get(), "name", "materialindex");
    if (matIndexAttribute)
    {
        indexArray data(new uint32_t[dracoMesh->num_faces()]);

        uint32_t tmp;
        for (draco::FaceIndex i(0); i < dracoMesh->num_faces(); i++)
        {
            if (!matIndexAttribute->ConvertValue<uint32_t, 1>(
                    matIndexAttribute->mapped_index(dracoMesh->face(i)[0]), &tmp))
            {
                data = uintArr(new uint32_t[dracoMesh->num_faces()]);
                break;
            }

            data[i.value()] = tmp;
        }

        modelPtr->m_mesh->setFaceMaterialIndices(data);
    }

    // get faces
    const draco::PointAttribute* faceAttribute =
        dracoMesh->GetNamedAttribute(draco::GeometryAttribute::Type::POSITION);
    uintArr faceArr(new unsigned int[dracoMesh->num_faces() * 3]);
    for (draco::FaceIndex i(0); i < dracoMesh->num_faces(); ++i)
    {
        faceArr[i.value() * 3 + 0] = faceAttribute->mapped_index(dracoMesh->face(i)[0]).value();
        faceArr[i.value() * 3 + 1] = faceAttribute->mapped_index(dracoMesh->face(i)[1]).value();
        faceArr[i.value() * 3 + 2] = faceAttribute->mapped_index(dracoMesh->face(i)[2]).value();
    }
    modelPtr->m_mesh->setFaceIndices(faceArr, dracoMesh->num_faces());

    return modelPtr;
}

ModelPtr decodeDraco(draco::DecoderBuffer& buffer, draco::EncodedGeometryType type)
{
    draco::Decoder decoder;

    if (type == draco::TRIANGULAR_MESH)
    {
        return readMesh(buffer, decoder);
    }
    else if (type == draco::POINT_CLOUD)
    {
        return readPointCloud(buffer, decoder);
    }

    return ModelPtr(new Model());
}

} // namespace lvr