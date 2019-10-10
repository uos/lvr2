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
 * @file      DracoEncoder.cpp
 * @brief     Encodes a lvr model into a draco compressed file
 * @details   Supports geometrys compressed using draco https://github.com/google/draco
 *
 * @author    Steffen Schupp (sschupp), sschupp@uos.de
 * @author	  Malte kl. Piening (mklpiening), mklpiening@uos.de
 *
 **/

#include "lvr2/io/DracoEncoder.hpp"
#include <draco/metadata/geometry_metadata.h>

namespace lvr2
{

/**
 *
 * @brief adds the given data to a new attribute and attaches the attribute to the pointcloud
 *
 * @tparam ArrayType LVR intern type for storing pointcloud information
 * @tparam DataType data type for the data that is to be stored in the attribute
 * @tparam size is the number of values linked to a single point in the point cloud
 *
 * @param array contains the data that is attempted to be stored in the attribute
 * @param drcPointcloud the draco intern pointcloud holding all the attributes
 * @param geometryType information about what is being stored in the attribute
 * @param dracoDataType data type that is used in draco
 * @param numPoints number of points in the array
 * @param normalized do the values have to be normalized?
 * @return id of the newly created attribute
 */
template <typename ArrayType, typename DataType, int size>
int saveAttributeToDraco(ArrayType array, draco::PointCloud* drcPointcloud,
                         draco::GeometryAttribute::Type geometryType, draco::DataType dracoDataType,
                         size_t numPoints, bool normalized)
{
    if (array == nullptr)
    {
        throw std::invalid_argument("attribute is not existing");
    }

    draco::PointAttribute attribute;
    attribute.Init(geometryType, nullptr, size, dracoDataType, normalized, sizeof(DataType) * size,
                   0);
    int attribute_id = drcPointcloud->AddAttribute(attribute, true, numPoints);

    std::array<DataType, size> tmp;
    for (int i = 0; i < numPoints; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            tmp[j] = array[i * size + j];
        }

        drcPointcloud->attribute(attribute_id)
            ->SetAttributeValue(draco::AttributeValueIndex(i), &tmp);
    }

    return attribute_id;
}

/**
 *
 * @brief pushes the given textures in a int32_t vector to store it in the draco structure
 *
 * @param textureValue the vector to be stored in draco
 * @param textures the exported textures from lvr
 * @param numTextures number of textures in the textureArr
 */
void saveTextures(std::vector<int32_t>* textureValue, textureArr textures, size_t numTextures)
{
    for (int i = 0; i < numTextures; ++i)
    {
        // get texture attributes
        GlTexture*     lvrTexture = textures[i];
        int            index      = lvrTexture->m_texIndex;
        int            height     = lvrTexture->m_height;
        int            width      = lvrTexture->m_width;
        unsigned char* pixel      = lvrTexture->m_pixels;

        // calculate number of values in the pixel-array (number of pixels * 3(rgb))
        unsigned long pixelDat = (unsigned long)height * width * 3;

        // store texture attributes
        textureValue->push_back(index);
        textureValue->push_back(height);
        textureValue->push_back(width);

        // store pixel data
        for (unsigned long j = 0; j < pixelDat; ++j)
        {
            textureValue->push_back((int32_t)pixel[j]);
        }
    }
}

/**
 *
 * @brief pushes the given textures in a int32_t vector to store it in the draco structure
 *
 * @param textureValue the vector to be stored in draco
 * @param textures the exported textures from lvr
 * @param numTextures number of textures in the textureArr
 */
void saveTextures(std::vector<int32_t>* textureValue, vector<Texture>& textures)
{
    for (int i = 0; i < textures.size(); ++i)
    {
        // get texture attributes
        Texture lvrTexture = textures[i];
        int            index      = lvrTexture.m_index;
        int            height     = lvrTexture.m_height;
        int            width      = lvrTexture.m_width;
        unsigned char* pixel      = lvrTexture.m_data;

        // calculate number of values in the pixel-array (number of pixels * 3(rgb))
        unsigned long pixelDat = (unsigned long)height * width * 3;

        // store texture attributes
        textureValue->push_back(index);
        textureValue->push_back(height);
        textureValue->push_back(width);

        // store pixel data
        for (unsigned long j = 0; j < pixelDat; ++j)
        {
            textureValue->push_back((int32_t)pixel[j]);
        }
    }
}

/**
 * @brief transfers a pointcloud of a modelPtr to a draco EncoderBuffer that can be written into a
 *file
 *
 * @param modelPtr pointer to model thats pointcloud shall be encoded
 * @param encoder is used to encode the modelptr to a encodeBuffer
 * @return unique pointer to a EncodeBuffer containing the draco encoded pointcloud that can be
 *written into a file or a nullptr in case of an error
 **/
std::unique_ptr<draco::EncoderBuffer> encodePointCloud(ModelPtr modelPtr, draco::Encoder& encoder)
{
    draco::PointCloud        pointCloud;
    draco::GeometryMetadata* metadata = new draco::GeometryMetadata();

    // assuming number of colors, normals, intensities and confidences are equal to number of points
    size_t numElem;
    size_t numPoints = modelPtr->m_pointCloud->numPoints();
    pointCloud.set_num_points(numPoints);

    // save point coordinates
    try
    {
        floatArr coordinates = modelPtr->m_pointCloud->getPointArray();
        saveAttributeToDraco<floatArr, float, 3>(coordinates, &pointCloud,
                                                 draco::GeometryAttribute::Type::POSITION,
                                                 draco::DT_FLOAT32, numPoints, false);
    }
    catch (const std::invalid_argument& ia)
    {
        std::cerr << "No point coordinates could be found in the pointcloud" << std::endl;
        return std::unique_ptr<draco::EncoderBuffer>(nullptr);
    }

    // save point normals
    try
    {
        floatArr normals = modelPtr->m_pointCloud->getNormalArray();
        if (normals)
        {
            saveAttributeToDraco<floatArr, float, 3>(normals, &pointCloud,
                                                 draco::GeometryAttribute::Type::NORMAL,
                                                 draco::DT_FLOAT32, numElem, true);
        }
    }
    catch (const std::invalid_argument& ia)
    {
    }

    // save point colors
    try
    {
        size_t w;
        ucharArr colors = modelPtr->m_pointCloud->getColorArray(w);
        if (colors && (w == 3))
        {

            saveAttributeToDraco<ucharArr, unsigned char, 3>(colors, &pointCloud,
                                                             draco::GeometryAttribute::Type::COLOR,
                                                             draco::DT_UINT8, numElem, false);
        }
    }
    catch (const std::invalid_argument& ia)
    {
    }
    /// TODO: Change to channel
    // save point confidences
    // try
    // {
    //     floatArr confidences = modelPtr->m_pointCloud->getPointIntensityArray(numElem);

    //     int attId = saveAttributeToDraco<floatArr, float, 1>(
    //         confidences, &pointCloud, draco::GeometryAttribute::Type::GENERIC, draco::DT_FLOAT32,
    //         numElem, false);

    //     draco::AttributeMetadata* attributeMeta = new draco::AttributeMetadata();
    //     attributeMeta->set_att_unique_id(attId);
    //     attributeMeta->AddEntryString("name", "confidence");

    //     metadata->AddAttributeMetadata(std::unique_ptr<draco::AttributeMetadata>(attributeMeta));
    // }
    // catch (const std::invalid_argument& ia)
    // {
    // }

    /// TODO: Change to channel
    // save point intensities
    // try
    // {
    //     floatArr intensities = modelPtr->m_pointCloud->getPointIntensityArray(numElem);

    //     int attId = saveAttributeToDraco<floatArr, float, 1>(
    //         intensities, &pointCloud, draco::GeometryAttribute::Type::GENERIC, draco::DT_FLOAT32,
    //         numElem, false);

    //     draco::AttributeMetadata* attributeMeta = new draco::AttributeMetadata();
    //     attributeMeta->set_att_unique_id(attId);
    //     attributeMeta->AddEntryString("name", "intensity");

    //     metadata->AddAttributeMetadata(std::unique_ptr<draco::AttributeMetadata>(attributeMeta));
    // }
    // catch (const std::invalid_argument& ia)
    // {
    // }

    pointCloud.AddMetadata(std::unique_ptr<draco::GeometryMetadata>(metadata));

    // Encode Data
    std::unique_ptr<draco::EncoderBuffer> buffer(new draco::EncoderBuffer());
    auto status = encoder.EncodePointCloudToBuffer(pointCloud, buffer.get());

    if (!status.ok())
    {
        std::cerr << "An error occurred:"
                  << " " << status.error_msg_string() << std::endl;
        return std::unique_ptr<draco::EncoderBuffer>(nullptr);
    }

    return buffer;
}

/**
 * @brief transfers a mesh of a modelPtr to a draco EncoderBuffer that can be written into a file
 *
 * @param modelPtr pointer to model thats mesh shall be encoded
 * @param encoder is used to encode the modelptr to a encodeBuffer
 * @return unique pointer to a EncodeBuffer containing the draco encoded mesh that can be written
 *into a file or a nullptr in case of an error
 **/
std::unique_ptr<draco::EncoderBuffer> encodeMesh(ModelPtr modelPtr, draco::Encoder& encoder)
{
    draco::Mesh              mesh;
    draco::GeometryMetadata* metadata = new draco::GeometryMetadata();

    // load sizes and arrays from modelPtr
    size_t w;
    size_t   numVertices = modelPtr->m_mesh->numVertices();
    floatArr vertices = modelPtr->m_mesh->getVertices();
    floatArr vertexNormals = modelPtr->m_mesh->getVertexNormals();    
    ucharArr vertexColors = modelPtr->m_mesh->getVertexColors(w);
    bool hasVertexColors = modelPtr->m_mesh->hasVertexColors();
    bool hasFaceColors = modelPtr->m_mesh->hasFaceColors();
    bool hasVertexNormals = modelPtr->m_mesh->hasVertexNormals();
    bool hasFaceNormals = modelPtr->m_mesh->hasFaceNormals();

    // size_t   numVertexConfidences;
    // floatArr vertexConfidences = modelPtr->m_mesh->getVertexConfidenceArray(numVertexConfidences);

    // size_t   numVertexIntensities;
    // floatArr vertexIntensities = modelPtr->m_mesh->getVertexIntensityArray(numVertexIntensities);

    size_t   numVertexTextureCoordinates = numVertices;
    floatArr vertexTextureCoordinates = modelPtr->m_mesh->getTextureCoordinates();

    size_t  numFaces = modelPtr->m_mesh->numFaces();
    uintArr faces = modelPtr->m_mesh->getFaceIndices();

    std::vector<Texture> textures = modelPtr->m_mesh->getTextures();
    std::vector<Material> materials = modelPtr->m_mesh->getMaterials();

    indexArray faceMaterialIndices = modelPtr->m_mesh->getFaceMaterialIndices();

    // put data into draco format
    mesh.set_num_points(numFaces * 3); // one point per corner
    mesh.SetNumFaces(numFaces);

    // init Attributes
    int verticesAttId                 = -1;
    int vertexNormalsAttId            = -1;
    int vertexColorsAttId             = -1;
    int vertexConfidencesAttId        = -1;
    int vertexIntensitiesAttId        = -1;
    int vertexTextureCoordinatesAttId = -1;
    int faceMaterialIndicesAttId      = -1;

    // textures and materials are stored as metadata

    // if there are material indices which exist per face, index draco mesh by face
    // if not, index draco mesh by vertex:
    bool faceIndexed = (faceMaterialIndices != 0);

    if (numVertices > 0)
    {
        draco::GeometryAttribute attribute;
        attribute.Init(draco::GeometryAttribute::POSITION, nullptr, 3, draco::DT_FLOAT32, false,
                       sizeof(float) * 3, 0);
        verticesAttId =
            mesh.AddAttribute(attribute, true, faceIndexed ? numFaces * 3 : numVertices);
        mesh.SetAttributeElementType(verticesAttId,
                                     draco::MeshAttributeElementType::MESH_VERTEX_ATTRIBUTE);
    }

    if (hasVertexNormals)
    {
        draco::GeometryAttribute attribute;
        attribute.Init(draco::GeometryAttribute::NORMAL, nullptr, 3, draco::DT_FLOAT32, true,
                       sizeof(float) * 3, 0);
        vertexNormalsAttId =
            mesh.AddAttribute(attribute, true, faceIndexed ? numFaces * 3 : numVertices);
        mesh.SetAttributeElementType(vertexNormalsAttId,
                                     draco::MeshAttributeElementType::MESH_VERTEX_ATTRIBUTE);
    }

    if (hasVertexColors)
    {
        draco::GeometryAttribute attribute;
        attribute.Init(draco::GeometryAttribute::COLOR, nullptr, 3, draco::DT_UINT8, false,
                       sizeof(uint8_t) * 3, 0);
        vertexColorsAttId =
            mesh.AddAttribute(attribute, true, faceIndexed ? numFaces * 3 : numVertices);
        mesh.SetAttributeElementType(vertexColorsAttId,
                                     draco::MeshAttributeElementType::MESH_VERTEX_ATTRIBUTE);
    }

    // if (numVertexConfidences > 0)
    // {
    //     draco::GeometryAttribute attribute;
    //     attribute.Init(draco::GeometryAttribute::GENERIC, nullptr, 1, draco::DT_FLOAT32, false,
    //                    sizeof(float), 0);
    //     vertexConfidencesAttId =
    //         mesh.AddAttribute(attribute, true, faceIndexed ? numFaces * 3 : numVertexConfidences);
    //     mesh.SetAttributeElementType(vertexConfidencesAttId,
    //                                  draco::MeshAttributeElementType::MESH_VERTEX_ATTRIBUTE);

    //     draco::AttributeMetadata* attributeMeta = new draco::AttributeMetadata();
    //     attributeMeta->set_att_unique_id(vertexConfidencesAttId);
    //     attributeMeta->AddEntryString("name", "confidence");

    //     metadata->AddAttributeMetadata(std::unique_ptr<draco::AttributeMetadata>(attributeMeta));
    // }

    // if (numVertexIntensities > 0)
    // {
    //     draco::GeometryAttribute attribute;
    //     attribute.Init(draco::GeometryAttribute::GENERIC, nullptr, 1, draco::DT_FLOAT32, false,
    //                    sizeof(float), 0);
    //     vertexIntensitiesAttId =
    //         mesh.AddAttribute(attribute, true, faceIndexed ? numFaces * 3 : numVertexIntensities);
    //     mesh.SetAttributeElementType(vertexIntensitiesAttId,
    //                                  draco::MeshAttributeElementType::MESH_VERTEX_ATTRIBUTE);

    //     draco::AttributeMetadata* attributeMeta = new draco::AttributeMetadata();
    //     attributeMeta->set_att_unique_id(vertexIntensitiesAttId);
    //     attributeMeta->AddEntryString("name", "intensity");

    //     metadata->AddAttributeMetadata(std::unique_ptr<draco::AttributeMetadata>(attributeMeta));
    // }

    if (vertexTextureCoordinates)
    {
        draco::GeometryAttribute attribute;
        attribute.Init(draco::GeometryAttribute::TEX_COORD, nullptr, 3, draco::DT_FLOAT32, false,
                       sizeof(float) * 3, 0);
        vertexTextureCoordinatesAttId = mesh.AddAttribute(
            attribute, true, faceIndexed ? numFaces * 3 : numVertexTextureCoordinates);
        mesh.SetAttributeElementType(vertexTextureCoordinatesAttId,
                                     draco::MeshAttributeElementType::MESH_VERTEX_ATTRIBUTE);
    }

    if (faceMaterialIndices)
    {
        draco::GeometryAttribute attribute;
        attribute.Init(draco::GeometryAttribute::GENERIC, nullptr, 1, draco::DT_UINT32, false,
                       sizeof(uint32_t), 0);
        faceMaterialIndicesAttId =
            mesh.AddAttribute(attribute, true, faceIndexed ? numFaces * 3 : numFaces);
        mesh.SetAttributeElementType(faceMaterialIndicesAttId,
                                     draco::MeshAttributeElementType::MESH_FACE_ATTRIBUTE);

        draco::AttributeMetadata* attributeMeta = new draco::AttributeMetadata();
        attributeMeta->set_att_unique_id(faceMaterialIndicesAttId);
        attributeMeta->AddEntryString("name", "materialindex");

        metadata->AddAttributeMetadata(std::unique_ptr<draco::AttributeMetadata>(attributeMeta));
    }

    // apply materials
    if (faceMaterialIndices && materials.size())
    {
        std::vector<int32_t> materialData;

        for (int i = 0; i < materials.size(); i++)
        {
            boost::optional<TextureHandle> opt_texture_index = materials[i].m_texture;
            boost::optional<Rgb8Color> opt_mat_color = materials[i].m_color;
            if(opt_mat_color)
            {
                Rgb8Color mat_color = *opt_mat_color;
                materialData.push_back(mat_color[0]);
                materialData.push_back(mat_color[1]);
                materialData.push_back(mat_color[2]);
            }
            else
            {
                // Push back default color if none defined
                materialData.push_back(126);
                materialData.push_back(126);
                materialData.push_back(126);
            }
            
            if(opt_texture_index)
            {
                TextureHandle handle = *opt_texture_index;
                materialData.push_back(handle.idx());
            }
            else
            {
                // Default 
                materialData.push_back(-1);
            }
            
            
        }
        metadata->AddEntryIntArray("material", materialData);

        // textures
        if (textures.size())
        {
            std::vector<int32_t> textureValue;
            saveTextures(&textureValue, textures);
            metadata->AddEntryIntArray("texture", textureValue);
        }
    }

    // apply attributes
    if (faceIndexed)
    {
        draco::Mesh::Face face;
        for (draco::FaceIndex i(0); i < numFaces; i++)
        {
            // TODO: check for illegal accesses of original arrays using the array size

            // face
            face[0] = draco::PointIndex(i.value() * 3 + 0);
            face[1] = draco::PointIndex(i.value() * 3 + 1);
            face[2] = draco::PointIndex(i.value() * 3 + 2);
            mesh.SetFace(i, face);

            // positions
            mesh.attribute(verticesAttId)
                ->SetAttributeValue(draco::AttributeValueIndex(face[0].value()),
                                    vertices.get() + faces[3 * i.value() + 0] * 3);
            mesh.attribute(verticesAttId)
                ->SetAttributeValue(draco::AttributeValueIndex(face[1].value()),
                                    vertices.get() + faces[3 * i.value() + 1] * 3);
            mesh.attribute(verticesAttId)
                ->SetAttributeValue(draco::AttributeValueIndex(face[2].value()),
                                    vertices.get() + faces[3 * i.value() + 2] * 3);

            // normals
            if (hasFaceNormals)
            {
                mesh.attribute(vertexNormalsAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[0].value()),
                                        vertexNormals.get() + faces[3 * i.value() + 0] * 3);
                mesh.attribute(vertexNormalsAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[1].value()),
                                        vertexNormals.get() + faces[3 * i.value() + 1] * 3);
                mesh.attribute(vertexNormalsAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[2].value()),
                                        vertexNormals.get() + faces[3 * i.value() + 2] * 3);
            }

            // colors
            if (hasVertexColors)
            {
                for (int j = 0; j < 3; j++)
                {
                    unsigned char color[3];
                    color[0] = vertexColors[3 * i.value()];
                    color[1] = vertexColors[3 * i.value() + 1];
                    color[2] = vertexColors[3 * i.value() + 2];
                    mesh.attribute(vertexColorsAttId)
                        ->SetAttributeValue(draco::AttributeValueIndex(face[j].value()), color);
                }
            }

            // // confidences
            // if (numVertexConfidences > 0)
            // {
            //     mesh.attribute(vertexConfidencesAttId)
            //         ->SetAttributeValue(draco::AttributeValueIndex(face[0].value()),
            //                             vertexConfidences.get() + faces[3 * i.value() + 0]);
            //     mesh.attribute(vertexConfidencesAttId)
            //         ->SetAttributeValue(draco::AttributeValueIndex(face[1].value()),
            //                             vertexConfidences.get() + faces[3 * i.value() + 1]);
            //     mesh.attribute(vertexConfidencesAttId)
            //         ->SetAttributeValue(draco::AttributeValueIndex(face[2].value()),
            //                             vertexConfidences.get() + faces[3 * i.value() + 2]);
            // }

            // // intensities
            // if (numVertexIntensities > 0)
            // {
            //     mesh.attribute(vertexIntensitiesAttId)
            //         ->SetAttributeValue(draco::AttributeValueIndex(face[0].value()),
            //                             vertexIntensities.get() + faces[3 * i.value() + 0]);
            //     mesh.attribute(vertexIntensitiesAttId)
            //         ->SetAttributeValue(draco::AttributeValueIndex(face[1].value()),
            //                             vertexIntensities.get() + faces[3 * i.value() + 1]);
            //     mesh.attribute(vertexIntensitiesAttId)
            //         ->SetAttributeValue(draco::AttributeValueIndex(face[2].value()),
            //                             vertexIntensities.get() + faces[3 * i.value() + 2]);
            // }

            // texture coordinates
            if (vertexTextureCoordinates)
            {
                mesh.attribute(vertexTextureCoordinatesAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[0].value()),
                                        vertexTextureCoordinates.get() +
                                            faces[3 * i.value() + 0] * 3);
                mesh.attribute(vertexTextureCoordinatesAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[1].value()),
                                        vertexTextureCoordinates.get() +
                                            faces[3 * i.value() + 1] * 3);
                mesh.attribute(vertexTextureCoordinatesAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[2].value()),
                                        vertexTextureCoordinates.get() +
                                            faces[3 * i.value() + 2] * 3);
            }

            // apply materialindices
            if (faceMaterialIndices)
            {
                mesh.attribute(faceMaterialIndicesAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[0].value()),
                                        faceMaterialIndices.get() + i.value());
                mesh.attribute(faceMaterialIndicesAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[1].value()),
                                        faceMaterialIndices.get() + i.value());
                mesh.attribute(faceMaterialIndicesAttId)
                    ->SetAttributeValue(draco::AttributeValueIndex(face[2].value()),
                                        faceMaterialIndices.get() + i.value());
            }
        }
    }
    else
    {
        // apply positions
        for (draco::AttributeValueIndex i(0); i < numVertices; i++)
        {
            mesh.attribute(verticesAttId)->SetAttributeValue(i, vertices.get() + i.value() * 3);
        }

        // apply normals
        for (draco::AttributeValueIndex i(0); i < numVertices; i++)
        {
            mesh.attribute(vertexNormalsAttId)
                ->SetAttributeValue(i, vertexNormals.get() + i.value() * 3);
        }

        // apply colors
        for (draco::AttributeValueIndex i(0); i < numVertices; i++)
        {
            unsigned char color[3];
            color[0] = vertexColors[3 * i.value()];
            color[1] = vertexColors[3 * i.value() + 1];
            color[2] = vertexColors[3 * i.value() + 2];
            mesh.attribute(vertexColorsAttId)->SetAttributeValue(i, color);
        }

        // // apply confidences
        // for (draco::AttributeValueIndex i(0); i < numVertexConfidences; i++)
        // {
        //     mesh.attribute(vertexConfidencesAttId)
        //         ->SetAttributeValue(i, vertexConfidences.get() + i.value());
        // }

        // // apply intensities
        // for (draco::AttributeValueIndex i(0); i < numVertexIntensities; i++)
        // {
        //     mesh.attribute(vertexIntensitiesAttId)
        //         ->SetAttributeValue(i, vertexIntensities.get() + i.value());
        // }

        // apply faces
        draco::Mesh::Face face;
        for (draco::FaceIndex i(0); i < numFaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                face[j] = faces[3 * i.value() + j];
            }
            mesh.SetFace(i, face);
        }
    }

    mesh.AddMetadata(std::unique_ptr<draco::GeometryMetadata>(metadata));

    // Encode Data
    std::unique_ptr<draco::EncoderBuffer> buffer(new draco::EncoderBuffer());
    auto                                  status = encoder.EncodeMeshToBuffer(mesh, buffer.get());

    if (!status.ok())
    {
        std::cerr << "An error occurred:"
                  << " " << status.error_msg_string() << std::endl;
        return std::unique_ptr<draco::EncoderBuffer>(nullptr);
    }

    return buffer;
}

std::unique_ptr<draco::EncoderBuffer> encodeDraco(ModelPtr                   modelPtr,
                                                  draco::EncodedGeometryType type)
{
    draco::Encoder encoder;
    // configure encoder
    encoder.SetSpeedOptions(0, 0);

    if (type == draco::TRIANGULAR_MESH)
    {
        return encodeMesh(modelPtr, encoder);
    }
    else if (type == draco::POINT_CLOUD)
    {
        // configure this only for pointclounds because it causes loss in visuals of faces on a
        // plain surface
        encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 12);
        encoder.SetAttributeQuantization(draco::GeometryAttribute::TEX_COORD, 12);
        encoder.SetAttributeQuantization(draco::GeometryAttribute::NORMAL, 10);
        encoder.SetAttributeQuantization(draco::GeometryAttribute::GENERIC, 8);

        return encodePointCloud(modelPtr, encoder);
    }

    return std::unique_ptr<draco::EncoderBuffer>(nullptr);
}

} // namespace lvr