#include "RayCastingTexturizer.hpp"

namespace lvr2
{

template <typename BaseVecT>
RayCastingTexturizer<BaseVecT>::RayCastingTexturizer(
    float texelMinSize,
    int texMinClusterSize,
    int texMaxClusterSize,
    const BaseMesh<BaseVector<float>>& mesh,
    const ClusterBiMapPtr<FaceHandle> clusters
): Texturizer<BaseVecT>(texelMinSize, texMinClusterSize, texMaxClusterSize)
{
    this->setGeometry(mesh);
    this->setClusters(clusters);
}

template <typename BaseVecT>
void RayCastingTexturizer<BaseVecT>::setGeometry(const BaseMesh<BaseVecT>& mesh)
{
    m_embreeToHandle.clear();
    MeshBufferPtr buffer = std::make_shared<MeshBuffer>();
    std::vector<float> vertices;
    std::vector<unsigned int> faceIndices;

    // Build vertex and face array
    for (auto face: mesh.faces())
    {
        m_embreeToHandle.insert({faceIndices.size() / 3, face});
        auto faceVertices = mesh.getVertexPositionsOfFace(face);
        for (auto vertex: faceVertices)
        {
            faceIndices.push_back(vertices.size() / 3);
            vertices.push_back(vertex.x);
            vertices.push_back(vertex.y);
            vertices.push_back(vertex.z);
        }
        
    }

    buffer->setVertices(Util::convert_vector_to_shared_array(vertices), vertices.size() / 3);
    buffer->setFaceIndices(Util::convert_vector_to_shared_array(faceIndices), faceIndices.size() / 3);

    m_tracer = std::make_shared<EmbreeRaycaster<IntersectionT>>(buffer);
}

template <typename BaseVecT>
void RayCastingTexturizer<BaseVecT>::setClusters(const ClusterBiMapPtr<FaceHandle> clusters)
{
    this->m_clusters = clusters;
}

template <typename BaseVecT>
TextureHandle RayCastingTexturizer<BaseVecT>::generateTexture(
    int index,
    const PointsetSurface<BaseVecT>&,
    const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect
)
{

    // Calculate the texture size
    unsigned short int sizeX = ceil(abs(boundingRect.m_maxDistA - boundingRect.m_minDistA) / this->m_texelSize);
    unsigned short int sizeY = ceil(abs(boundingRect.m_maxDistB - boundingRect.m_minDistB) / this->m_texelSize);
    
    Texture tex(
        index,
        sizeX,
        sizeY,
        3,
        1,
        this->m_texelSize
    );
    tex.m_layerName = "RGB";

    size_t num_pixel = sizeX * sizeY;

    // Init red
    for (int i = 0; i < num_pixel; i++)
    {
        tex.m_data[i * 3 + 0] = 255;
        tex.m_data[i * 3 + 1] = 0;
        tex.m_data[i * 3 + 2] = 0;
    }

    // List of uv coordinates
    std::vector<TexCoords> uvCoords;
    uvCoords.reserve(num_pixel);
    // List of 3D points corresponding to uv coords
    std::vector<BaseVecT> points(num_pixel);

    // Calculate uv coords
    for (size_t x = 0; x < sizeX; x++)
    {
        for (size_t y = 0; y < sizeY; y++)
        {
            float u = (float) x / sizeX;
            float v = (float) y / sizeY;
            uvCoords.push_back(TexCoords(u, v));
        }
    }
    // Calculate 3D points
    std::transform(
        uvCoords.begin(),
        uvCoords.end(),
        points.begin(),
        [this, boundingRect](const TexCoords& uv)
        {
            return this->calculateTexCoordsInv(TextureHandle(), boundingRect, uv);
        });

    // TODO: Cast Rays from Camera to points
    // TODO: Project all points which hit the right face

    return this->m_textures.push(std::move(tex));
}




} // namespace lvr2

