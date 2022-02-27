#include "RayCastingTexturizer.hpp"
#include "lvr2/algorithm/raycasting/BVHRaycaster.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "lvr2/util/Util.hpp"
#include "lvr2/io/baseio/PLYIO.hpp"
#include <fstream>
#include <numeric>

namespace lvr2
{

const Vector3f DEBUG_ORIGIN(0, 0, 10);

template <typename BaseVecT>
RayCastingTexturizer<BaseVecT>::RayCastingTexturizer(
    float texelMinSize,
    int texMinClusterSize,
    int texMaxClusterSize,
    const BaseMesh<BaseVector<float>>& mesh,
    const ClusterBiMapPtr<FaceHandle> clusters,
    ScanProjectPtr project
): Texturizer<BaseVecT>(texelMinSize, texMinClusterSize, texMaxClusterSize)
 , m_project(project), m_debug(mesh)
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

    for (auto vertexH: mesh.vertices())
    {
        auto v = mesh.getVertexPosition(vertexH);
        vertices.push_back(v.x);
        vertices.push_back(v.y);
        vertices.push_back(v.z);
    }

    // Build vertex and face array
    for (auto face: mesh.faces())
    {
        m_embreeToHandle.insert({faceIndices.size() / 3, face});
        if ((faceIndices.size() / 3) != face.idx())
        {
            std::cout << "FaceIndex to Handle missmatch: " << faceIndices.size()/3 << " vs. " << face.idx() << std::endl;
        }
        auto faceVertices = mesh.getVerticesOfFace(face);
        for (auto vertex: faceVertices)
        {
            faceIndices.push_back(vertex.idx());
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


void setPixel(size_t index, Texture& tex, uint8_t r, uint8_t g, uint8_t b)
{
    tex.m_data[3 * index + 0] = r;
    tex.m_data[3 * index + 1] = g;
    tex.m_data[3 * index + 2] = b;
}

void setPixel(uint16_t x, uint16_t y, Texture& tex, uint8_t r, uint8_t g, uint8_t b)
{
    size_t index = (y * tex.m_width) + x;
    setPixel(index, tex, r, g, b);
}

template <typename BaseVecT>
void RayCastingTexturizer<BaseVecT>::DEBUGDrawBorder(TextureHandle texH, const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect, ClusterHandle clusterH)
{
    Texture& tex = this->m_textures[texH];
    // Draw in vertices of cluster
    for (auto face: m_clusters->getCluster(clusterH))
    {
        for (auto vertex: m_debug.getVerticesOfFace(face))
        {
            IntersectionT intersection;
            BaseVecT pos = m_debug.getVertexPosition(vertex);
            Vector3f direction = (Vector3f(pos.x, pos.y, pos.z) - DEBUG_ORIGIN).normalized();
            if (!m_tracer->castRay(DEBUG_ORIGIN, direction, intersection)) continue;

            if (m_clusters->getClusterH(FaceHandle(intersection.face_id)) != clusterH) continue;
            

            
            TexCoords uv = this->calculateTexCoords(texH, boundingRect, pos);
            uint16_t x = uv.u * (tex.m_width - 1);
            uint16_t y = uv.v * (tex.m_height - 1);
            setPixel(x, y, tex, 0, 0, 0);
        }
    }
}

template <typename BaseVecT>
TextureHandle RayCastingTexturizer<BaseVecT>::generateTexture(
    int index,
    const PointsetSurface<BaseVecT>&,
    const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect,
    ClusterHandle clusterH
)
{

    // Calculate the texture size
    unsigned short int sizeX = ceil(abs(boundingRect.m_maxDistA - boundingRect.m_minDistA) / this->m_texelSize);
    unsigned short int sizeY = ceil(abs(boundingRect.m_maxDistB - boundingRect.m_minDistB) / this->m_texelSize);
    // Make sure the texture contains at least 1 pixel
    sizeX = std::max<unsigned short int>({sizeX, 1});
    sizeY = std::max<unsigned short int>({sizeY, 1});
    
    TextureHandle texH = this->m_textures.push(
        this->initTexture(
            index,
            sizeX,
            sizeY,
            3,
            1,
            this->m_texelSize
        )
    );

    Texture& tex = this->m_textures[texH];

    size_t num_pixel = sizeX * sizeY;

    if (!m_project)
    {
        std::cout << timestamp << "[RaycastingTexturizer] No scan project set, cannot texturize cluster" << std::endl;
        //TODO: add return back in // return texH;
    }

    // if (index != 38)
    // return texH;

    // List of uv coordinates
    std::vector<TexCoords> uvCoords = this->calculateUVCoordsPerPixel(tex);
    // List of 3D points corresponding to uv coords
    std::vector<BaseVecT> points = this->calculate3DPointsPerPixel(uvCoords, boundingRect);
    // A list of booleans indicating wether the point is visible
    std::vector<bool> visible = this->calculateVisibilityPerPixel(Vector3f(0, 0, 10), points, clusterH);
    
    this->DEBUGDrawBorder(texH, boundingRect, clusterH);

    cv::namedWindow("debug", cv::WINDOW_NORMAL);
    cv::Mat cvTex(tex.m_height, tex.m_width, CV_8UC3, tex.m_data);
    cv::imshow("debug", cvTex);
    cv::waitKey(0);

    for (size_t i = 0; i < num_pixel; i++)
    {
        //get pixel coords
        uint16_t pX = i % tex.m_width;
        uint16_t pY = i / tex.m_width;

        float u = (float) pX / (tex.m_width - 1);
        float v = (float) pY / (tex.m_height - 1);

        BaseVecT point = this->calculateTexCoordsInv( texH, boundingRect, TexCoords(u,v));
        Vector3f direction = (Vector3f(point.x, point.y, point.z) - DEBUG_ORIGIN).normalized();

        IntersectionT intersection;
        if (this->m_tracer->castRay(DEBUG_ORIGIN, direction, intersection))
        {
            if (m_clusters->getClusterH(FaceHandle(intersection.face_id)) == clusterH)
            {
                setPixel(pX, pY, tex, 0, 255, 0);
            }
        }
    }

    cvTex = cv::Mat(tex.m_height, tex.m_width, CV_8UC3, tex.m_data);
    cv::imshow("debug", cvTex);
    cv::waitKey(0);

    return texH;
}

template <typename BaseVecT>
template <typename... Args>
Texture RayCastingTexturizer<BaseVecT>::initTexture(Args&&... args) const
{
    Texture ret(std::forward<Args>(args)...);

    ret.m_layerName = "RGB";
    size_t num_pixel = ret.m_width * ret.m_height;

    // Init white
    for (int i = 0; i < num_pixel; i++)
    {
        ret.m_data[i * 3 + 0] = 255;
        ret.m_data[i * 3 + 1] = 0;
        ret.m_data[i * 3 + 2] = 0;
    }

    return std::move(ret);
}

template <typename BaseVecT>
std::vector<TexCoords> RayCastingTexturizer<BaseVecT>::calculateUVCoordsPerPixel(const Texture& tex) const
{
    std::vector<TexCoords> ret;
    ret.reserve(tex.m_width * tex.m_height);

    for (size_t x = 0; x < tex.m_width; x++)
    {
        for (size_t y = 0; y < tex.m_height; y++)
        {
            float u = (float) x / tex.m_width;
            float v = (float) y / tex.m_height;
            ret.push_back(TexCoords(u, v));
        }
    }

    return std::move(ret);
}

template <typename BaseVecT>
std::vector<BaseVecT> RayCastingTexturizer<BaseVecT>::calculate3DPointsPerPixel(
    const std::vector<TexCoords>& texel,
    const BoundingRectangle<typename BaseVecT::CoordType>& bb)
{
    std::vector<BaseVecT> ret(texel.size());
    // Calculate 3D points
    std::transform(
        texel.begin(),
        texel.end(),
        ret.begin(),
        [this, bb](const TexCoords& uv)
        {
            return this->calculateTexCoordsInv(TextureHandle(), bb, uv);
        });

    return std::move(ret);
}

template <typename BaseVecT>
std::vector<bool> RayCastingTexturizer<BaseVecT>::calculateVisibilityPerPixel(
    const Vector3f from,
    const std::vector<BaseVecT>& to,
    const ClusterHandle cluster) const
{
    std::vector<bool> ret(to.size());

    std::vector<Vector3f>       directions(ret.size());
    std::vector<IntersectionT>  intersections(ret.size());
    std::vector<uchar>          hits(ret.size());
    // Calculate directions
    std::transform(
        to.begin(),
        to.end(),
        directions.begin(),
        [from](const auto& point)
        {
            Vector3f p(point.x, point.y, point.z);
            return (p - from).normalized();
        }
    );
    // Cast Rays from Camera to points and check for visibility
    this->m_tracer->castRays(from, directions, intersections, hits);

    auto ret_it = ret.begin();
    auto int_it = intersections.begin();
    auto hit_it = hits.begin();
    auto dir_it = directions.begin();

    // For each
    for(;ret_it != ret.end(); ++ret_it, ++int_it, ++hit_it, ++dir_it)
    {
        if (*hit_it)
        {
            // check if the hit face belongs to the cluster we are texturzing
            FaceHandle              fHandle = m_embreeToHandle.at(int_it->face_id);
            ClusterHandle           cHandle = m_clusters->getClusterH(fHandle);
            if (cHandle == cluster)
            {
                *ret_it = true;
                continue;
            }
        }
        *ret_it = false;
    }
    return std::move(ret);
}

} // namespace lvr2

