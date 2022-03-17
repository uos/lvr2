// lvr2 includes
#include "RaycastingTexturizer.hpp"
#include "lvr2/texture/Triangle.hpp"
#include "lvr2/util/Util.hpp"
#include "lvr2/util/TransformUtils.hpp"
#include "lvr2/io/baseio/PLYIO.hpp"
#include "lvr2/util/Util.hpp"

// opencv includes
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// std includes
#include <fstream>
#include <numeric>
#include <variant>
#include <atomic>

using Eigen::Quaterniond;
using Eigen::Quaternionf;
using Eigen::AngleAxisd;
using Eigen::Translation3f;
using Eigen::Vector2i;

std::ofstream timings("timings.log");
std::ofstream barycentric_sum("bary.txt");

namespace lvr2
{

const Vector3f DEBUG_ORIGIN(0, 0, 1);

template <typename BaseVecT>
RaycastingTexturizer<BaseVecT>::RaycastingTexturizer(
    float texelMinSize,
    int texMinClusterSize,
    int texMaxClusterSize,
    const BaseMesh<BaseVector<float>>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const ScanProjectPtr project
): Texturizer<BaseVecT>(texelMinSize, texMinClusterSize, texMaxClusterSize)
 , m_mesh(mesh)
{
    this->setGeometry(mesh);
    this->setClusters(clusters);
    this->setScanProject(project);
}

template <typename BaseVecT>
void RaycastingTexturizer<BaseVecT>::setGeometry(const BaseMesh<BaseVecT>& mesh)
{
    m_mesh = std::cref(mesh);
    m_embreeToHandle.clear();
    MeshBufferPtr buffer = std::make_shared<MeshBuffer>();
    std::vector<float> vertices;
    std::vector<unsigned int> faceIndices;

    std::map<VertexHandle, size_t> vertexHToIndex;

    for (auto vertexH: mesh.vertices())
    {
        vertexHToIndex.insert({vertexH, vertices.size() / 3});
        auto v = mesh.getVertexPosition(vertexH);
        vertices.push_back(v.x);
        vertices.push_back(v.y);
        vertices.push_back(v.z);
    }

    // Build vertex and face array
    for (auto face: mesh.faces())
    {
        m_embreeToHandle.insert({faceIndices.size() / 3, face});
        auto faceVertices = mesh.getVerticesOfFace(face);
        for (auto vertexH: faceVertices)
        {
            faceIndices.push_back(vertexHToIndex.at(vertexH));
        }
    }

    buffer->setVertices(Util::convert_vector_to_shared_array(vertices), vertices.size() / 3);
    buffer->setFaceIndices(Util::convert_vector_to_shared_array(faceIndices), faceIndices.size() / 3);

    m_tracer = std::make_shared<EmbreeRaycaster<IntersectionT>>(buffer);
}

template <typename BaseVecT>
void RaycastingTexturizer<BaseVecT>::setClusters(const ClusterBiMap<FaceHandle>& clusters)
{
    this->m_clusters = clusters;
}

template <typename BaseVecT>
void RaycastingTexturizer<BaseVecT>::setScanProject(const ScanProjectPtr project)
{
    m_images.clear();
    // Iterate all Images and add them to the m_images
    for (auto position: project->positions)
    {
        for (auto camera: position->cameras)
        {
            std::queue<std::pair<CameraImageOrGroup, ImageInfo>> processList;
            for (auto elem: camera->images)
            {
                ImageInfo info;
                info.model = camera->model;
                auto pair = std::make_pair(elem, info);
                processList.push(pair);
            }
            
            
            while (!processList.empty())
            {
                // Get the element to process
                auto [imgOrGrp, info] = processList.front();                
                // Pop the element to be processed
                processList.pop();

                if (imgOrGrp.template is_type<CameraImageGroupPtr>())
                {   // Add all elements to process list
                    for (CameraImageOrGroup elem: imgOrGrp.template get<CameraImageGroupPtr>()->images)
                    {
                        processList.push({elem, info});
                    }
                }
                // If its an image add the transform, the image and the camera model to the list
                if (imgOrGrp.template is_type<CameraImagePtr>())
                {
                    // Set the image
                    info.image = imgOrGrp.template get<CameraImagePtr>();

                    // Calculate rotation of the image in world space
                    Quaterniond positionR(position->transformation.template topLeftCorner<3,3>());
                    Quaterniond cameraR(camera->transformation.template topLeftCorner<3,3>());
                    Quaterniond imageR(info.image->transformation.template topLeftCorner<3,3>());
                    /**
                     * The camera and image transformations are out of order because the image camera transform puts the Z axis straight ahead
                     * but the image rotation assumes Z is up.
                     */
                    Quaterniond rotation = positionR * imageR * cameraR;
                    info.rotation = rotation.cast<float>().normalized();

                    // Calculate Translation
                    Vector3d positionT(position->transformation.template topRightCorner<3,1>());
                    Vector3d cameraT(camera->transformation.template topRightCorner<3,1>());
                    Vector3d imageT(info.image->transformation.template topRightCorner<3,1>());
                    // Rotate current translation with the rotation from the next level and add the new translation part
                    Vector3d translation = (imageR * cameraT) + imageT;
                    translation = (positionR * translation) + positionT;
                    info.translation = Translation3f(translation.cast<float>());

                    // Precalculate inverse transforms
                    info.inverse_rotation = info.rotation.inverse().normalized();
                    info.inverse_translation = info.translation.inverse();

                    // === The camera intrinsics in the ringlok ScanProject are wrong === //
                    // === These are the correct values for the Riegl camera === //
                    info.model.fx = 2395.4336550315002;
                    info.model.fy = 2393.3126174899603;
                    info.model.cx = 3027.8728609530291;
                    info.model.cy = 2031.02743729632;

                    m_images.push_back(info);
                }   
            }
        }
    }

    std::cout << timestamp << "[RaycastingTexturizer] Loaded " << m_images.size() << " images" << std::endl;
}

inline Vector2i texelFromUV(const TexCoords& uv, const Texture& tex)
{
    size_t x = uv.u * tex.m_width;
    size_t y = uv.v * tex.m_height;
    x = std::min<size_t>({x, (size_t) tex.m_width - 1});
    y = std::min<size_t>({y, (size_t) tex.m_height - 1});
    return Vector2i(x, y);
}

inline TexCoords uvFromTexel(const Vector2i& texel, const Texture& tex)
{
    return TexCoords(
        ((float) texel.x() + 0.5f) / tex.m_width,
        ((float) texel.y() + 0.5f) / tex.m_height
    );
}

inline void setPixel(size_t index, Texture& tex, cv::Vec3b color)
{
    tex.m_data[3 * index + 0] = color[0];
    tex.m_data[3 * index + 1] = color[1];
    tex.m_data[3 * index + 2] = color[2];
}

inline void setPixel(uint16_t x, uint16_t y, Texture& tex, cv::Vec3b color)
{
    size_t index = (y * tex.m_width) + x;
    setPixel(index, tex, color);
}

inline void setPixel(TexCoords uv, Texture& tex, cv::Vec3b color)
{
    auto p = texelFromUV(uv, tex);
    setPixel(p.x(), p.y(), tex, color);
}

template <typename BaseVecT>
void RaycastingTexturizer<BaseVecT>::DEBUGDrawBorder(TextureHandle texH, const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect, ClusterHandle clusterH)
{
    Texture& tex = this->m_textures[texH];
    // Draw in vertices of cluster
    for (auto face: m_clusters.getCluster(clusterH))
    {
        for (auto vertex: m_mesh.get().getVerticesOfFace(face))
        {
            IntersectionT intersection;
            BaseVecT pos = m_mesh.get().getVertexPosition(vertex);
            Vector3f direction = (Vector3f(pos.x, pos.y, pos.z) - DEBUG_ORIGIN).normalized();
            if (!m_tracer->castRay(DEBUG_ORIGIN, direction, intersection)) continue;

            if (m_clusters.getClusterH(FaceHandle(intersection.face_id)) != clusterH) continue;
            
            TexCoords uv = this->calculateTexCoords(texH, boundingRect, pos);
            setPixel(uv, tex, cv::Vec3b(0, 0, 0));
        }
    }
}
// Completely redo this
void undistorted_to_distorted_uv(
    double &u,
    double &v,
    PinholeModel model)
{
    double x, y, ud, vd, r_2, r_4, r_6, r_8, fx, fy, Cx, Cy, k1, k2, k3, k4, p1, p2;
    fx = model.fx;
    fy = model.fy;
    Cx = model.cx;
    Cy = model.cy;

    k1 = model.distortionCoefficients[0];
    k2 = model.distortionCoefficients[1];
    p1 = model.distortionCoefficients[2];
    p2 = model.distortionCoefficients[3];
    k3 = model.distortionCoefficients[4];
    k4 = model.distortionCoefficients[5];
    
    x = (u - Cx)/fx;
    y = (v - Cy)/fy;

    r_2 = std::pow(std::atan(std::sqrt(std::pow(x, 2) + std::pow(y, 2))), 2);
    r_4 = std::pow(r_2, 2);
    r_6 = std::pow(r_2, 3);
    r_8 = std::pow(r_2, 4);

    ud = u + x*fx*(k1*r_2 + k2*r_4 + k3*r_6 + k4*r_8) + 2*fx*x*y*p1 + p2*fx*(r_2 + 2*std::pow(x, 2));
    vd = v + y*fy*(k1*r_2 + k2*r_4 + k3*r_6 + k4*r_8) + 2*fy*x*y*p2 + p1*fy*(r_2 + 2*std::pow(y, 2));

    u = ud;
    v = vd;
}


template <typename BaseVecT>
TextureHandle RaycastingTexturizer<BaseVecT>::generateTexture(
    int index,
    const PointsetSurface<BaseVecT>&,
    const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect,
    ClusterHandle clusterH
)
{
    Timestamp time_func;
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

    if (m_images.size() == 0)
    {
        std::cout << timestamp << "[RaycastingTexturizer] No images set, cannot texturize cluster" << std::endl;
        return texH;
    }

    this->DEBUGDrawBorder(texH, boundingRect, clusterH);

    // List containing the useable images for texturing the cluster
    std::vector<ImageInfo> images = this->rankImagesForCluster(boundingRect);
    // Paint all faces
    #pragma omp parallel for
    for (FaceHandle faceH: m_clusters.getCluster(clusterH))
    {
        this->paintTriangle(texH, faceH, boundingRect, images);
    }

    timings << "[" << index << "] Generation took " << time_func.getElapsedTimeInMs() << "ms ("  << time_func.getElapsedTimeInS() << "s)\n";
    return texH;
}

template <typename BaseVecT>
template <typename... Args>
Texture RaycastingTexturizer<BaseVecT>::initTexture(Args&&... args) const
{
    Texture ret(std::forward<Args>(args)...);

    ret.m_layerName = "RGB";
    size_t num_pixel = ret.m_width * ret.m_height;

    // Init red
    for (int i = 0; i < num_pixel; i++)
    {
        ret.m_data[i * 3 + 0] = 255;
        ret.m_data[i * 3 + 1] = 0;
        ret.m_data[i * 3 + 2] = 0;
    }

    return std::move(ret);
}

template <typename BaseVecT>
void RaycastingTexturizer<BaseVecT>::paintTriangle(
    TextureHandle texH,
    FaceHandle faceH,
    const BoundingRectangle<typename BaseVecT::CoordType>& bRect,
    const std::vector<ImageInfo>& images)
{
    // Texture
    Texture& tex = this->m_textures[texH];
    // Corners in 3D
    std::array<BaseVecT, 3UL> vertices = m_mesh.get().getVertexPositionsOfFace(faceH);
    // The triangle in 3D World space
    auto worldTriangle = Triangle(
        Vector3f(vertices[0].x, vertices[0].y, vertices[0].z),
        Vector3f(vertices[1].x, vertices[1].y, vertices[1].z),
        Vector3f(vertices[2].x, vertices[2].y, vertices[2].z)
    );
    // Vertices in texture uvs
    std::array<TexCoords, 3UL> triUV;
    triUV[0] = this->calculateTexCoords(texH, bRect, vertices[0]);
    triUV[1] = this->calculateTexCoords(texH, bRect, vertices[1]);
    triUV[2] = this->calculateTexCoords(texH, bRect, vertices[2]);
    // The triangle in uv space
    Triangle<Vector2f, float> uvTriangle(
        Vector2f(triUV[0].u, triUV[0].v),
        Vector2f(triUV[1].u, triUV[1].v),
        Vector2f(triUV[2].u, triUV[2].v)
    );
    // The triangle in texel space
    Triangle<Vector2i> texelTriangle(
        texelFromUV(triUV[0], tex),
        texelFromUV(triUV[1], tex),
        texelFromUV(triUV[2], tex)
    );
    // Determine texel bb
    auto [minP, maxP] = texelTriangle.getAABoundingBox();
    // Iterate bb and check if texel center is inside the triangle
    for (int y = minP.y(); y <= maxP.y(); y++ )
    {
        for (int x = minP.x(); x <= maxP.x(); x++)
        {
            auto tmp = uvFromTexel(Vector2i(x, y), tex);
            Vector2f pointUV(tmp.u, tmp.v);
            // Skip texel if not inside this triangle
            if (!uvTriangle.contains(pointUV)) continue;
            // Calc barycentric coordinates
            Vector3f barycentrics = uvTriangle.barycentric(pointUV);
            // Calculate 3D point using barycentrics
            Vector3f pointWorld = worldTriangle.point(barycentrics);
            // Set pixel color pixel
            this->paintTexel(texH, faceH, Vector2i(x, y), pointWorld, images);
        }
    }
}

template <typename BaseVecT>
void RaycastingTexturizer<BaseVecT>::paintTexel(
    TextureHandle texH,
    FaceHandle faceH,
    Vector2i texel,
    Vector3f point,
    const std::vector<ImageInfo>& images)
{
    for (ImageInfo img: images)
    {   
        // Check if the point is visible
        if (!this->isVisible(img.translation.vector(), point, faceH)) continue;

        cv::Vec3b color;
        // If the color could not be calculated process next image
        if (!this->calcPointColor(point, img, color)) continue;
        
        setPixel(texel.x(), texel.y(), this->m_textures[texH], color);
        
        // After the pixel is texturized we are done
        return;
    }
}

template <typename BaseVecT>
std::vector<typename RaycastingTexturizer<BaseVecT>::ImageInfo> RaycastingTexturizer<BaseVecT>::rankImagesForCluster(const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect) const
{
    // TODO: check if rect intersects frustum
    Vector3f normal = Vector3f(
        boundingRect.m_normal.getX(),
        boundingRect.m_normal.getY(),
        boundingRect.m_normal.getZ()
        );
    
    BaseVecT tmp = boundingRect.center();
    Vector3f center(tmp.x, tmp.y, tmp.z);

    std::vector<std::pair<ImageInfo, float>> ranked;

    std::transform(
        m_images.begin(),
        m_images.end(),
        std::back_insert_iterator(ranked),
        [normal, center](ImageInfo img)
        {
            // View vector in world coordinates
            Vector3f view = img.rotation * Vector3f::UnitZ();
            // Check if cluster is seen from the back
            if (normal.dot(view) < 0)
            {
                return std::make_pair(img, 0.0f);
            }
            // Direction vector from the camera to the center of the cluster
            Vector3f direction = (center - img.translation.vector()).normalized();

            // Cosine of the angle between the view vector of the image and the cluster normal
            float angle = view.dot(normal);
            // Preload the image if its not already loaded
            if(!img.image->loaded())
            {
                img.image->load();
            }
            return std::make_pair(img, angle);
        });

    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const auto& first, const auto& second)
        {
            return first.second > second.second;
        });

    auto is_valid = [](auto& elem){ return elem.second > 0.0f;};

    size_t numValidImages = std::count_if(ranked.begin(), ranked.end(), is_valid);
    auto pastValidImageIt = std::partition_point(ranked.begin(), ranked.end(), is_valid);
    std::vector<ImageInfo> ret;

    std::transform(
        ranked.begin(),
        pastValidImageIt,
        std::back_insert_iterator(ret),
        [](auto pair)
        {
            return pair.first;
        }
    );

    return ret;
}

template <typename BaseVecT>
bool RaycastingTexturizer<BaseVecT>::isVisible(Vector3f origin, Vector3f point, FaceHandle faceH) const
{
    // Cast ray to point
    IntersectionT intersection;
    bool hit = this->m_tracer->castRay( origin, (point - origin).normalized(), intersection);
    // Did not hit anything
    if (!hit) return false;
    // Dit not hit the cluster we are interested in
    FaceHandle hitFaceH = m_embreeToHandle.at(intersection.face_id);
    // Wrong face
    if (faceH != hitFaceH) return false;
    float dist = (intersection.point - point).norm();
    // Default
    return true;
}

template <typename BaseVecT>
bool RaycastingTexturizer<BaseVecT>::calcPointColor(Vector3f point, const ImageInfo& img, cv::Vec3b& color) const
{
    // Transform the point to camera space
    Vector3f transformedPoint = img.inverse_rotation * (point - img.translation.vector());
    // If the point is behind the camera no color will be extracted
    if (transformedPoint.z() <= 0) return false;

    // Project the point to the camera image
    Vector2f uv = img.model.projectPoint(transformedPoint);
    double imgU = uv.x();
    double imgV = uv.y();
    // Distort the uv coordinates
    undistorted_to_distorted_uv(imgU, imgV, img.model);
    size_t imgX = std::floor(imgU);
    size_t imgY = std::floor(imgV);

    // Skip if the projected pixel is outside the camera image
    if (imgX < 0 || imgY < 0 || imgX >= img.image->image.cols || imgY >= img.image->image.rows) return false;

    // Calculate color
    color = img.image->image.template at<cv::Vec3b>(imgY, imgX);
    return true;
}

} // namespace lvr2

