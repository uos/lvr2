// lvr2 includes
#include "RaycastingTexturizer.hpp"
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

using Eigen::Quaterniond;
using Eigen::Quaternionf;
using Eigen::AngleAxisd;
using Eigen::Translation3f;

std::ofstream timings("timings.log");

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
 , m_debug(mesh)
{
    this->setGeometry(mesh);
    this->setClusters(clusters);
    this->setScanProject(project);
}

template <typename BaseVecT>
void RaycastingTexturizer<BaseVecT>::setGeometry(const BaseMesh<BaseVecT>& mesh)
{
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
void RaycastingTexturizer<BaseVecT>::DEBUGDrawBorder(TextureHandle texH, const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect, ClusterHandle clusterH)
{
    Texture& tex = this->m_textures[texH];
    // Draw in vertices of cluster
    for (auto face: m_clusters.getCluster(clusterH))
    {
        for (auto vertex: m_debug.getVerticesOfFace(face))
        {
            IntersectionT intersection;
            BaseVecT pos = m_debug.getVertexPosition(vertex);
            Vector3f direction = (Vector3f(pos.x, pos.y, pos.z) - DEBUG_ORIGIN).normalized();
            if (!m_tracer->castRay(DEBUG_ORIGIN, direction, intersection)) continue;

            if (m_clusters.getClusterH(FaceHandle(intersection.face_id)) != clusterH) continue;
            
            TexCoords uv = this->calculateTexCoords(texH, boundingRect, pos);
            uint16_t x = uv.u * (tex.m_width - 1);
            uint16_t y = uv.v * (tex.m_height - 1);
            setPixel(x, y, tex, 0, 0, 0);
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
    // Number of pixels in the texture
    size_t numPixel = sizeX * sizeY;

    if (m_images.size() == 0)
    {
        std::cout << timestamp << "[RaycastingTexturizer] No images set, cannot texturize cluster" << std::endl;
        return texH;
    }

    this->DEBUGDrawBorder(texH, boundingRect, clusterH);

    // List of uv coordinates
    std::vector<TexCoords> uvCoords = this->calculateUVCoordsPerPixel(tex);
    // List of 3D points corresponding to uv coords
    std::vector<Vector3f> points = this->calculate3DPointsPerPixel(uvCoords, boundingRect);
    // List of booleans indicating if a texel is already Texturized
    std::vector<bool> texturized(numPixel, false);
    // List containing the useable images for texturing the cluster
    std::vector<ImageInfo> images = this->rankImagesForCluster(boundingRect);

    // cv::namedWindow("Texture", cv::WINDOW_KEEPRATIO);
    #pragma omp parallel for
    for (size_t i = 0; i < numPixel; i++)
    {
        // if (i % 10 == 0)
        // {
            // cv::Mat cvTexture(tex.m_height, tex.m_width, CV_8UC3, tex.m_data);
            // cv::Mat converted;
            // cv::cvtColor(cvTexture, converted, cv::COLOR_BGR2RGB);
            // converted.at<cv::Vec3b>(i) = cv::Vec3b(0,0,0);
            // cv::imshow("Texture", converted);
            // cv::waitKey(0);
        // }
        

        // Calculate x,y
        size_t x = i % tex.m_width;
        size_t y = i / tex.m_width;
        // Calculate tex uv
        double u = ((float) x) / (tex.m_width - 1);
        double v = ((float) y) / (tex.m_height - 1);
        // Calculate 3D point
        BaseVecT tmp = this->calculateTexCoordsInv(texH, boundingRect, TexCoords(u, v));
        Vector3f point(tmp.x, tmp.y, tmp.z);
        for (ImageInfo img: images)
        {
            // Cast ray to point
            IntersectionT intersection;
            bool hit = this->m_tracer->castRay( img.translation.vector(), (point - img.translation.vector()).normalized(), intersection);
            // Did not hit anything
            if (!hit) continue;
            // Dit not hit the cluster we are interested in
            FaceHandle faceH = m_embreeToHandle.at(intersection.face_id);
            OptionalClusterHandle hitClusterHOpt = m_clusters.getClusterOf(faceH);
            if (!(hitClusterHOpt && (hitClusterHOpt.unwrap() == clusterH))) continue;
            // Transform the point to camera space
            Vector3f transformedPoint = img.inverse_rotation * (point - img.translation.vector());
            // If the point is behind the camera skip this image
            if (transformedPoint.z() <= 0) continue;
            // Project the point
            Vector2f uv = img.model.projectPoint(transformedPoint);
            double imgU = uv.x();
            double imgV = uv.y();
            undistorted_to_distorted_uv(imgU, imgV, img.model);
            size_t imgX = std::floor(imgU);
            size_t imgY = std::floor(imgV);
            // Skip if the projected pixel is outside the camera image
            if (imgX < 0 || imgY < 0 || imgX >= img.image->image.cols || imgY >= img.image->image.rows) continue;
            // Calculate color
            const cv::Vec3b& color = img.image->image.template at<cv::Vec3b>(imgY, imgX);
            setPixel(i, tex, color[0], color[1], color[2]);
            // After the pixel is texturized we are done
            break;
        }
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
std::vector<TexCoords> RaycastingTexturizer<BaseVecT>::calculateUVCoordsPerPixel(const Texture& tex) const
{
    std::vector<TexCoords> ret;
    ret.reserve(tex.m_width * tex.m_height);

    for (size_t y = 0; y < tex.m_height; y++)
    {
        for (size_t x = 0; x < tex.m_width; x++)
        {
            float u = ((float) x + 0.5f) / (tex.m_width  - 1);
            float v = ((float) y + 0.5f) / (tex.m_height - 1);
            ret.push_back(TexCoords(u, v));
        }
    }

    return std::move(ret);
}

template <typename BaseVecT>
std::vector<Vector3f> RaycastingTexturizer<BaseVecT>::calculate3DPointsPerPixel(
    const std::vector<TexCoords>& texel,
    const BoundingRectangle<typename BaseVecT::CoordType>& bb)
{
    std::vector<Vector3f> ret(texel.size());
    // Calculate 3D points
    std::transform(
        texel.begin(),
        texel.end(),
        ret.begin(),
        [this, bb](const TexCoords& uv)
        {
            BaseVecT tmp = this->calculateTexCoordsInv(TextureHandle(), bb, uv);

            Vector3f ret;
            ret(0) = tmp[0];
            ret(1) = tmp[1];
            ret(2) = tmp[2];
            
            return ret;
        });

    return std::move(ret);
}

template <typename BaseVecT>
std::vector<bool> RaycastingTexturizer<BaseVecT>::calculateVisibilityPerPixel(
    const Vector3f from,
    const std::vector<Vector3f> &to,
    const std::vector<bool> &texturized,
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
            // Vector3f direction = (point - from).normalized();
            // old_points << direction.x() << " " << direction.y() << " " << direction.z() << std::endl;
            return (point - from).normalized();
        }
    );
    // Cast Rays from Camera to points and check for visibility
    // this->m_tracer->castRays(from, directions, intersections, hits);
    #pragma omp parallel for
    for (size_t i = 0; i < to.size(); i++)
    {
        if (texturized[i])
        {
            hits[i] = false;
            continue;
        }
        
        hits[i] = this->m_tracer->castRay(from, directions[i], intersections[i]);
    }


    auto ret_it = ret.begin();
    auto int_it = intersections.begin();
    auto hit_it = hits.begin();

    // For each
    for(;ret_it != ret.end(); ++ret_it, ++int_it, ++hit_it)
    {
        if (*hit_it)
        {
            // check if the hit face belongs to the cluster we are texturzing
            FaceHandle              fHandle = m_embreeToHandle.at(int_it->face_id);
            ClusterHandle           cHandle = m_clusters.getClusterH(fHandle);
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

} // namespace lvr2

