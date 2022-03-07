#include "RayCastingTexturizer.hpp"
#include "lvr2/algorithm/raycasting/BVHRaycaster.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "lvr2/util/Util.hpp"
#include "lvr2/util/TransformUtils.hpp"
#include "lvr2/io/baseio/PLYIO.hpp"
#include <fstream>
#include <numeric>
#include <variant>
#include <opencv2/imgproc.hpp>
#include <lvr2/util/Util.hpp>

namespace lvr2
{

const Vector3f DEBUG_ORIGIN(0, 0, 1);

template <typename BaseVecT>
RayCastingTexturizer<BaseVecT>::RayCastingTexturizer(
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
void RayCastingTexturizer<BaseVecT>::setGeometry(const BaseMesh<BaseVecT>& mesh)
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
void RayCastingTexturizer<BaseVecT>::setClusters(const ClusterBiMap<FaceHandle>& clusters)
{
    this->m_clusters = clusters;
}

template <typename BaseVecT>
void RayCastingTexturizer<BaseVecT>::setScanProject(const ScanProjectPtr project)
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
                // info.ImageToWorld = (position->transformation * camera->transformation).cast<float>();
                // info.WorldToImage = (camera->transformation.inverse() * position->transformation.inverse()).cast<float>();
                info.model         = camera->model;
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
                    info.image = imgOrGrp.template get<CameraImagePtr>();
                    info.ImageToWorld = (position->transformation * info.image->transformation * camera->transformation).template cast<float>();
                    info.WorldToImage = (camera->transformation.inverse() * info.image->transformation.inverse() * position->transformation.inverse()).template cast<float>();
                    
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
void RayCastingTexturizer<BaseVecT>::DEBUGDrawBorder(TextureHandle texH, const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect, ClusterHandle clusterH)
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

    // DEBUG
    // if (index != 10)
    // {
    //     return texH;
    // }
    // this->DEBUGDrawBorder(texH, boundingRect, clusterH);

    Texture& tex = this->m_textures[texH];

    size_t num_pixel = sizeX * sizeY;

    if (m_images.size() == 0)
    {
        std::cout << timestamp << "[RaycastingTexturizer] No images set, cannot texturize cluster" << std::endl;
        return texH;
    }

    // List of uv coordinates
    std::vector<TexCoords> uvCoords = this->calculateUVCoordsPerPixel(tex);
    // List of 3D points corresponding to uv coords
    std::vector<Vector3f> points = this->calculate3DPointsPerPixel(uvCoords, boundingRect);
    // List of booleans indicating if a texel is already Texturized
    std::vector<bool> texturized(num_pixel, false);
    int DEBUGimageIndex = 0;
    for (auto& imageInfo : m_images)
    {
        // TODO: Check cluster normal against view vector
        
        // The position of the camera
        Vector3f cameraOrigin;
        Vector3f tmp;
        matrixToPose<float>(imageInfo.ImageToWorld, cameraOrigin, tmp);
        // Cluster normal transformed to camera frame
        Vector3f clusterNormal = imageInfo.WorldToImage.template topLeftCorner<3, 3>().inverse().transpose() * Vector3f(
            boundingRect.m_normal.getX(),
            boundingRect.m_normal.getY(),
            boundingRect.m_normal.getZ()
        );
        {
            std::stringstream sstr;
            sstr << "Cluster" << DEBUGimageIndex++ << "ImageFrame.xyz";
            std::ofstream out;
            out.open(sstr.str());
            for (Vector3f point: points)
            {
                Vector3f proj = (imageInfo.WorldToImage * Vector4f(point.x(), point.y(), point.z(), 1)).template head<3>();
                out << proj.x() << " " << proj.y() << " " << proj.z() << "\n";
            }
            out << std::flush;
        }

        // Camera view vector
        Vector3f viewVec(0, 0, 1);
        // TODO: Maybe flip the condition if the normals are consistantly on the wrong side
        if (clusterNormal.dot(viewVec) < 0)
        {
            continue;
        }

        // A list of booleans indicating wether the point is visible
        std::vector<bool> visible = this->calculateVisibilityPerPixel(cameraOrigin, points, clusterH);
        
        cv::Mat camMat(3, 3, CV_64F);
        camMat.at<double>(0, 0) = imageInfo.model.fx;
        camMat.at<double>(0, 1) = 0;
        camMat.at<double>(0, 2) = imageInfo.model.cx;
        
        camMat.at<double>(1, 0) = 0;
        camMat.at<double>(1, 1) = imageInfo.model.fy;
        camMat.at<double>(1, 2) = imageInfo.model.cy;

        camMat.at<double>(2, 0) = 0;
        camMat.at<double>(2, 1) = 0;
        camMat.at<double>(2, 2) = 1;

        if(!imageInfo.image->loaded())
        {
            imageInfo.image->load();
        }

        for (size_t i = 0; i < num_pixel; i++)
        {
            if (!texturized[i] && visible[i])
            {
                Vector4f point = imageInfo.WorldToImage * Vector4f(points[i].x(), points[i].y(), points[i].z(), 1);
                // Skip if the point is behind the camera
                if (point[2] <= 0)
                {
                    continue;
                }

                cv::Mat in(3, 1, CV_32F);
                std::vector<cv::Point2f> out;

                in.at<float>(0) = point[0];
                in.at<float>(1) = point[1];
                in.at<float>(2) = point[2];

                cv::projectPoints(
                    in,
                    cv::Vec3f::zeros(),
                    cv::Vec3f::zeros(),
                    camMat,
                    cv::noArray(),
                    out
                );

                cv::Point2f uv = out[0];
                double u, v;
                u = uv.x;
                v = uv.y;
                undistorted_to_distorted_uv(u, v, imageInfo.model);
                uv.x = u;
                uv.y = v;
                // Skip out of bounds pixels
                if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
                {
                    continue;
                }

                // Calc pixel in image
                int x = std::floor((uv.x * (imageInfo.image->image.cols - 1)) - 0.5f);
                int y = std::floor((uv.y * (imageInfo.image->image.rows - 1)) - 0.5f);
                // Skip pixel outside the img coordinates
                if (x < 0 || y < 0 || x >= imageInfo.image->image.cols || y >= imageInfo.image->image.rows)
                {
                    continue;
                }

                cv::Vec3b red;
                red[0] = 0;
                red[1] = 0;
                red[2] = 255;

                // Retrieve the color
                cv::Vec3b color = imageInfo.image->image.template at<cv::Vec3b>(y, x);
                imageInfo.image->image.template at<cv::Vec3b>(y, x) = red;
                // Opencv uses BGR instead of RGB
                uint8_t r = color[0];
                uint8_t g = color[1];
                uint8_t b = color[2];
                setPixel(i, tex, r, g, b);
                texturized[i] = true;
            }                        
        }  
    }

    return texH;
}

template <typename BaseVecT>
template <typename... Args>
Texture RayCastingTexturizer<BaseVecT>::initTexture(Args&&... args) const
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
std::vector<TexCoords> RayCastingTexturizer<BaseVecT>::calculateUVCoordsPerPixel(const Texture& tex) const
{
    std::vector<TexCoords> ret;
    ret.reserve(tex.m_width * tex.m_height);

    for (size_t y = 0; y < tex.m_height; y++)
    {
        for (size_t x = 0; x < tex.m_width; x++)
        {
            float u = ((float) x + 0.5f) / (tex.m_width - 1);
            float v = ((float) y + 0.5f) / (tex.m_height - 1);
            ret.push_back(TexCoords(u, v));
        }
    }

    return std::move(ret);
}

template <typename BaseVecT>
std::vector<Vector3f> RayCastingTexturizer<BaseVecT>::calculate3DPointsPerPixel(
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
std::vector<bool> RayCastingTexturizer<BaseVecT>::calculateVisibilityPerPixel(
    const Vector3f from,
    const std::vector<Vector3f>& to,
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
            return (point - from).normalized();
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

} // namespace lvr2

