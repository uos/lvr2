#include "lvr2/algorithm/fapm/PixelProjector.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2 {

PixelProjector::PixelProjector(
    RaycasterBasePtr raycaster,
    ScanCameraPtr scan_cam)
: m_raycaster(raycaster)
, m_scan_cam(scan_cam)
{
    std::cout << "init Feature Projector" << std::endl; 
}

Extrinsicsd rigidInverse(const Extrinsicsd& T)  {
    Extrinsicsd Tinv = Extrinsicsd::Identity();
    Tinv.block<3,3>(0,0) = T.block<3,3>(0,0).transpose();
    Tinv.block<3,1>(0,3) =-(Tinv.block<3,3>(0,0) * T.block<3,1>(0,3));
    return Tinv;
}

// void extrinsicsToRay(const Extrinsicsd& T, 
//     Vector3f& ray_origin, Vector3f& ray_dir)
// {   
//     ray_origin = T.block<3, 1>(0, 3).cast<float>();
//     ray_dir = T.block<3,3>(0,0).cast<float>() * Vector3f(1.0, 0.0, 0.0);
// }

void PixelProjector::project(
    const ScanImagePtr& image,
    const cv::Point2f& pixel,
    uint8_t& hit,
    Vector3f& intersection) const
{
    // std::cout << "Project pixel" << std::endl;

    // 1. Generate Ray
    
    // 2. Raycast Ray
    Vector3f ray_origin(0.0, 0.0, 0.0);
    Vector3f ray_dir(1.0, 0.0, 0.0);


    //////
    // pixel space
    // ---------------- x
    // |
    // |
    // |
    // |
    // y

    //////
    // cam space raycasting
    //                  z
    //                  |
    //                  |
    //                  |
    //                  |
    // y ----------------
    

    // std::cout << "Cam specs: " << std::endl;
    // std::cout << this->m_scan_cam->camera.fx << " " << this->m_scan_cam->camera.fy << std::endl;
    // std::cout << this->m_scan_cam->camera.cx << " " << this->m_scan_cam->camera.cy << std::endl;


    ray_dir.y() = -( (pixel.x - this->m_scan_cam->camera.cx) 
                / this->m_scan_cam->camera.fx);

    ray_dir.z() = -( (pixel.y - this->m_scan_cam->camera.cy) 
                / this->m_scan_cam->camera.fy);

    // Extrinsicsd Tinv = rigidInverse(image->extrinsics);
    ray_origin = image->extrinsics.block<3,1>(0,3).cast<float>();
    ray_dir = image->extrinsics.block<3,3>(0,0).cast<float>() * ray_dir;
    hit = m_raycaster->castRay(ray_origin, ray_dir, intersection);
}


void PixelProjector::project(
    const ScanImagePtr& image,
    const std::vector<cv::Point2f>& pixels,
    std::vector<uint8_t>& hits,
    std::vector<Vector3f>& intersections) const
{
    std::cout << "Project pixels" << std::endl;
}

void PixelProjector::project(
    const ScanImagePtr& image,
    std::vector<std::vector<uint8_t> >& hits,
    std::vector<std::vector<Vector3f> >& intersections) const
{
    // init memory
    hits.resize(m_scan_cam->camera.width);
    intersections.resize(m_scan_cam->camera.width);

    for(size_t i=0; i<m_scan_cam->camera.width; i++)
    {
        hits[i].resize(m_scan_cam->camera.height);
        intersections[i].resize(m_scan_cam->camera.height);
        for(size_t j=0; j<m_scan_cam->camera.height; j++)
        {
            cv::Point2f pixel(
                static_cast<float>(i) + 0.5, 
                static_cast<float>(j) + 0.5
            );
            project(image, pixel, hits[i][j], intersections[i][j]);
        }
    }
}

} // namespace lvr2