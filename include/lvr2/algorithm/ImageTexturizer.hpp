#pragma once

#include <lvr2/algorithm/Texturizer.hpp>
#include <lvr2/geometry/Normal.hpp>

#include <lvr2/io/ScanprojectIO.hpp>
#include <lvr2/geometry/Matrix4.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace lvr2
{

template<typename BaseVecT>
struct ImageData {
    cv::Mat data;
    Vector<BaseVecT>  pos;
    Vector<BaseVecT>  dir;
    Matrix4<BaseVecT> project_to_image_transform;
    float distortion_params[6];
    float intrinsic_params[4];
};

template<typename BaseVecT>
class ImageTexturizer : public Texturizer<BaseVecT> {

public:

    ImageTexturizer(
        float texelSize,
        int minClusterSize,
        int maxClusterSize
    ) : Texturizer<BaseVecT>(texelSize, minClusterSize, maxClusterSize)
    {
        image_data_initialized = false;
    }

    void set_project(Scanproject& project)
    {
        this->project = project;
    }

    TextureHandle generateTexture(
        int index,
        const PointsetSurface<BaseVecT>& surface,
        const BoundingRectangle<BaseVecT>& boundingRect
    );

private:
    Scanproject project;

    bool image_data_initialized;
    std::vector<ImageData<BaseVecT> > images;

    void init_image_data();

    template<typename ValueType>
    void undistorted_to_distorted_uv(ValueType &u, ValueType &v, const ImageData<BaseVecT> &img);

    bool exclude_image(Vector<BaseVecT> pos, const ImageData<BaseVecT> &image_data);

    bool point_behind_camera(Vector<BaseVecT> pos, const ImageData<BaseVecT> &image_data);
};

} // namespace lvr2

#include "ImageTexturizer.tcc"
