#ifndef VLR2_ALGORITHM_SPECTRALTEXTURIZER_HPP
#define VLR2_ALGORITHM_SPECTRALTEXTURIZER_HPP

#include "lvr2/algorithm/Texturizer.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{
/**
 * @brief A texturizer that uses h5 spectral panoramas instead of pointcloud colors for creating textures
 * for meshes. 
 */
template<typename BaseVecT>
class SpectralTexturizer : public Texturizer<BaseVecT> 
{
public:
    /**
     * @brief constructor
     */
    SpectralTexturizer(
        float texelSize,
        int minClusterSize,
        int maxClusterSize
    ) : Texturizer<BaseVecT>(texelSize, minClusterSize, maxClusterSize)
    {
        image_data_initialized = false;
    }

    virtual TextureHandle generateTexture(
        int index,
        const PointsetSurface<BaseVecT>& surface,
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect   
    ) override;

    /**
     * @brief set the image data to a specific panorama image of a given scanPosition
     * 
     * @param panoChannel Panorama channel to use
     */
    void init_image_data(HyperspectralPanoramaChannelPtr panoChannel);


private:

    bool image_data_initialized;
    cv::Mat spectralPanorama;
    // camera data
    Vector2d principal_point;
    Vector2d focal_length;
    Vector2d camera_fov;
    float distortions[3];

    Vector2d point_to_panorama_coord(Vector3d point, Vector2d principal_point, Vector2d focal_length, float distortion[]);
    void prepare_camera_data();
};
}


#include "SpectralTexturizer.tcc"

#endif