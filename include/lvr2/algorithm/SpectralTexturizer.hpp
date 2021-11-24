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
    /**
     * @brief Sets the internal ScanProjectPtr to the given one
     * 
     * @param scanPosition The UOS ScanPosition Pointer the internal project will be set to.
     */
    void set_scanPosition(ScanPositionPtr& scanPosition)
    {
        this->scanPosition = scanPosition;
    }




    virtual TextureHandle generateTexture(
        int index,
        const PointsetSurface<BaseVecT>& surface,
        const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect   
    ) override;

    /**
     * @brief set the image data to a specific panorama image of a given scanPosition
     * 
     * @param spectralIndex index to specify what spectral channel to use
     */
    void init_image_data(int spectralIndex);


private:
    // ScanProjectPtr project;
    ScanPositionPtr scanPosition;

    bool image_data_initialized;
    cv::Mat spectralPanorama;
    Vector2d point_to_panorama_coord(Vector3d point, Vector2d principal_point, Vector2d focal_length, float distortion[]);

};
}


#include "SpectralTexturizer.tcc"

#endif