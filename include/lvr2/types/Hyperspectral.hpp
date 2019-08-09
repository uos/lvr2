#ifndef __HYPERSPECTRAL_HPP__
#define __HYPERSPECTRAL_HPP__

#include <vector>

#include <opencv2/core.hpp>

/**
 * @brief   Struct to hold hyperspectral a hyperspectral panorama
 *          cube and corresponding model parameters to align it 
 *          with a laser scan
 */
struct HyperspectralPanorama
{
    /// Distortion
    float   distortion;

    /// Origin x
    float   ox;

    /// Origin y
    float   oy;
    
    /// Origin z
    float   oz;

    /// Principal 1
    float   p1;

    /// Principal 2
    float   p2;

    /// Horizontal field of view
    float   fovh;

    /// Vertical field of view
    float   fovv;

    /// Min wavelength in nm, i.e., wavelength of the image 
    /// in the first channel
    float   wmin;

    /// Maximum wavelength, i.e., wavelangth of the image in 
    /// the last channel
    float   wmax;

    /// Vector of intensity (greyscale) images, one for each
    /// channel
    std::vector<cv::Mat> channels;

    HyperspectralPanorama() :
        distortion(0.0f), ox(0.0f), oy(0.0f), oz(0.0f),
        p1(0.0f), p2(0.0f), fovh((0.0f)), fovv(0.0f),
        wmin(0.0f), wmax(0.0f) {}
}

#endif