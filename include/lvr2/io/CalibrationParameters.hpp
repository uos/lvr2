#ifndef CALIBRATIONPARAMETERS_HPP
#define CALIBRATIONPARAMETERS_HPP

namespace lvr2
{

typedef struct HyperspectralCalibration_
{
    HyperspectralCalibration_() :
        a0(0.0f), a1(0.0f), a2(0.0f),
        angle_x(0.0f), angle_y(0.0f), angle_z(0.0f),
        origin_x(0.0f), origin_y(0.0f), origin_z(0.0f),
        principal_x(0.0f), principal_y(0.0f) {}

    // 1st degree (linear) vertical distortion (aka. aspect ratio correction)
    float a0;

    // 2nd degree vertical distortion
    float a1;

    // 4th degree vertical distortion
    float a2;

    // Rotation around x axis
    float angle_x;

    // Rotation around y axis
    float angle_y;

    // Rotation around z axis
    float angle_z;

    // Translation from camera origin in x direction
    float origin_x;

    // Translation from camera origin in y direction
    float origin_y;

    // Translation from camera origin in z direction
    float origin_z;

    // Vertical offset of the camera image center
    float principal_y;

    // Horizontal offset of the camera image senter
    float principal_x;
} HyperspectralCalibration;

} // namespace lvr2

#endif // CALIBRATIONPARAMETERS_HPP
