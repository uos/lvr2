#ifndef PANORAMANORMALS_H
#define PANORAMANORMALS_H

#include <lvr2/reconstruction/ModelToImage.hpp>
#include <lvr2/io/Model.hpp>
#include <lvr2/io/Progress.hpp>

namespace lvr2
{

////
/// \brief  The PanoramaNormals class computes normals for a given panorama
///
class PanoramaNormals
{
public:

    PanoramaNormals(ModelToImage* mti);

    PointBuffer2Ptr computeNormals(int with, int height, bool interpolate);

private:
    ModelToImage*       m_mti;
    PointBuffer2Ptr      m_buffer;
};

} // namespace lvr2

#include "PanoramaNormals.cpp"

#endif // PANORAMANORMALS_H
