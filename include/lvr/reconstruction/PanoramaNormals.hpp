#ifndef PANORAMANORMALS_H
#define PANORAMANORMALS_H

#include <lvr/reconstruction/ModelToImage.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/Progress.hpp>

namespace lvr
{

////
/// \brief  The PanoramaNormals class computes normals for a given panorama
///
class PanoramaNormals
{
public:

    PanoramaNormals(ModelToImage* mti);

    PointBufferPtr computeNormals(int with, int height, bool interpolate);

private:
    ModelToImage*       m_mti;
    PointBufferPtr      m_buffer;
};

} // namespace lvr

#endif // PANORAMANORMALS_H