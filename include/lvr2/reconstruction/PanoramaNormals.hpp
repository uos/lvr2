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

    PointBufferPtr computeNormals(int with, int height, bool interpolate);

private:
    ModelToImage*       m_mti;
    PointBufferPtr      m_buffer;
};

} // namespace lvr2

#endif // PANORAMANORMALS_H
