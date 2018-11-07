#ifndef SCANDATA_HPP_
#define SCANDATA_HPP_

#include "PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/geometry/Matrix4.hpp"

namespace lvr2
{

struct ScanData 
{
	ScanData() :
        m_points(nullptr),
        m_hFieldOfView(0),
        m_vFieldOfView(0),
        m_hResolution(0),
        m_vResolution(0) {}

	~ScanData() {};

    PointBufferPtr                      m_points;
    Matrix4<BaseVector<float> >         m_registration;
    Matrix4<BaseVector<float> >         m_poseEstimation;
    BoundingBox<BaseVector<float> >     m_boundingBox;

    float                               m_hFieldOfView;
    float                               m_vFieldOfView;
    float                               m_hResolution;
    float                               m_vResolution;
};

} // namespace lvr2
#endif /* !SCANDATA_HPP_ */
