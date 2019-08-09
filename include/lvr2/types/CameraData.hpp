#ifndef CAMDATA_HPP_
#define CAMDATA_HPP_

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include <opencv2/core.hpp>

namespace lvr2
{

struct CamData
{
    Matrix4<BaseVector<float> >         m_intrinsics;
    Matrix4<BaseVector<float> >         m_extrinsics;
    cv::Mat                             m_image_data;
};


} // namespace lvr2
#endif /* !CAMDATA_HPP_ */
