#ifndef SCANDATA_HPP_
#define SCANDATA_HPP_

#include "PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/types/MatrixTypes.hpp"

#include <string>
#include <iostream>

#include <Eigen/Dense>

namespace lvr2
{

struct ScanData
{
    ScanData() :
        m_points(nullptr),
        m_hFieldOfView(0),
        m_vFieldOfView(0),
        m_hResolution(0),
        m_vResolution(0),
        m_pointsLoaded(false),
        m_positionNumber(-1),
        m_scanDataRoot("") {}

	~ScanData() {};

    PointBufferPtr                          m_points;
    Transformd                              m_registration;
    Transformd                              m_poseEstimation;
    BoundingBox<BaseVector<float> >         m_boundingBox;

    float                                   m_hFieldOfView;
    float                                   m_vFieldOfView;
    float                                   m_hResolution;
    float                                   m_vResolution;

    bool                                    m_pointsLoaded;
    int                                     m_positionNumber;
    std::string                             m_scanDataRoot;
};


void parseSLAMDirectory(std::string dir, vector<ScanData>& scans);


} // namespace lvr2
#endif /* !SCANDATA_HPP_ */
