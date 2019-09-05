#ifndef SCAN_HPP_
#define SCAN_HPP_

#include "MatrixTypes.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

#include <string>
#include <iostream>

#include <Eigen/Dense>

namespace lvr2
{

struct Scan
{
    Scan() :
        m_points(nullptr),
        m_registration(Transformd::Identity()),
        m_poseEstimation(Transformd::Identity()),
        m_hFieldOfView(0),
        m_vFieldOfView(0),
        m_hResolution(0),
        m_vResolution(0),
        m_pointsLoaded(false),
        m_positionNumber(0),
        m_scanRoot("")
    {}

    ~Scan() {};

    PointBufferPtr                  m_points;
    Transformd                      m_registration;
    Transformd                      m_poseEstimation;
    BoundingBox<BaseVector<float> > m_boundingBox;

    float                           m_hFieldOfView;
    float                           m_vFieldOfView;
    float                           m_hResolution;
    float                           m_vResolution;

    bool                            m_pointsLoaded;
    int                             m_positionNumber;
    std::string                     m_scanRoot;
};

using ScanPtr = std::shared_ptr<Scan>;

void parseSLAMDirectory(std::string dir, std::vector<ScanPtr>& scans);

} // namespace lvr2
#endif /* !SCAN_HPP_ */
