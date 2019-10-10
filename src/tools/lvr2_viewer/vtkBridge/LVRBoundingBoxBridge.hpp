#ifndef LVR2_TOOLS_VIEWER_VTKBRIDGE_LVRBOUNDINGBOXBRIDGE_HPP
#define LVR2_TOOLS_VIEWER_VTKBRIDGE_LVRBOUNDINGBOXBRIDGE_HPP

#include <boost/shared_ptr.hpp>

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"


namespace lvr2
{

class Pose;

class LVRBoundingBoxBridge
{
    using Vec = BaseVector<float>;

    public:
        LVRBoundingBoxBridge(BoundingBox<Vec> bb);
        BoundingBox<Vec> getBoundingBox() { return m_boundingBox; }
        vtkSmartPointer<vtkActor> getActor() { return m_actor; }
        void setPose(const Pose& pose);
        void setVisibility(bool visible);
        void setColor(double r, double g, double b);

    private:
        BoundingBox<Vec>            m_boundingBox;
        vtkSmartPointer<vtkActor>   m_actor;

};

using BoundingBoxBridgePtr = boost::shared_ptr<LVRBoundingBoxBridge>;

} // namespace lvr2

#endif
