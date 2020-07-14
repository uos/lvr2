//#include "LVRScanProjectItem.hpp"
#include "LVRScanProjectBridge.hpp"
//#include "LVRPointBufferBridge.hpp"

//#include "lvr2/geometry/Matrix4.hpp"

//#include <vtkTransform.h>
//#include <vtkActor.h>
//#include <vtkProperty.h>

namespace lvr2
{


LVRScanProjectBridge::LVRScanProjectBridge(ScanProjectPtr project)
{
    for (auto position : project->positions)
    {
        for(auto scan : position->scans)
        {
            ModelPtr model(new Model);
            model->m_pointCloud = scan->points;
            ModelBridgePtr modelBridge(new LVRModelBridge(model));
            //modelBridge->setTransform(poseEsimation);
               //TODO set pose?
            models.push_back(modelBridge);
        }        
    }
}

LVRScanProjectBridge::LVRScanProjectBridge(const LVRScanProjectBridge& b)
{
    models = b.models;
}

void LVRScanProjectBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    for(auto model : models)
    {
        if(model->validPointBridge())
        {
            renderer->AddActor(model->getPointBridge()->getPointCloudActor());
        }
    }
}

void LVRScanProjectBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    for(auto model : models)
    {
        if(model->validPointBridge()) renderer->RemoveActor(model->getPointBridge()->getPointCloudActor());
    }
}

LVRScanProjectBridge::~LVRScanProjectBridge()
{
    // TODO Auto-generated destructor stub
}
} /* namespace lvr2 */
