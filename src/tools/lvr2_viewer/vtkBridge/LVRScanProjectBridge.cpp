#include "LVRScanProjectBridge.hpp"

namespace lvr2
{


LVRScanProjectBridge::LVRScanProjectBridge(ScanProjectPtr project) : m_scanproject(project)
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
    m_scanproject = b.m_scanproject;
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

ScanProjectPtr LVRScanProjectBridge::getScanProject()
{
    return m_scanproject;
}

std::vector<ModelBridgePtr> LVRScanProjectBridge::getModels()
{
    return models; 
}

LVRScanProjectBridge::~LVRScanProjectBridge()
{
    // TODO Auto-generated destructor stub
}
} /* namespace lvr2 */
