#include "LVRScanPositionBridge.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{


LVRScanPositionBridge::LVRScanPositionBridge(ScanPositionPtr position) : m_scanposition(position)
{

    for(auto scan : position->scans)
    {
        ModelPtr model(new Model);
        model->m_pointCloud = scan->points;
        ModelBridgePtr modelBridge(new LVRModelBridge(model));
        models.push_back(modelBridge);
    }        
    Eigen::Vector3d pos;
    Eigen::Vector3d angles;
    matrixToPose(position->registration, pos, angles);
    m_pose.x = pos[0];
    m_pose.y = pos[1];
    m_pose.z = pos[2];
    m_pose.r = angles[0];
    m_pose.t = angles[1];
    m_pose.p = angles[2];
}

LVRScanPositionBridge::LVRScanPositionBridge(const LVRScanPositionBridge& b)
{
    m_scanposition = b.m_scanposition;
    models = b.models;
    m_pose = b.m_pose;
}

void LVRScanPositionBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    for(auto model : models)
    {
        if(model->validPointBridge())
        {
            renderer->AddActor(model->getPointBridge()->getPointCloudActor());
        }
    }
}

void LVRScanPositionBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    for(auto model : models)
    {
        if(model->validPointBridge()) renderer->RemoveActor(model->getPointBridge()->getPointCloudActor());
    }
}

void LVRScanPositionBridge::setVisibility(bool visible)
{
    for(auto model : models)
    {
        model->setVisibility(visible);
    }
}


Pose LVRScanPositionBridge::getPose()
{
    return m_pose;
}
ScanPositionPtr LVRScanPositionBridge::getScanPosition()
{
    return m_scanposition;
}

std::vector<ModelBridgePtr> LVRScanPositionBridge::getModels()
{
    return models; 
}

LVRScanPositionBridge::~LVRScanPositionBridge()
{
    // TODO Auto-generated destructor stub
}
} /* namespace lvr2 */
