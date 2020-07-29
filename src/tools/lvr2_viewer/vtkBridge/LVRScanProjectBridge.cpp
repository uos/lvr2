#include "LVRScanProjectBridge.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{


LVRScanProjectBridge::LVRScanProjectBridge(ScanProjectPtr project) : m_scanproject(project)
{

    std::cout << "creating scanproject Bridge" << std::endl;
    for (auto position : project->positions)
    {
    std::cout << "creating scanproject Bridge pos" << std::endl;
        for(auto scan : position->scans)
        {
    std::cout << "creating scanproject Bridge scan" << std::endl;
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

LVRScanProjectBridge::LVRScanProjectBridge(ModelBridgePtr modelBridge)
{
    //create new ScanProjectPtr with one ScanPosition which has one Scan
    ScanProjectPtr modelProject(new ScanProject);
    ScanPositionPtr posPtr(new ScanPosition);
    ScanPtr scanPtr(new Scan);
    posPtr->scans.push_back(scanPtr);
    modelProject->positions.push_back(posPtr);
    
    //set Pointcloud
    scanPtr->points = modelBridge->getPointBridge()->getPointBuffer();

    //set Pose
    Vector3<double> pos;
    pos[0] = modelBridge->getPose().x;
    pos[1] = modelBridge->getPose().y;
    pos[2] = modelBridge->getPose().z;

    Vector3<double> angle;
    angle[0] = modelBridge->getPose().r;
    angle[1] = modelBridge->getPose().t;
    angle[2] = modelBridge->getPose().p;

    posPtr->registration = poseToMatrix(pos, angle);
    m_scanproject = modelProject;

    //TODO What about the meshes?
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
