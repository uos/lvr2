#include "LVRScanProjectBridge.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{


LVRScanProjectBridge::LVRScanProjectBridge(ScanProjectPtr project) : m_scanproject(project)
{

    for (auto position : project->positions)
    {

        ScanPositionBridgePtr posBridgePtr;
        posBridgePtr = ScanPositionBridgePtr(new LVRScanPositionBridge(position));
        m_scanPositions.push_back(posBridgePtr);
    }
}

LVRScanProjectBridge::LVRScanProjectBridge(const LVRScanProjectBridge& b)
{
    m_scanproject = b.m_scanproject;
    m_scanPositions = m_scanPositions;
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

    ScanPositionBridgePtr posBridgePtr;
    posBridgePtr = ScanPositionBridgePtr(new LVRScanPositionBridge(posPtr));
    m_scanPositions.push_back(posBridgePtr);

    //TODO What about the meshes?
}
void LVRScanProjectBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    for(auto position : m_scanPositions)
    {
        position->addActors(renderer);
    }
}

void LVRScanProjectBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    for(auto position : m_scanPositions)
    {
        position->removeActors(renderer);
    }
}

ScanProjectPtr LVRScanProjectBridge::getScanProject()
{
    return m_scanproject;
}

std::vector<ScanPositionBridgePtr> LVRScanProjectBridge::getScanPositions()
{
    return m_scanPositions; 
}

LVRScanProjectBridge::~LVRScanProjectBridge()
{
    // TODO Auto-generated destructor stub
}
} /* namespace lvr2 */
