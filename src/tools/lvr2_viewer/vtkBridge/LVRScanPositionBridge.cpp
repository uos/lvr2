#include "LVRScanPositionBridge.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{


LVRScanPositionBridge::LVRScanPositionBridge(ScanPositionPtr position) :
    m_scanposition(position),
    m_scannerPositionIsVisible(false)
{
    Eigen::Vector3d pos;
    Eigen::Vector3d angles;
    matrixToPose(m_scanposition->registration, pos, angles);
    m_pose.x = pos[0];
    m_pose.y = pos[1];
    m_pose.z = pos[2];
    m_pose.r = angles[0];
    m_pose.t = angles[1];
    m_pose.p = angles[2];
    for(auto scan : position->scans)
    {
            ModelPtr model(new Model);
            if (scan->points)
            {
            std::cout << "Unreduced Points: " << scan->points->numPoints() << std::endl;
            model->m_pointCloud = scan->points;
            }
            ModelBridgePtr modelBridge(new LVRModelBridge(model));
            modelBridge->setPose(m_pose);
            models.push_back(modelBridge);
    }        
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
        if(model->validPointBridge()){
            renderer->RemoveActor(model->getPointBridge()->getPointCloudActor());
        }
    }
}

void LVRScanPositionBridge::setModels(std::vector<ModelBridgePtr> newModels)
{
    models = newModels;
}

void LVRScanPositionBridge::setVisibility(bool visible)
{
    for(auto model : models)
    {
        model->setVisibility(visible);
    }
}


void LVRScanPositionBridge::showScannerPosition(vtkSmartPointer<vtkRenderer> renderer)
{
    if(m_cylinderActor != nullptr)
    {
        return;
    }
    vtkSmartPointer<vtkCylinderSource> cylinderSource = vtkSmartPointer<vtkCylinderSource>::New();
    //cylinderSource->SetCenter(m_pose.x, m_pose.y, m_pose.z+0.15);
    double startPoint[3], endpoint[3];
    startPoint[0] = m_pose.x;
    startPoint[1] = m_pose.y;
    startPoint[2] = m_pose.z;

    cylinderSource->SetRadius(0.15);
    cylinderSource->SetHeight(0.3);
    cylinderSource->SetResolution(100);
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Translate(startPoint);   // translate to starting point
    transform->RotateX(-90.0);          // align cylinder to x axis

    // Transform the polydata
    vtkSmartPointer<vtkTransformPolyDataFilter> transformPD = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformPD->SetTransform(transform);
    transformPD->SetInputConnection(cylinderSource->GetOutputPort());

    // Create a mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(transformPD->GetOutputPort());
    m_cylinderActor = vtkSmartPointer<vtkActor>::New();
    m_cylinderActor->SetMapper(mapper);
    m_cylinderActor->GetProperty()->SetColor(250.0/255.0, 128.0/255.0, 114.0/255.0);
    
    
    renderer->AddActor(m_cylinderActor);

    m_scannerPositionIsVisible = true;
}

void LVRScanPositionBridge::hideScannerPosition(vtkSmartPointer<vtkRenderer> renderer)
{
    renderer->RemoveActor(m_cylinderActor);
    m_cylinderActor = nullptr;
    m_scannerPositionIsVisible = false;
}
bool LVRScanPositionBridge::scannerPositionIsVisible()
{
    return m_scannerPositionIsVisible;
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
