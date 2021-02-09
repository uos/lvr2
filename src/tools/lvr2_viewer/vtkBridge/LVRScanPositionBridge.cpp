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
    //calculate rotation for the models according to the scanposition pose
    m_pose.x = pos[0];
    m_pose.y = pos[1];
    m_pose.z = pos[2];
    m_pose.r = -angles[0] * 180 / PI;
    m_pose.t = -angles[1] * 180 / PI;
    m_pose.p = -angles[2] * 180 / PI;
    
    //add all the scans in the scanposition to the models
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


void LVRScanPositionBridge::showScannerPosition(vtkSmartPointer<vtkRenderer> renderer, int scaleFactor)
{
    if(m_cylinderActor != nullptr)
    {
        return;
    }
    vtkSmartPointer<vtkCylinderSource> cylinderSource = vtkSmartPointer<vtkCylinderSource>::New();
    //cylinderSource->SetCenter(m_pose.x, m_pose.y, m_pose.z+0.15);

    //calculate size of cylinder from scale of the project (30 cm high, 15 cm diameter)
    double radius = 75 / static_cast<double>(scaleFactor);
    double height = 300 / static_cast<double>(scaleFactor);

    //create cylinder and orient it according to the pose
    cylinderSource->SetRadius(radius);
    cylinderSource->SetHeight(height);
    cylinderSource->SetResolution(100);
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PostMultiply();
    transform->RotateX(m_pose.r + 90.0);
    transform->RotateY(m_pose.t);
    transform->RotateZ(m_pose.p);
    transform->Translate(m_pose.x, m_pose.y, m_pose.z + (height / 2));

    // Transform the polydata
    vtkSmartPointer<vtkTransformPolyDataFilter> transformPD = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformPD->SetTransform(transform);
    transformPD->SetInputConnection(cylinderSource->GetOutputPort());


    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(transformPD->GetOutputPort());
    m_cylinderActor = vtkSmartPointer<vtkActor>::New();
    m_cylinderActor->SetMapper(mapper);
    m_cylinderActor->GetProperty()->SetColor(250.0/255.0, 128.0/255.0, 114.0/255.0);
    
    renderer->AddActor(m_cylinderActor);


    //calculate coordinates and vectors for position and orientation of the scanner
    double startD[] = {0, 0, 0, 1};
    double endD[] = {0, 1, 0, 1};
    double endSideD[] = {1, 0, 0, 1};
    double endUpD[] = {0, 0, -1, 1};
    
    transform->MultiplyPoint(startD, startD);
    transform->MultiplyPoint(endD, endD);
    transform->MultiplyPoint(endSideD, endSideD);
    transform->MultiplyPoint(endUpD, endUpD);

    Vec start(startD[0], startD[1], startD[2]);
    Vec end(endD[0], endD[1], endD[2]);
    Vec endSide(endSideD[0], endSideD[1], endSideD[2]);
    Vec endUp(endUpD[0], endUpD[1], endUpD[2]);

    //create arrows for the orientation
    LVRVtkArrow* arrowSide = new LVRVtkArrow(start, endSide);
    arrowSide->setTmpColor(1, 0, 0);


    LVRVtkArrow* arrowUp = new LVRVtkArrow(start, endUp);
    arrowUp->setTmpColor(0, 1, 0);


    LVRVtkArrow* arrow = new LVRVtkArrow(start, end);
    arrow->setTmpColor(0, 0, 1);

    //add arrows to the renderer
    renderer->AddActor(arrow->getArrowActor());
    renderer->AddActor(arrowUp->getArrowActor());
    renderer->AddActor(arrowSide->getArrowActor());

    //store arrows in the member variable
    m_arrows.push_back(arrow);
    m_arrows.push_back(arrowUp);
    m_arrows.push_back(arrowSide);    

    m_scannerPositionIsVisible = true;
}

void LVRScanPositionBridge::hideScannerPosition(vtkSmartPointer<vtkRenderer> renderer)
{
    //remove scanner and arrow actors from the renderer and delete the actors 
    renderer->RemoveActor(m_cylinderActor);
    m_cylinderActor = nullptr;
    m_scannerPositionIsVisible = false;
    for (auto arrow : m_arrows)
    {
        renderer->RemoveActor(arrow->getArrowActor());
        renderer->RemoveActor(arrow->getStartActor());
        renderer->RemoveActor(arrow->getEndActor());
        delete arrow;
    }
    m_arrows.clear();
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
