#include "LVRLabelBridge.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{


LVRLabelBridge::LVRLabelBridge(LabelRootPtr project) :
    m_labelRoot(project)
{
    if(project->points)
    {
        m_pointBridge = PointBufferBridgePtr(new LVRPointBufferBridge(project->points));
    }
}

LVRLabelBridge::LVRLabelBridge(const LVRLabelBridge& b)
{
   m_labelRoot = b.m_labelRoot;
   m_pointBridge = b.m_pointBridge;
}

bool LVRLabelBridge::validPointBridge()
{
    if(!m_pointBridge)
        return false;

    return (m_pointBridge->getNumPoints() > 0) ? true : false;
}

void LVRLabelBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    if(validPointBridge())
    {
    	renderer->AddActor(m_pointBridge->getPointCloudActor());
    }
}

void LVRLabelBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    if(validPointBridge()) renderer->RemoveActor(m_pointBridge->getPointCloudActor());
}
void LVRLabelBridge::setVisibility(bool visible)
{
    if(validPointBridge())
    {
        m_pointBridge->getPointCloudActor()->SetVisibility(visible);
    }
}
LabelRootPtr LVRLabelBridge::getLabelRoot()
{
    return m_labelRoot;
}


LVRLabelBridge::~LVRLabelBridge()
{
    // TODO Auto-generated destructor stub
}
}
