#include "LVRLabeledScanProjectEditMarkBridge.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{


LVRLabeledScanProjectEditMarkBridge::LVRLabeledScanProjectEditMarkBridge(LabeledScanProjectEditMarkPtr project) : m_labeledScanProjectEditMark(project)
{
    if(project->editMarkProject && project->editMarkProject->project)
    {
        ScanProjectBridgePtr scanProjectBridge = ScanProjectBridgePtr(new LVRScanProjectBridge(project->editMarkProject->project));
        
        m_scanProjectBridgePtr = scanProjectBridge;
    }

}

LVRLabeledScanProjectEditMarkBridge::LVRLabeledScanProjectEditMarkBridge(const LVRLabeledScanProjectEditMarkBridge& b)
{
}

LVRLabeledScanProjectEditMarkBridge::LVRLabeledScanProjectEditMarkBridge(ModelBridgePtr modelBridge)
{
}
void LVRLabeledScanProjectEditMarkBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
}

void LVRLabeledScanProjectEditMarkBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
}

LabeledScanProjectEditMarkPtr LVRLabeledScanProjectEditMarkBridge::getLabelScanProjectEditMark()
{
    return m_labeledScanProjectEditMark;
}


LVRLabeledScanProjectEditMarkBridge::~LVRLabeledScanProjectEditMarkBridge()
{
    // TODO Auto-generated destructor stub
}
}
