#include "LVRLabeledScanProjectEditMarkBridge.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{


LVRLabeledScanProjectEditMarkBridge::LVRLabeledScanProjectEditMarkBridge(LabeledScanProjectEditMarkPtr project) : m_labeledScanProjectEditMark(project)
{
    if (project->editMarkProject && project->editMarkProject->project)
    {
        ScanProjectBridgePtr scanProjectBridge = ScanProjectBridgePtr(new LVRScanProjectBridge(project->editMarkProject->project));
        
        m_scanProjectBridgePtr = scanProjectBridge;
    }
    if (project->labelRoot)
    {
        LabelBridgePtr labelBridge = LabelBridgePtr(new LVRLabelBridge(project->labelRoot));
        m_labelBridgePtr = labelBridge;
    }
    else {
    }
}

LVRLabeledScanProjectEditMarkBridge::LVRLabeledScanProjectEditMarkBridge(const LVRLabeledScanProjectEditMarkBridge& b)
{
    m_labeledScanProjectEditMark = b.m_labeledScanProjectEditMark;
    m_labelBridgePtr = b.m_labelBridgePtr;
    m_scanProjectBridgePtr = b.m_scanProjectBridgePtr;
}

LVRLabeledScanProjectEditMarkBridge::LVRLabeledScanProjectEditMarkBridge(ModelBridgePtr modelBridge)
{
    LabeledScanProjectEditMarkPtr labeledScanProject(new LabeledScanProjectEditMark);
    ScanProjectEditMarkPtr scanProjectEditMark(new ScanProjectEditMark);
    ScanProjectBridgePtr scanProjectBridge(new LVRScanProjectBridge(modelBridge));
    scanProjectEditMark->project = scanProjectBridge->getScanProject();
    labeledScanProject->editMarkProject = scanProjectEditMark;


    m_labeledScanProjectEditMark = labeledScanProject;
    m_scanProjectBridgePtr = scanProjectBridge;

}

LVRLabeledScanProjectEditMarkBridge::LVRLabeledScanProjectEditMarkBridge(ScanProjectBridgePtr scanProjectBridge)
{
    LabeledScanProjectEditMarkPtr labeledScanProject(new LabeledScanProjectEditMark);
    ScanProjectEditMarkPtr scanProjectEditMark(new ScanProjectEditMark);
    scanProjectEditMark->project = scanProjectBridge->getScanProject();
    labeledScanProject->editMarkProject = scanProjectEditMark;

    m_labeledScanProjectEditMark = labeledScanProject;
    m_scanProjectBridgePtr = scanProjectBridge;
}

void LVRLabeledScanProjectEditMarkBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    if (m_labelBridgePtr)
    {
        m_labelBridgePtr->addActors(renderer);
    }
    if (m_scanProjectBridgePtr)
    {
        m_scanProjectBridgePtr->addActors(renderer);
    }
}

void LVRLabeledScanProjectEditMarkBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    if (m_labelBridgePtr)
    {
        m_labelBridgePtr->removeActors(renderer);
    }
    if (m_scanProjectBridgePtr)
    {
        m_scanProjectBridgePtr->removeActors(renderer);
    }
}

LabeledScanProjectEditMarkPtr LVRLabeledScanProjectEditMarkBridge::getLabeledScanProjectEditMark()
{
    return m_labeledScanProjectEditMark;
}


LVRLabeledScanProjectEditMarkBridge::~LVRLabeledScanProjectEditMarkBridge()
{
    // TODO Auto-generated destructor stub
}
}
