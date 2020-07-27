#include "LVRLabeledScanProjectEditMarkItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRTextureMeshItem.hpp"
#include "LVRItemTypes.hpp"
#include "LVRModelItem.hpp"
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>

namespace lvr2
{

LVRLabeledScanProjectEditMarkItem::LVRLabeledScanProjectEditMarkItem(LabeledScanProjectEditMarkBridgePtr bridge, QString name) :
    QTreeWidgetItem(LVRLabeledScanProjectEditMarkItemType), m_labelScanProjectEditMarkBridge(bridge), m_name(name)
{
    
    // Setup item properties
    setText(0, name);
}

LVRLabeledScanProjectEditMarkItem::LVRLabeledScanProjectEditMarkItem(const LVRLabeledScanProjectEditMarkItem& item)
{
    //m_scanProjectBridge   = item.m_scanProjectBridge;
    m_name          = item.m_name;
}


QString LVRLabeledScanProjectEditMarkItem::getName()
{
    return m_name;
}

void LVRLabeledScanProjectEditMarkItem::setName(QString name)
{
    m_name = name;
    setText(0, m_name);
}

LabeledScanProjectEditMarkBridgePtr LVRLabeledScanProjectEditMarkItem::getLabeledScanProjectEditMarkBridge()
{
	return m_labelScanProjectEditMarkBridge;
}

bool LVRLabeledScanProjectEditMarkItem::isEnabled()
{
    return this->checkState(0);
}

void LVRLabeledScanProjectEditMarkItem::setVisibility(bool visible)
{
}

LVRLabeledScanProjectEditMarkItem::~LVRLabeledScanProjectEditMarkItem()
{
    // TODO Auto-generated destructor stub
}

}
