#include "LVRScanProjectItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRTextureMeshItem.hpp"
#include "LVRItemTypes.hpp"
#include "LVRModelItem.hpp"
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>

namespace lvr2
{

LVRScanProjectItem::LVRScanProjectItem(ScanProjectBridgePtr bridge, QString name) :
    QTreeWidgetItem(LVRScanProjectItemType), m_scanProjectBridge(bridge), m_name(name)
{
    for(auto model : m_scanProjectBridge->models)
    {
        LVRModelItem* modelItem = new LVRModelItem(model, "Scan");
        addChild(modelItem);
    }
    
    // Setup item properties
    setText(0, name);
    //setCheckState(0, Qt::Checked);


    /*
    // Setup tree widget icon
    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_model_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);

    // Setup item properties
    setText(0, m_name);
    setCheckState(0, Qt::Checked);

    // Insert sub items
    if(bridge->m_pointBridge->getNumPoints())
    {
        LVRPointCloudItem* pointItem = new LVRPointCloudItem(bridge->m_pointBridge, this);
        addChild(pointItem);
        pointItem->setExpanded(true);
    }

    // Setup Pose
    //m_poseItem = new LVRPoseItem(bridge, this);
    //addChild(m_poseItem);
*/
}

LVRScanProjectItem::LVRScanProjectItem(const LVRScanProjectItem& item)
{
    m_scanProjectBridge   = item.m_scanProjectBridge;
    m_name          = item.m_name;
}


QString LVRScanProjectItem::getName()
{
    return m_name;
}

void LVRScanProjectItem::setName(QString name)
{
    m_name = name;
    setText(0, m_name);
}

ScanProjectBridgePtr LVRScanProjectItem::getScanProjectBridge()
{
	return m_scanProjectBridge;
}

bool LVRScanProjectItem::isEnabled()
{
    return this->checkState(0);
}

void LVRScanProjectItem::setVisibility(bool visible)
{
    for (auto model : m_scanProjectBridge->models)
    {
        model->setVisibility(visible);
    }
}

LVRScanProjectItem::~LVRScanProjectItem()
{
    // TODO Auto-generated destructor stub
}

}
