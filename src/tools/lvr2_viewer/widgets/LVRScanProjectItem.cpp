#include "LVRScanProjectItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRTextureMeshItem.hpp"
#include "LVRItemTypes.hpp"
#include "LVRScanPositionItem.hpp"
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <sstream>

namespace lvr2
{

LVRScanProjectItem::LVRScanProjectItem(ScanProjectBridgePtr bridge, QString name) :
    QTreeWidgetItem(LVRScanProjectItemType), m_scanProjectBridge(bridge), m_name(name)
{
    for(int i = 0; i < bridge->getScanProject()->positions.size(); i++)
    {
        std::stringstream pos;
        pos << "" << std::setfill('0') << std::setw(8) << i;
        std::string posName = pos.str();
        LVRScanPositionItem* scanPosItem = new LVRScanPositionItem(bridge->getScanPositions()[i], QString::fromStdString(posName));
        addChild(scanPosItem);

    }
    
    // Setup item properties
    setText(0, name);

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

void LVRScanProjectItem::setBridge(ScanProjectBridgePtr bridge)
{
    m_scanProjectBridge = bridge;
    for(int i = 0; i < bridge->getScanProject()->positions.size(); i++)
    {
        delete child(0);
    }
    for(int i = 0; i < bridge->getScanProject()->positions.size(); i++)
    {
        std::stringstream pos;
        pos << "" << std::setfill('0') << std::setw(8) << i;
        std::string posName = pos.str();
        LVRScanPositionItem* scanPosItem = new LVRScanPositionItem(bridge->getScanPositions()[i], QString::fromStdString(posName));
        addChild(scanPosItem);

    }
}

void LVRScanProjectItem::setVisibility(bool visible)
{
    for (auto position : m_scanProjectBridge->getScanPositions())
    {
        position->setVisibility(visible);
    }
}

LVRScanProjectItem::~LVRScanProjectItem()
{
    // TODO Auto-generated destructor stub
}

}
