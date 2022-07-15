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

LVRScanProjectItem::LVRScanProjectItem(
    ScanProjectBridgePtr bridge, 
    std::shared_ptr<FeatureBuild<scanio::ScanProjectIO>> io, 
    QString name) : QTreeWidgetItem(LVRScanProjectItemType), m_scanProjectBridge(bridge), m_name(name)
{
    m_io = io;
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
    m_io = item.m_io;
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
    //delete children of item
    m_scanProjectBridge = bridge;
    for(int i = 0; i < bridge->getScanProject()->positions.size(); i++)
    {
        delete child(0);
    }

    //create new child item for each position
    for(int i = 0; i < bridge->getScanProject()->positions.size(); i++)
    {
        std::stringstream pos;
        pos << "" << std::setfill('0') << std::setw(8) << i;
        std::string posName = pos.str();
        LVRScanPositionItem* scanPosItem = new LVRScanPositionItem(bridge->getScanPositions()[i], QString::fromStdString(posName));
        addChild(scanPosItem);
    }
}

std::shared_ptr<FeatureBuild<scanio::ScanProjectIO>> LVRScanProjectItem::getIO()
{
    return m_io;
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
