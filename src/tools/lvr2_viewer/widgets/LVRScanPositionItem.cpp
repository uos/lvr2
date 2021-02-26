#include "LVRScanPositionItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRTextureMeshItem.hpp"
#include "LVRItemTypes.hpp"
#include "LVRModelItem.hpp"
#include "LVRScanCamItem.hpp"
#include "LVRScanImageItem.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>

#include <sstream>

#include <QTextStream>

namespace lvr2
{

LVRScanPositionItem::LVRScanPositionItem(ScanPositionBridgePtr bridge, QString name) :
    QTreeWidgetItem(LVRScanPositionItemType), 
    m_scanPositionBridge(bridge), 
    m_name(name)
{
    ScanPositionPtr pos = bridge->getScanPosition();

    //add child items for each scan
    for(auto lidar: pos->lidars)
    {
        for (int i = 0; i < lidar->scans.size(); i++)
        {
            std::stringstream pos;
            pos << "" << std::setfill('0') << std::setw(8) << i;
            std::string posName = pos.str();
            std::vector<ModelBridgePtr> models;
            models = bridge->getModels();
            LVRModelItem *modelItem = new LVRModelItem(models[i], QString::fromStdString(posName));
            // if (lidar->scans[i]->waveform)
            // {
            //     modelItem->getModelBridge()->setWaveform(bridge->getScanPosition()->scans[i]->waveform);
            // }
            addChild(modelItem);
        }
    }

    //add child items for each cam
    for(int i = 0; i < bridge->getScanPosition()->cameras.size(); i++)
    {
        ScanCamBridgePtr camBridge(new LVRScanCamBridge(bridge->getScanPosition()->cameras[0]));
        QString camName;
        QTextStream(&camName) << "camera_" << i;
        LVRScanCamItem* camItem = new LVRScanCamItem(camBridge, camName);
        addChild(camItem);
    }

    //add pose item as child
    LVRPoseItem* posItem = new LVRPoseItem(bridge->getPose());
    addChild(posItem);

    // Setup item properties
    setText(0, name);
    int nr = name.toInt();
    setData(0, Qt::UserRole, nr);
    
    setCheckState(0, Qt::Checked);
}

LVRScanPositionItem::LVRScanPositionItem(const LVRScanPositionItem& item)
{
    m_scanPositionBridge   = item.m_scanPositionBridge;
    m_name          = item.m_name;
}


QString LVRScanPositionItem::getName()
{
    return m_name;
}

void LVRScanPositionItem::setName(QString name)
{
    m_name = name;
    setText(0, m_name);
}

ScanPositionBridgePtr LVRScanPositionItem::getScanPositionBridge()
{
	return m_scanPositionBridge;
}

bool LVRScanPositionItem::isEnabled()
{
    return this->checkState(0);
}

void LVRScanPositionItem::setBridge(ScanPositionBridgePtr bridge)
{
    //delete children of this item
    // TODO: Does this make sense? Has bridge always the same
    // elemets as the current tree???
    m_scanPositionBridge = bridge;
    ScanPositionPtr pos = bridge->getScanPosition();
    for (auto lidar : pos->lidars)
    {
        for (int i = 0; i < lidar->scans.size(); i++)
        {
            delete child(0);
        }
    }

    for (int i = 0; i < bridge->getScanPosition()->cameras.size(); i++)
    {
        delete child(0);
    }
    delete child(0);

    //create new items for scans
    for (auto lidar : pos->lidars)
    {
        for (int i = 0; i < lidar->scans.size(); i++)
        {
            std::stringstream pos;
            pos << "" << std::setfill('0') << std::setw(8) << i;
            std::string posName = pos.str();
            std::vector<ModelBridgePtr> models;
            models = bridge->getModels();
            LVRModelItem *modelItem = new LVRModelItem(models[i], QString::fromStdString(posName));
            // if (bridge->getScanPosition()->scans[i]->waveform)
            // {
            //     modelItem->getModelBridge()->setWaveform(lidar->scans[i]->waveform);
            // }
            addChild(modelItem);
        }
    }

    //create new items for cams
    for(int i = 0; i < bridge->getScanPosition()->cameras.size(); i++)
    {
        ScanCamBridgePtr camBridge(new LVRScanCamBridge(bridge->getScanPosition()->cameras[i]));
        QString camName;
        QTextStream(&camName) << "camera_" << i;
        LVRScanCamItem* camItem = new LVRScanCamItem(camBridge, camName);
        addChild(camItem);
    }

    //create new pose item
    LVRPoseItem* posItem = new LVRPoseItem(bridge->getPose());
    addChild(posItem);
}

void LVRScanPositionItem::setVisibility(bool visible)
{
    for (auto model : m_scanPositionBridge->getModels())
    {
        model->setVisibility(visible);
    }
}
void LVRScanPositionItem::setModelVisibility(int column, bool globalValue)
{
    if(checkState(column) == globalValue || globalValue == true)
    {
        setVisibility(checkState(column));
    }
}
LVRScanPositionItem::~LVRScanPositionItem()
{
    // TODO Auto-generated destructor stub
}

}
