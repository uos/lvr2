#include "LVRScanDataItem.hpp"

#include "LVRModelItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRPoseItem.hpp"

#include "LVRItemTypes.hpp"

namespace lvr2
{

LVRScanDataItem::LVRScanDataItem(ScanData data, std::shared_ptr<ScanDataManager> sdm, size_t idx, QString name, QTreeWidget *parent) : QTreeWidgetItem(parent, LVRScanDataItemType)
{
    m_bbItem = NULL;
    m_data = data;
    m_name = name;
    m_sdm = sdm;
    m_idx = idx;

    setText(0, m_name);
    setCheckState(0, Qt::Checked);

    m_bb = BoundingBoxBridgePtr(new LVRBoundingBoxBridge(m_data.m_boundingBox));
    m_bbItem = new LVRBoundingBoxItem(m_bb, "Bounding Box", this);

    float pose[6];
    m_data.m_registration.transpose();
    m_data.m_registration.toPostionAngle(pose);

    Pose p;
    p.x = pose[0];
    p.y = pose[1];
    p.z = pose[2];
    p.r = pose[3]  * 57.295779513;
    p.t = pose[4]  * 57.295779513;
    p.p = pose[5]  * 57.295779513;
    m_bb->setPose(p);

    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_scandata_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);
}

void LVRScanDataItem::loadPointCloudData()
{
    m_data = m_sdm->loadPointCloudData(m_idx);

    m_model = ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model(m_data.m_points))));

    auto pcBridge = m_model->getPointBridge();
    LVRPointCloudItem *item = new LVRPointCloudItem(pcBridge);
    addChild(item);

    float pose[6];
    m_data.m_registration.transpose();
    m_data.m_registration.toPostionAngle(pose);

    LVRPoseItem *pitem = new LVRPoseItem(m_model, this);
    addChild(item);

    Pose p;
    p.x = pose[0];
    p.y = pose[1];
    p.z = pose[2];
    p.r = pose[3]  * 57.295779513;
    p.t = pose[4]  * 57.295779513;
    p.p = pose[5]  * 57.295779513;
    pitem->setPose(p);
    m_model->setPose(p);
}

void LVRScanDataItem::unloadPointCloudData()
{
    m_data.m_points = nullptr;
    m_data.m_pointsLoaded = false;

    QTreeWidgetItemIterator it(this);

    while (*it)
    {
        if ((*it)->type() == LVRPointCloudItemType)
        {
            removeChild(*it);
        }
        if ((*it)->type() == LVRPoseItemType)
        {
            removeChild(*it);
        }
        it++;
    }

    m_model = nullptr;
}

ModelBridgePtr LVRScanDataItem::getModelBridgePtr()
{
    return m_model;
}

void LVRScanDataItem::setVisibility(bool visible, bool pc_visible)
{
    if (m_model)
    {
        m_model->setVisibility(pc_visible && checkState(0) && visible);
    }

    if (m_bbItem)
    {
        m_bbItem->setVisibility(checkState(0) && visible);
    }
}

} // namespace lvr2
