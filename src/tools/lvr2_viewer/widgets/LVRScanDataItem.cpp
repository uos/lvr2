#include "LVRScanDataItem.hpp"
#include "LVRModelItem.hpp"

#include "LVRItemTypes.hpp"

namespace lvr2
{

LVRScanDataItem::LVRScanDataItem(ScanData data, std::shared_ptr<ScanDataManager> sdm, size_t idx, QString name, QTreeWidget *parent) : QTreeWidgetItem(parent, LVRScanDataItemType)
{
    m_bbItem = NULL;
    m_pcItem = NULL;
    m_pItem  = NULL;
    m_data   = data;
    m_name   = name;
    m_sdm    = sdm;
    m_idx    = idx;

    setText(0, m_name);
    setCheckState(0, Qt::Checked);

    m_bb = BoundingBoxBridgePtr(new LVRBoundingBoxBridge(m_data.m_boundingBox));

    if (!m_bbItem)
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
    m_sdm->loadPointCloudData(m_data);

    m_model = ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model(m_data.m_points))));

    auto pcBridge = m_model->getPointBridge();

    if (!m_pcItem)
    {
        m_pcItem = new LVRPointCloudItem(pcBridge, this);
    }

    float pose[6];
    m_data.m_registration.transpose();
    m_data.m_registration.toPostionAngle(pose);

    if (!m_pItem)
    {
        m_pItem = new LVRPoseItem(m_model, this);
    }

    Pose p;
    p.x = pose[0];
    p.y = pose[1];
    p.z = pose[2];
    p.r = pose[3]  * 57.295779513;
    p.t = pose[4]  * 57.295779513;
    p.p = pose[5]  * 57.295779513;
    m_pItem->setPose(p);
    m_model->setPose(p);
}

void LVRScanDataItem::unloadPointCloudData()
{
    m_data.m_points.reset();
    m_data.m_pointsLoaded = false;

    if (m_pcItem)
    {
        delete m_pcItem;
        m_pcItem = nullptr;
    }

    if (m_pItem)
    {
        delete m_pItem;
        m_pItem = nullptr;
    }

    m_model.reset();
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

LVRScanDataItem::~LVRScanDataItem()
{
    m_model.reset();
    m_data.m_points.reset();
    if (m_bbItem)
    {
        delete m_bbItem;
    }
    if (m_pcItem)
    {
        delete m_pcItem;
    }

    if (m_pItem)
    {
        delete m_pItem;
    }
}

} // namespace lvr2
