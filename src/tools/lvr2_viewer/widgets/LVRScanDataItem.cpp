#include "LVRScanDataItem.hpp"
#include "LVRModelItem.hpp"

#include "LVRItemTypes.hpp"

namespace lvr2
{

LVRScanDataItem::LVRScanDataItem(ScanData data, std::shared_ptr<ScanDataManager> sdm, size_t idx, vtkSmartPointer<vtkRenderer> renderer, QString name, QTreeWidgetItem *parent) : QTreeWidgetItem(parent, LVRScanDataItemType)
{
    m_pcItem = nullptr;
    m_pItem  = nullptr;
    m_bbItem = nullptr;
    m_data   = data;
    m_name   = name;
    m_sdm    = sdm;
    m_idx    = idx;
    //m_model.reset();


    // init pose
    float pose[6];
    m_data.m_registration.transpose();
    m_data.m_registration.toPostionAngle(pose);
    m_data.m_registration.transpose();

    m_pose.x = pose[0];
    m_pose.y = pose[1];
    m_pose.z = pose[2];
    m_pose.r = pose[3]  * 57.295779513;
    m_pose.t = pose[4]  * 57.295779513;
    m_pose.p = pose[5]  * 57.295779513;


    // init bb
    m_bb = BoundingBoxBridgePtr(new LVRBoundingBoxBridge(m_data.m_boundingBox));
    m_bbItem = new LVRBoundingBoxItem(m_bb, "Bounding Box", this);
    renderer->AddActor(m_bb->getActor());
    m_bb->setPose(m_pose);


    //init preview
    if (m_data.m_preview)
    {
        loadPreview(renderer);
    }


    setText(0, m_name);
    setCheckState(0, Qt::Checked);
}

void LVRScanDataItem::loadPreview(vtkSmartPointer<vtkRenderer> renderer)
{
        if (m_model)
        {
            m_model->removeActors(renderer);
        }

        if (m_pcItem)
        {
            delete m_pcItem;
        }

        m_model = ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model(m_data.m_preview))));
        m_pcItem = new LVRPointCloudItem(m_model->getPointBridge(), this);

        m_model->addActors(renderer);
        m_model->setPose(m_pose);

        setText(1, "Preview");
}

bool LVRScanDataItem::isPointCloudLoaded()
{
    return m_data.m_pointsLoaded;
}

void LVRScanDataItem::loadPointCloudData(vtkSmartPointer<vtkRenderer> renderer)
{
    m_sdm->loadPointCloudData(m_data);

    if (m_data.m_pointsLoaded)
    {
        if (m_model)
        {
            m_model->removeActors(renderer);
        }

        if (m_pcItem)
        {
            delete m_pcItem;
        }

        m_model = ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model(m_data.m_points))));
        m_pcItem = new LVRPointCloudItem(m_model->getPointBridge(), this);
        m_model->addActors(renderer);

        if (!m_pItem)
        {
            m_pItem = new LVRPoseItem(m_model, this);
        }

        m_pItem->setPose(m_pose);
        m_model->setPose(m_pose);

        setText(1, "");
    }
}

void LVRScanDataItem::unloadPointCloudData(vtkSmartPointer<vtkRenderer> renderer)
{
    m_model->removeActors(renderer);
    m_model.reset();
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

    if (m_data.m_preview)
    {
        loadPreview(renderer);
    }
}

ModelBridgePtr LVRScanDataItem::getModelBridgePtr()
{
    return m_model;
}

void LVRScanDataItem::setVisibility(bool visible, bool pc_visible)
{
    bool parents_checked = true;
    QTreeWidgetItem *parent = this;

    while (parent)
    {
        if (!parent->checkState(0))
        {
            parents_checked = false;
            break;
        }

        parent = parent->parent();
    }

    visible = visible && parents_checked;

    if (m_model)
    {
        m_model->setVisibility(pc_visible && visible);
    }

    if (m_bbItem)
    {
        m_bbItem->setVisibility(visible);
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
