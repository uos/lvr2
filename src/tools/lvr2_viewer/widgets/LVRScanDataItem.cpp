#include "LVRScanDataItem.hpp"
#include "LVRModelItem.hpp"
#include "LVRItemTypes.hpp"

#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{

LVRScanDataItem::LVRScanDataItem(
    ScanPtr data, std::shared_ptr<ScanDataManager> sdm, size_t idx,
    vtkSmartPointer<vtkRenderer> renderer, 
    QString name, QTreeWidgetItem *parent) 
    : QTreeWidgetItem(parent, LVRScanDataItemType) ,m_renderer(renderer)
{
    m_showSpectralsItem = nullptr;
    m_pcItem = nullptr;
    m_pItem  = nullptr;
    m_bbItem = nullptr;
    m_data   = data;
    m_name   = name;
    m_sdm    = sdm;
    m_idx    = idx;


    // init pose
    double pose[6];
    eigenToEuler<double>(m_data->registration, pose);

    m_matrix = m_data->registration;

    m_pose.x = pose[0];
    m_pose.y = pose[1];
    m_pose.z = pose[2];
    m_pose.r = pose[3] * 57.295779513;
    m_pose.t = pose[4] * 57.295779513;
    m_pose.p = pose[5] * 57.295779513;

    m_pItem = new LVRPoseItem(ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model))), this);

    m_pItem->setPose(m_pose);

    // init bb
    m_bb = BoundingBoxBridgePtr(new LVRBoundingBoxBridge(m_data->boundingBox));
    m_bbItem = new LVRBoundingBoxItem(m_bb, "Bounding Box", this);
    renderer->AddActor(m_bb->getActor());
    m_bb->setPose(m_pose);

    setTransform(m_matrix);

    // load data
    reload();

    setText(0, m_name);
    setCheckState(0, Qt::Checked);
}

void LVRScanDataItem::reload()
{
    reload(m_renderer);
}

void LVRScanDataItem::reload(vtkSmartPointer<vtkRenderer> renderer)
{
        if (m_model)
        {
            m_model->removeActors(renderer);
        }

        if (m_pcItem)
        {
            delete m_pcItem;
        }

        if (m_showSpectralsItem)
        {
            delete m_showSpectralsItem;
            m_showSpectralsItem = nullptr;
        }

        if (m_data->points)
        {
            m_model = ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model(m_data->points))));
            m_pcItem = new LVRPointCloudItem(m_model->getPointBridge(), this);

            m_model->addActors(renderer);
            m_model->setTransform(m_matrix);

            if (!isPointCloudLoaded())
            {
                setText(1, "(Preview)");
            }

            if (m_data->points->getUCharChannel("spectral_channels"))
            {
                m_showSpectralsItem = new QTreeWidgetItem(this);
                m_showSpectralsItem->setText(0, "Spectrals");
                m_showSpectralsItem->setCheckState(0, Qt::Checked);
            }
        }
}

bool LVRScanDataItem::isPointCloudLoaded()
{
    return m_data->pointsLoaded;
}

void LVRScanDataItem::loadPointCloudData(vtkSmartPointer<vtkRenderer> renderer)
{
    m_sdm->loadPointCloudData(m_data);

    if (isPointCloudLoaded())
    {
        reload(renderer);

        setText(1, "");
    }
}

void LVRScanDataItem::unloadPointCloudData(vtkSmartPointer<vtkRenderer> renderer)
{
    m_sdm->loadPointCloudData(m_data, true);

    reload(renderer);
}

ModelBridgePtr LVRScanDataItem::getModelBridgePtr()
{
    return m_model;
}

void LVRScanDataItem::setVisibility(bool visible, bool pc_visible)
{
    if (!this->checkState(0))
    {
        visible = false;
    }

    for (int i = 0; i < childCount(); i++)
    {
        QTreeWidgetItem* item = child(i);
        
        if(item->type() == LVRCamerasItemType)
        {
            for(int j=0; j < item->childCount(); j++)
            {
                QTreeWidgetItem* cam_item = item->child(j);

                if(cam_item->type() == LVRCamDataItemType)
                {
                    LVRCamDataItem* cam_item_c = static_cast<LVRCamDataItem*>(cam_item);
                    cam_item_c->setVisibility(visible);
                }
                
            }
        }

        item->setHidden(!visible);
    }

    if (m_model)
    {
        m_model->setVisibility(pc_visible && visible);
    }

    if (m_showSpectralsItem)
    {
        m_model->getPointBridge()->setColorsVisibility(m_showSpectralsItem->checkState(0));
    }

    if (m_bbItem)
    {
        m_bbItem->setVisibility(visible);
    }

}

LVRScanDataItem::~LVRScanDataItem()
{
    // we don't want to do delete m_bbItem, m_pItem and m_pcItem here
    // because QTreeWidgetItem deletes its childs automatically in its destructor.
}

} // namespace lvr2
