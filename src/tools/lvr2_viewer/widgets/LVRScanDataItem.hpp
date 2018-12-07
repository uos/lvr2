#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP

#include <QString>
#include <QTreeWidgetItem>

#include <lvr2/io/ScanDataManager.hpp>

#include "../vtkBridge/LVRModelBridge.hpp"
#include "../vtkBridge/LVRBoundingBoxBridge.hpp"

#include "LVRBoundingBoxItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRPoseItem.hpp"

namespace lvr2
{

class LVRScanDataItem : public QTreeWidgetItem
{
    public:

        LVRScanDataItem(ScanData data, std::shared_ptr<ScanDataManager> sdm, size_t idx, QString name = "", QTreeWidgetItem *parent = NULL);

        ~LVRScanDataItem();

        void loadPointCloudData();

        void unloadPointCloudData();

        QString getName() { return m_name; }

        Pose getPose() { return getModelBridgePtr()->getPose(); }

        ModelBridgePtr getModelBridgePtr();

        BoundingBoxBridgePtr getBoundingBoxBridge() { return m_bb; }

        void setVisibility(bool visible, bool pc_visible);



    private:

        QString                                 m_name;
        std::shared_ptr<ScanDataManager>        m_sdm;
        size_t                                  m_idx;
        ScanData                                m_data;
        ModelBridgePtr                          m_model;
        BoundingBoxBridgePtr                    m_bb;
        LVRBoundingBoxItem                     *m_bbItem;
        LVRPointCloudItem                      *m_pcItem;
        LVRPoseItem                            *m_pItem;
};

} // namespace lvr2

#endif
