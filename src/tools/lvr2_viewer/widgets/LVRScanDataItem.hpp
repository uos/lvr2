#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP

#include <QString>
#include <QTreeWidgetItem>

#include <lvr2/io/ScanDataManager.hpp>

#include "../vtkBridge/LVRModelBridge.hpp"

namespace lvr2
{

class LVRScanDataItem : public QTreeWidgetItem
{
    public:

        LVRScanDataItem(ScanData data, std::shared_ptr<ScanDataManager> sdm, size_t idx, QString name = "", QTreeWidget* parent = NULL);
        void loadPointCloudData();
        void unloadPointCloudData();
        QString getName() { return m_name; }
        Pose getPose() { return getModelBridgePtr()->getPose(); }

        ModelBridgePtr getModelBridgePtr();



    private:

        QString                                 m_name;
        std::shared_ptr<ScanDataManager>        m_sdm;
        size_t                                  m_idx;
        ScanData                                m_data;
        ModelBridgePtr                          m_model;
};

} // namespace lvr2

#endif
