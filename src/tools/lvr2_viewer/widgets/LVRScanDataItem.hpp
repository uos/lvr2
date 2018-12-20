#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP

#include <QString>
#include <QTreeWidgetItem>

#include <vtkMatrix4x4.h>

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

        LVRScanDataItem(ScanData data,
                        std::shared_ptr<ScanDataManager> sdm,
                        size_t idx,
                        vtkSmartPointer<vtkRenderer> renderer,
                        QString name = "",
                        QTreeWidgetItem *parent = NULL);

        ~LVRScanDataItem();

        void loadPointCloudData(vtkSmartPointer<vtkRenderer> renderer);

        void unloadPointCloudData(vtkSmartPointer<vtkRenderer> renderer);

        QString getName() { return m_name; }

        Pose getPose() { return getModelBridgePtr()->getPose(); }

        ModelBridgePtr getModelBridgePtr();

        BoundingBoxBridgePtr getBoundingBoxBridge() { return m_bb; }

        void setVisibility(bool visible, bool pc_visible);

        bool isPointCloudLoaded();


    private:

        void reload(vtkSmartPointer<vtkRenderer> renderer);

        QString                                 m_name;
        std::shared_ptr<ScanDataManager>        m_sdm;
        size_t                                  m_idx;
        ScanData                                m_data;
        ModelBridgePtr                          m_model;
        BoundingBoxBridgePtr                    m_bb;
        Pose                                    m_pose;
        LVRBoundingBoxItem                     *m_bbItem;
        LVRPointCloudItem                      *m_pcItem;
        LVRPoseItem                            *m_pItem;
        QTreeWidgetItem                     *m_showSpectralsItem;
        Matrix4<BaseVector<float> >             m_matrix;
};

} // namespace lvr2

#endif
