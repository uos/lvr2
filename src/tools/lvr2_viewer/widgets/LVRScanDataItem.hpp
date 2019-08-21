#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRSCANDATAITEM_HPP

#include <QString>
#include <QTreeWidgetItem>

#include <vtkMatrix4x4.h>

#include "lvr2/io/ScanDataManager.hpp"
#include "lvr2/geometry/Transformable.hpp"

#include "../vtkBridge/LVRModelBridge.hpp"
#include "../vtkBridge/LVRBoundingBoxBridge.hpp"

#include "LVRBoundingBoxItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRPoseItem.hpp"
#include "LVRCamDataItem.hpp"

#include <Eigen/Dense>

namespace lvr2
{

class LVRScanDataItem : public QTreeWidgetItem, public Transformable
{
    public:

        LVRScanDataItem(ScanPtr data,
                        std::shared_ptr<ScanDataManager> sdm,
                        size_t idx,
                        vtkSmartPointer<vtkRenderer> renderer,
                        QString name = "",
                        QTreeWidgetItem *parent = NULL);

        ~LVRScanDataItem();

        void loadPointCloudData(vtkSmartPointer<vtkRenderer> renderer);

        void unloadPointCloudData(vtkSmartPointer<vtkRenderer> renderer);

        size_t getScanId() { return m_idx; }

        QString getName() { return m_name; }

        Pose getPose() { return getModelBridgePtr()->getPose(); }

        ModelBridgePtr getModelBridgePtr();

        BoundingBoxBridgePtr getBoundingBoxBridge() { return m_bb; }

        void setVisibility(bool visible, bool pc_visible);

        bool isPointCloudLoaded();

        void reload();

    private:

        void reload(vtkSmartPointer<vtkRenderer> renderer);

        QString                                 m_name;
        std::shared_ptr<ScanDataManager>        m_sdm;
        size_t                                  m_idx;
        ScanPtr                                 m_data;
        ModelBridgePtr                          m_model;
        BoundingBoxBridgePtr                    m_bb;
        Pose                                    m_pose;
        LVRBoundingBoxItem                     *m_bbItem;
        LVRPointCloudItem                      *m_pcItem;
        LVRPoseItem                            *m_pItem;
        QTreeWidgetItem                        *m_showSpectralsItem;
        Transformd                              m_matrix;
        vtkSmartPointer<vtkRenderer>            m_renderer;

};

} // namespace lvr2

#endif
