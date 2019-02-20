#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRCAMDATAITEM_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRCAMDATAITEM_HPP

#include <QString>
#include <QTreeWidgetItem>
#include <QAbstractItemModel>
#include <QObject>

#include <vtkMatrix4x4.h>

#include <lvr2/io/ScanDataManager.hpp>

#include "../vtkBridge/LVRModelBridge.hpp"
#include "../vtkBridge/LVRBoundingBoxBridge.hpp"

#include "LVRBoundingBoxItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRPoseItem.hpp"


#include <vtkSmartPointer.h>
#include <vtkActor.h>

namespace lvr2
{


class LVRCamDataItem : public QTreeWidgetItem
{

    public:

        LVRCamDataItem(CamData data,
                        std::shared_ptr<ScanDataManager> sdm,
                        size_t idx,
                        vtkSmartPointer<vtkRenderer> renderer,
                        QString name = "",
                        QTreeWidgetItem *parent = NULL);

        ~LVRCamDataItem();

        QString getName() { return m_name; }

        Pose getPose() { return m_pose; }

        void setVisibility(bool visible);

        Matrix4<BaseVector<float> > getTransformation();

    private:

        vtkSmartPointer<vtkActor> genFrustrum();

        void reload(vtkSmartPointer<vtkRenderer> renderer);

        QString                                 m_name;
        std::shared_ptr<ScanDataManager>        m_sdm;
        size_t                                  m_idx;
        CamData                                 m_data;
        Pose                                    m_pose;
        LVRPoseItem*                            m_pItem;
        Matrix4<BaseVector<float> >             m_matrix;
        vtkSmartPointer<vtkActor>               m_frustrum_actor;
        vtkSmartPointer<vtkRenderer>            m_renderer;
};

} // namespace lvr2

#endif
