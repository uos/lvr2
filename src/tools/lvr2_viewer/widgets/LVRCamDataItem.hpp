#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRCAMDATAITEM_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRCAMDATAITEM_HPP

#include <QString>
#include <QTreeWidgetItem>
#include <QAbstractItemModel>
#include <QObject>

#include <vtkMatrix4x4.h>

#include <Eigen/Dense>

#include "lvr2/io/ScanDataManager.hpp"
#include "lvr2/geometry/Transformable.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/registration/TransformUtils.hpp"

#include "../vtkBridge/LVRModelBridge.hpp"
#include "../vtkBridge/LVRBoundingBoxBridge.hpp"

#include "LVRBoundingBoxItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRPoseItem.hpp"
#include "LVRCvImageItem.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>

namespace lvr2
{


class LVRCamDataItem : public QTreeWidgetItem, public Transformable
{

    public:

        LVRCamDataItem(ScanImage& data,
                        std::shared_ptr<ScanDataManager> sdm,
                        size_t cam_id,
                        vtkSmartPointer<vtkRenderer> renderer,
                        QString name = "",
                        QTreeWidgetItem *parent = NULL);

        ~LVRCamDataItem();

        QString getName() { return m_name; }

        Pose getPose() { return m_pose; }

        void setVisibility(bool visible);

        size_t getCamId() { return m_cam_id; }

        void setCameraView();

    private:

        /**
         * @brief   Get Transformation from Camera frame to Global.
         *          QTree used as TF tree, lvr2::Transformable types
         *          are used to determine the global Transform.
         *          Output T can be used for:
         *          p_global = T * p_local
         *          
         * @return  Returns the Transformation as type lvr2::Matrix4
         */
        Transformd getGlobalTransform();

        std::vector<Vector3d > genFrustrumLVR(float scale=1.0);

        vtkSmartPointer<vtkActor> genFrustrum(float scale=1.0);

        void reload(vtkSmartPointer<vtkRenderer> renderer);

        QString                                 m_name;
        std::shared_ptr<ScanDataManager>        m_sdm;
        size_t                                  m_cam_id;
        ScanImage                               m_data;
        Pose                                    m_pose;
        LVRPoseItem*                            m_pItem;
        LVRCvImageItem*                         m_cvItem;
        Transformd                              m_matrix;
        Intrinsicsd                             m_intrinsics;

        vtkSmartPointer<vtkActor>               m_frustrum_actor;
        vtkSmartPointer<vtkRenderer>            m_renderer;
};

} // namespace lvr2

#endif
