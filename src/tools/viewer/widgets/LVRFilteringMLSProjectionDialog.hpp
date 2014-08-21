#ifndef FILTERINGMLSPROJECTIONDIALOG_H_
#define FILTERINGMLSPROJECTIONDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "reconstruction/FastReconstruction.hpp"
#include "io/PLYIO.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/HalfEdgeMesh.hpp"
#include "texture/Texture.hpp"
#include "texture/Transform.hpp"
#include "texture/Texturizer.hpp"
#include "texture/Statistics.hpp"
#include "geometry/QuadricVertexCosts.hpp"
#include "reconstruction/SharpBox.hpp"
#include "../vtkBridge/LVRModelBridge.hpp"

#include "LVRFilteringMLSProjectionDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

using Ui::MLSProjectionDialog;

namespace lvr
{

class LVRMLSProjectionDialog : public QObject
{
    Q_OBJECT

public:
    LVRMLSProjectionDialog(LVRPointCloudItem* pc_item, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRMLSProjectionDialog();
    typedef ColorVertex<float, unsigned char>         cVertex;
    typedef Normal<float>                               cNormal;

public Q_SLOTS:
    void applyMLSProjection();

private:
    void connectSignalsAndSlots();

    MLSProjectionDialog*                    m_dialog;
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_optimizedPointCloud;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;
};

} // namespace lvr

#endif /* FILTERINGMLSPROJECTIONDIALOG_H_ */
