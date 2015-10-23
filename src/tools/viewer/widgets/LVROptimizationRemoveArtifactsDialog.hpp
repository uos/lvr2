#ifndef OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_
#define OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include <lvr/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr/reconstruction/FastReconstruction.hpp>
#include <lvr/io/PLYIO.hpp>
#include <lvr/geometry/Matrix4.hpp>
#include <lvr/geometry/HalfEdgeMesh.hpp>
#include <lvr/texture/Texture.hpp>
#include <lvr/texture/Transform.hpp>
#include <lvr/texture/Texturizer.hpp>
#include <lvr/texture/Statistics.hpp>
#include <lvr/geometry/QuadricVertexCosts.hpp>
#include <lvr/reconstruction/SharpBox.hpp>

#include "../vtkBridge/LVRModelBridge.hpp"

#include "LVROptimizationRemoveArtifactsDialogUI.h"
#include "LVRMeshItem.hpp"
#include "LVRModelItem.hpp"

using Ui::RemoveArtifactsDialog;

namespace lvr
{

class LVRRemoveArtifactsDialog : public QObject
{
    Q_OBJECT

public:
    LVRRemoveArtifactsDialog(LVRMeshItem* mesh, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRRemoveArtifactsDialog();
    typedef ColorVertex<float, unsigned char>         cVertex;
    typedef Normal<float>                               cNormal;

public Q_SLOTS:
    void removeArtifacts();

private:
    void connectSignalsAndSlots();

    RemoveArtifactsDialog*                 m_dialog;
    LVRMeshItem*                            m_mesh;
    LVRModelItem*                           m_optimizedModel;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;

};

} // namespace lvr

#endif /* OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_ */
