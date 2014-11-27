#ifndef OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_
#define OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_

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
