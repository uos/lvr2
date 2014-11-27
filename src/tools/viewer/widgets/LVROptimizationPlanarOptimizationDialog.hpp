#ifndef OPTIMIZATIONPLANAROPTIMIZATIONDIALOG_H_
#define OPTIMIZATIONPLANAROPTIMIZATIONDIALOG_H_

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

#include "LVROptimizationPlanarOptimizationDialogUI.h"
#include "LVRMeshItem.hpp"
#include "LVRModelItem.hpp"

using Ui::PlanarOptimizationDialog;

namespace lvr
{

class LVRPlanarOptimizationDialog : public QObject
{
    Q_OBJECT

public:
    LVRPlanarOptimizationDialog(LVRMeshItem* mesh, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRPlanarOptimizationDialog();
    typedef ColorVertex<float, unsigned char>         cVertex;
    typedef Normal<float>                               cNormal;

public Q_SLOTS:
    void optimizeMesh();
    void toggleSmallRegionRemoval(int state);
    void toggleRetesselation(int state);

private:
    void connectSignalsAndSlots();

    PlanarOptimizationDialog*               m_dialog;
    LVRMeshItem*                            m_mesh;
    LVRModelItem*                           m_optimizedModel;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;

};

} // namespace lvr

#endif /* OPTIMIZATIONPLANAROPTIMIZATIONDIALOG_H_ */
