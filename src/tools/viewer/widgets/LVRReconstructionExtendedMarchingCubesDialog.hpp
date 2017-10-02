#ifndef RECONSTRUCTIONEXTENDEDMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONEXTENDEDMARCHINGCUBESDIALOG_H_

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

#include "ui_LVRReconstructionExtendedMarchingCubesDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

using Ui::ReconstructViaExtendedMarchingCubesDialog;

namespace lvr
{

class LVRReconstructViaExtendedMarchingCubesDialog  : public QObject
{
    Q_OBJECT

public:
    LVRReconstructViaExtendedMarchingCubesDialog(string decomposition, LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRReconstructViaExtendedMarchingCubesDialog();
    typedef ColorVertex<float, unsigned char>         cVertex;
    typedef Normal<float>                               cNormal;
    typedef PointsetSurface<cVertex>                    psSurface;
    typedef AdaptiveKSearchSurface<cVertex, cNormal>    akSurface;

public Q_SLOTS:
    void generateMesh();
    void toggleRANSACcheckBox(const QString &text);
    void switchGridSizeDetermination(int index);

private:
    void connectSignalsAndSlots();

    string                                          m_decomposition;
    ReconstructViaExtendedMarchingCubesDialog*      m_dialog;
    LVRPointCloudItem*                              m_pc;
    LVRModelItem*                                   m_parent;
    QTreeWidget*                                    m_treeWidget;
    LVRModelItem*                                   m_generatedModel;
    vtkRenderWindow*                                m_renderWindow;

};

} // namespace lvr

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */