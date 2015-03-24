#ifndef RECONSTRUCTIONMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONMARCHINGCUBESDIALOG_H_

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

#include "LVRReconstructionMarchingCubesDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

#include <QProgressDialog>

using Ui::ReconstructViaMarchingCubesDialog;

namespace lvr
{

class LVRReconstructViaMarchingCubesDialog : public QObject
{
    Q_OBJECT

public:
    LVRReconstructViaMarchingCubesDialog(string decomposition, LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRReconstructViaMarchingCubesDialog();
    typedef ColorVertex<float, unsigned char>         cVertex;
    typedef Normal<float>                               cNormal;
    typedef PointsetSurface<cVertex>                    psSurface;
    typedef AdaptiveKSearchSurface<cVertex, cNormal>    akSurface;

    static void updateProgressbar(int p);
    static LVRReconstructViaMarchingCubesDialog* master;

    void setProgressvalue(int v);

public Q_SLOTS:
    void generateMesh();
    void toggleRANSACcheckBox(const QString &text);
    void switchGridSizeDetermination(int index);


private:
    void connectSignalsAndSlots();

    string                                  m_decomposition;
    ReconstructViaMarchingCubesDialog*      m_dialog;
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    LVRModelItem*                           m_generatedModel;
    vtkRenderWindow*                        m_renderWindow;
    QProgressDialog*						m_progressDialog;



};

} // namespace lvr

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */
