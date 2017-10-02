#ifndef RECONSTRUCTIONMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONMARCHINGCUBESDIALOG_H_

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

#include "ui_LVRReconstructionMarchingCubesDialogUI.h"
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
    typedef ColorVertex<float, unsigned char>           cVertex;
    typedef Normal<float>                               cNormal;
    typedef PointsetSurface<cVertex>                    psSurface;
    typedef AdaptiveKSearchSurface<cVertex, cNormal>    akSurface;

    static void updateProgressbar(int p);
    static void updateProgressbarTitle(string t);


    void setProgressValue(int v);
    void setProgressTitle(string);

public Q_SLOTS:
    void generateMesh();
    void toggleRANSACcheckBox(const QString &text);
    void switchGridSizeDetermination(int index);

Q_SIGNALS:
    void progressValueChanged(int);
    void progressTitleChanged(const QString&);


private:
    void connectSignalsAndSlots();

    string                                          m_decomposition;
    ReconstructViaMarchingCubesDialog*              m_dialog;
    LVRPointCloudItem*                              m_pc;
    LVRModelItem*                                   m_parent;
    QTreeWidget*                                    m_treeWidget;
    LVRModelItem*                                   m_generatedModel;
    vtkRenderWindow*                                m_renderWindow;
    QProgressDialog*                                m_progressDialog;
    static LVRReconstructViaMarchingCubesDialog*    m_master;


};

} // namespace lvr

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */