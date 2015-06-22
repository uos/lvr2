#ifndef RECONSTRUCTIONEXTENDEDMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONEXTENDEDMARCHINGCUBESDIALOG_H_

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

#include "LVRReconstructionExtendedMarchingCubesDialogUI.h"
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

    string                                  		m_decomposition;
    ReconstructViaExtendedMarchingCubesDialog*      m_dialog;
    LVRPointCloudItem*                     		    m_pc;
    LVRModelItem*                           		m_parent;
    QTreeWidget*                           		 	m_treeWidget;
    LVRModelItem*                           		m_generatedModel;
    vtkRenderWindow*                        		m_renderWindow;

};

} // namespace lvr

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */
