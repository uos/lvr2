#ifndef RECONSTRUCTIONMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONMARCHINGCUBESDIALOG_H_

#include <vtkRenderWindow.h>

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

#include "LVRReconstructionMarchingCubesDialogUI.h"
#include "LVRPointCloudItem.hpp"

using Ui::ReconstructViaMarchingCubesDialog;

namespace lvr
{

class LVRReconstructViaMarchingCubesDialog : public QObject
{
    Q_OBJECT

public:
    LVRReconstructViaMarchingCubesDialog(LVRPointCloudItem* parent, vtkRenderWindow* renderer);
    virtual ~LVRReconstructViaMarchingCubesDialog();
    typedef ColorVertex<float, unsigned char>         cVertex;
    typedef Normal<float>                               cNormal;
    typedef PointsetSurface<cVertex>                    psSurface;
    typedef AdaptiveKSearchSurface<cVertex, cNormal>    akSurface;

public Q_SLOTS:
    void save();
    void printAllValues();
    void toggleRANSACcheckBox(const QString &text);
    void switchGridSizeDetermination(int index);

private:
    void connectSignalsAndSlots();

    ReconstructViaMarchingCubesDialog*      m_dialog;
    LVRPointCloudItem*                      m_parent;
    vtkRenderWindow*                        m_renderWindow;

};

} // namespace lvr

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */
