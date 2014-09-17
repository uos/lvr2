#ifndef ANIMATIONDIALOG_H_
#define ANIMATIONDIALOG_H_

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkCameraRepresentation.h>
#include <vtkCameraInterpolator.h>
#include <vtkCommand.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <QtGui>
#include "LVRAnimationDialogUI.h"

using Ui::AnimationDialog;

namespace lvr
{

class LVRAnimationDialog : public QObject
{
    Q_OBJECT

public:
    LVRAnimationDialog(vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor, vtkSmartPointer<vtkCameraRepresentation> pathCamera, QTreeWidget* treeWidget);
    virtual ~LVRAnimationDialog();

public Q_SLOTS:
    void addFrame();
    void removeFrame();
    void clearFrames();
    void play();

private:
    void connectSignalsAndSlots();

    AnimationDialog*                           m_dialog;
    vtkSmartPointer<vtkRenderWindowInteractor>  m_renderWindowInteractor;
    vtkSmartPointer<vtkCameraRepresentation>    m_pathCamera;
    QTreeWidget*                                m_treeWidget;
    int                                         m_timerID;
    unsigned int                               m_frameCounter;
};

} // namespace lvr

#endif /* ANIMATIONDIALOG_H_ */
