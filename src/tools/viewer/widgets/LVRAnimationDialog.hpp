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
#include "LVRRecordedFrameItem.hpp"

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
    void changeInterpolation(const QString& text);
    void play();

private:
    void connectSignalsAndSlots();

    AnimationDialog*                           m_dialog;
    QListWidget*                                m_timeline;
    vtkSmartPointer<vtkRenderWindowInteractor>  m_renderWindowInteractor;
    vtkSmartPointer<vtkCameraRepresentation>    m_pathCamera;
    vtkSmartPointer<vtkCamera>                  m_mainCamera;
    QTreeWidget*                                m_treeWidget;
    int                                         m_timerID;
    unsigned int                               m_frameCounter;
};

} // namespace lvr

#endif /* ANIMATIONDIALOG_H_ */
