#ifndef RECORDPATHDIALOG_H_
#define RECORDPATHDIALOG_H_

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
#include "LVRRecordPathDialogUI.h"
#include "../app/LVRTimerCallback.hpp"

using Ui::RecordPathDialog;

namespace lvr
{

class LVRRecordPathDialog : public QObject
{
    Q_OBJECT

public:
    LVRRecordPathDialog(vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor, vtkSmartPointer<vtkCameraRepresentation> pathCamera, vtkSmartPointer<LVRTimerCallback> timerCallback, QTreeWidget* treeWidget);
    virtual ~LVRRecordPathDialog();

public Q_SLOTS:
    void recordPath();
    void changeMode(const QString &text);

private:
    void connectSignalsAndSlots();

    RecordPathDialog*                           m_dialog;
    vtkSmartPointer<vtkRenderWindowInteractor>  m_renderWindowInteractor;
    vtkSmartPointer<vtkCameraRepresentation>    m_pathCamera;
    vtkSmartPointer<LVRTimerCallback>           m_timerCallback;
    QTreeWidget*                                m_treeWidget;
    int                                         m_timerID;
};

} // namespace lvr

#endif /* RECORDPATHDIALOG_H_ */
