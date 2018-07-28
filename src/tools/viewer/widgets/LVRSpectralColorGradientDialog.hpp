#ifndef SPECTRALCOLORGRADIENTDIALOG_H_
#define SPECTRALCOLORGRADIENTDIALOG_H_

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkCameraRepresentation.h>
#include <vtkCameraInterpolator.h>
#include <vtkCommand.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <vtkWindowToImageFilter.h>


#include <lvr/io/PointBuffer.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/DataStruct.hpp>

#include <lvr/registration/ICPPointAlign.hpp>
//#include <lvr/io/boost/shared_ptr.hpp>

#include <lvr/display/ColorMap.hpp>
#include "../vtkBridge/LVRPointBufferBridge.hpp"

#include <QtWidgets>
#include<QDialog>
#include "ui_LVRSpectralColorGradient.h"
#include "LVRRecordedFrameItem.hpp"

using Ui::SpectralColorGradient;

namespace lvr
{

class LVRSpectralColorGradientDialog : public QObject
{
    Q_OBJECT

public:
    LVRSpectralColorGradientDialog(QTreeWidget* treeWidget, QMainWindow* mainWindow, PointBufferBridgePtr points, vtkRenderer* renderer);
    virtual ~LVRSpectralColorGradientDialog();

public Q_SLOTS:
    void setTypeGradient();
    void updateGradientView();
    void exitDialog();
    
private:
    void connectSignalsAndSlots();
    void refreshDisplays();

    QMainWindow*         m_mainWindow;
    SpectralColorGradient*      m_spectralDialog;                       
    QDialog*             m_dialog;
    PointBufferBridgePtr m_points;
    GradientType         m_gradient;
    size_t               m_gradientChannel;
    bool                 m_useNormalizedGradient;
    bool                 m_useNDVI;
    vtkRenderer*         m_renderer;
};

} // namespace lvr

#endif /* SPECTRALCOLORGRADIENTDIALOG_H_ */
