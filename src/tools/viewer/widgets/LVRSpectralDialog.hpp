#ifndef SPECTRALDIALOG_H_
#define SPECTRALDIALOG_H_

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
#include "ui_LVRSpectralSettings.h"
#include "LVRRecordedFrameItem.hpp"
#include "../widgets/LVRHistogram.hpp"

using Ui::SpectralDialog;

namespace lvr
{

class LVRSpectralDialog : public QObject
{
    Q_OBJECT

public:
    LVRSpectralDialog(QTreeWidget* treeWidget, QMainWindow* mainWindow, PointBufferBridgePtr points, vtkRenderer* renderer);
    virtual ~LVRSpectralDialog();

public Q_SLOTS:
    void valueChangeFinished();
    void setTypeChannel();
    void setTypeGradient();
    void updateGradientView();
    void exitDialog();
    void showhistogram();
    
private:
    void connectSignalsAndSlots();
    void refreshDisplays();

    QMainWindow*         m_mainWindow;
    SpectralDialog*      m_spectralDialog;
    LVRHistogram*        m_histogram;                         
    QDialog*             m_dialog;
    PointBufferBridgePtr m_points;
    size_t               m_r, m_g, m_b;
    bool                 m_use_r, m_use_g, m_use_b;
    GradientType         m_gradient;
    size_t               m_gradientChannel;
    bool                 m_useNormalizedGradient;
    vtkRenderer*         m_renderer;
};

} // namespace lvr

#endif /* SPECTRALDIALOG_H_ */
