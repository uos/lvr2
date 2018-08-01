#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

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

#include "../vtkBridge/LVRPointBufferBridge.hpp"

#include <QtWidgets>
#include <QDialog>
#include "ui_LVRHistogram.h"
#include "LVRPlotter.hpp"

using Ui::Histogram;

namespace lvr
{

class LVRHistogram : public QDialog
{
    Q_OBJECT

public:
    LVRHistogram(QWidget* parent, PointBufferPtr points);
    virtual ~LVRHistogram();

public Q_SLOTS:
    void refresh();
    
private:
    Histogram      m_histogram;
    floatArr       m_data;
    size_t         m_numChannels;
};

} // namespace lvr

#endif /* HISTOGRAM_H_ */