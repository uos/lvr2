#ifndef POINTINFO_H_
#define POINTINFO_H_

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
#include "ui_LVRPointInfo.h"
#include "LVRPlotter.hpp"

using Ui::PointInfo;

namespace lvr
{

class LVRPointInfo : public QObject
{
    Q_OBJECT

public:
    LVRPointInfo(QTreeWidget* treeWidget, PointBufferPtr points);
    virtual ~LVRPointInfo();

    void setPoint(size_t pointId);

public Q_SLOTS:
    void refresh(int);
    
private:
    PointInfo*     m_pointInfo;
    QDialog*       m_dialog;
    LVRPlotter*    m_plotter;
    PointBufferPtr m_points;
    size_t         m_pointId;
};

} // namespace lvr

#endif /* POINTINFO_H_ */