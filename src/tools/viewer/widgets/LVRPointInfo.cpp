#include "LVRPointInfo.hpp"

//#include <vtkFFMPEGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <lvr/io/PointBuffer.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/DataStruct.hpp>
#include <lvr/registration/ICPPointAlign.hpp>
#include "lvr/io/PointBuffer.hpp"

#include <cstring>

namespace lvr
{

LVRPointInfo::LVRPointInfo(QTreeWidget* treeWidget)
{
    // Setup DialogUI and events
    m_dialog = new QDialog(treeWidget);
    m_pointInfo = new PointInfo;
    m_pointInfo->setupUi(m_dialog);

    m_plotter = new LVRPlotter(m_dialog);
    m_pointInfo->gridLayout->addWidget(m_plotter, 4, 0, 1, 1);

    QObject::connect(m_pointInfo->shouldScale, SIGNAL(stateChanged(int)), this, SLOT(refresh(int)));
}

LVRPointInfo::~LVRPointInfo()
{
    // TODO Auto-generated destructor stub
}

void LVRPointInfo::setPointBuffer(PointBufferPtr points)
{
    m_points = points;
}

void LVRPointInfo::setPoint(int pointId)
{
    m_pointId = pointId;

    size_t n;
    floatArr points = m_points->getPointArray(n);

    if (pointId < 0 || pointId >= n)
    {
        m_dialog->done(0);
        return;
    }

    m_pointInfo->infoText->setText(QString("Selected: %1, %2, %3")
        .arg(points[pointId * 3], 10, 'g', 4)
        .arg(points[pointId * 3 + 1], 10, 'g', 4)
        .arg(points[pointId * 3 + 2], 10, 'g', 4));
    
    size_t n_spec, n_channels;
    floatArr spec = m_points->getPointSpectralChannelsArray(n_spec, n_channels);
    
    if (pointId >= n_spec)
    {
        m_plotter->removePoints();
    }
    else
    {
        floatArr data = floatArr(new float[n_channels]);
        for (int i = 0; i < n_channels; i++)
        {
            data[i] = spec[pointId * n_channels + i];
        }
        if (m_pointInfo->shouldScale->isChecked())
        {
            m_plotter->setPoints(data, n_channels);
        }
        else
        {
            m_plotter->setPoints(data, n_channels, 0, 1);
        }
    }

    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void LVRPointInfo::refresh(int)
{
    setPoint(m_pointId);
}

}