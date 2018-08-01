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

LVRPointInfo::LVRPointInfo(QWidget* parent, PointBufferPtr points, int pointId) :
    QDialog(parent), m_points(points), m_pointId(pointId)
{
    setAttribute(Qt::WA_DeleteOnClose);

    m_pointInfo.setupUi(this);

    QObject::connect(m_pointInfo.shouldScale, SIGNAL(stateChanged(int)), this, SLOT(refresh()));
    refresh();
}

LVRPointInfo::~LVRPointInfo()
{
    // TODO Auto-generated destructor stub
}

void LVRPointInfo::refresh()
{
    size_t n;
    floatArr points = m_points->getPointArray(n);

    if (m_pointId < 0 || m_pointId >= n)
    {
        done(0);
        return;
    }

    m_pointInfo.infoText->setText(QString("Selected: %1, %2, %3")
        .arg(points[m_pointId * 3], 10, 'g', 4)
        .arg(points[m_pointId * 3 + 1], 10, 'g', 4)
        .arg(points[m_pointId * 3 + 2], 10, 'g', 4));
    
    size_t n_spec, n_channels;
    floatArr spec = m_points->getPointSpectralChannelsArray(n_spec, n_channels);
    
    if (m_pointId >= n_spec)
    {
        m_pointInfo.plotter->removePoints();
    }
    else
    {
        floatArr data = floatArr(new float[n_channels]);
        for (int i = 0; i < n_channels; i++)
        {
            data[i] = spec[m_pointId * n_channels + i];
        }
        if (m_pointInfo.shouldScale->isChecked())
        {
            m_pointInfo.plotter->setPoints(data, n_channels);
        }
        else
        {
            m_pointInfo.plotter->setPoints(data, n_channels, 0, 1);
        }
    }

    show();
    raise();
    activateWindow();
}

}