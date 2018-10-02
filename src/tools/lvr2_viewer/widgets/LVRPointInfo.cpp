#include "LVRPointInfo.hpp"

namespace lvr2
{

LVRPointInfo::LVRPointInfo(QWidget* parent, PointBuffer2Ptr pointBuffer, int pointId) :
    QDialog(parent)
{
    setAttribute(Qt::WA_DeleteOnClose);

    m_pointInfo.setupUi(this);

    size_t n = pointBuffer->numPoints();
    floatArr points = pointBuffer->getPointArray();

    if (pointId < 0 || pointId >= n)
    {
        done(0);
        return;
    }

    m_pointInfo.infoText->setText(QString("Selected: %1, %2, %3")
        .arg(points[pointId * 3], 10, 'g', 4)
        .arg(points[pointId * 3 + 1], 10, 'g', 4)
        .arg(points[pointId * 3 + 2], 10, 'g', 4));
    
    FloatChannelOptional spec_channels = pointBuffer->getFloatChannel("spectral_channels"); 
    size_t n_spec = spec_channels->numAttributes();
    m_numChannels = spec_channels->width();
    
    if (pointId >= n_spec)
    {
        done(0);
        return;
    }
    else
    {
        m_data = floatArr(new float[m_numChannels]);
        for (size_t i = 0; i < m_numChannels; i++)
        {
            m_data[i] = (*spec_channels)[pointId][i];
        }
    }

    refresh();

    QObject::connect(m_pointInfo.shouldScale, SIGNAL(stateChanged(int)), this, SLOT(refresh()));
}

LVRPointInfo::~LVRPointInfo()
{
    // TODO Auto-generated destructor stub
}

void LVRPointInfo::refresh()
{
    if (m_pointInfo.shouldScale->isChecked())
    {
        m_pointInfo.plotter->setPoints(m_data, m_numChannels);
    }
    else
    {
        m_pointInfo.plotter->setPoints(m_data, m_numChannels, 0, 1);
    }

    show();
    raise();
    activateWindow();
}

} // namespace lvr2
