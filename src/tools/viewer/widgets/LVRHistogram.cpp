#include "LVRHistogram.hpp"

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

LVRHistogram::LVRHistogram(QWidget* parent, PointBufferPtr points)
    : QDialog(parent)
{
    m_histogram.setupUi(this);
    m_histogram.plotter->setPlotMode(PlotMode::BAR);

    size_t n;
    size_t n_spec;

    floatArr spec = points->getPointSpectralChannelsArray(n_spec, m_numChannels);
       
    m_data = floatArr(new float[m_numChannels]);

    #pragma omp parallel for
    for (int channel = 0; channel < m_numChannels; channel++)
    {
        m_data[channel] = 0;
       
        for (int i = 0; i < n_spec; i++)
        {
            m_data[channel] += spec[m_numChannels * i + channel];           
        }                                       
        m_data[channel] /= n_spec;
    }

    refresh();

    QObject::connect(m_histogram.shouldScale, SIGNAL(stateChanged(int)), this, SLOT(refresh()));
}

LVRHistogram::~LVRHistogram()
{
}

void LVRHistogram::refresh()
{
    if (m_histogram.shouldScale->isChecked())
    {
        m_histogram.plotter->setPoints(m_data, m_numChannels);
    }
    else
    {
        m_histogram.plotter->setPoints(m_data, m_numChannels, 0, 1);
    }

    show();
    raise();
    activateWindow();
}

}