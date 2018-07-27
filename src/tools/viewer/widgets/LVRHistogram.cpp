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

LVRHistogram::LVRHistogram()
{
    //Setup DialogUI and events
    m_dialog = new QDialog();
    m_histogram = new Histogram;
    m_histogram->setupUi(m_dialog);

    m_plotter = new LVRPlotter(m_dialog, false);
    m_histogram->gridLayout->addWidget(m_plotter, 4, 0, 1, 1);

    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();

    //QObject::connect(m_histogram->shouldScale, SIGNAL(stateChanged(int)), this, SLOT(refresh(int)));
}

LVRHistogram::~LVRHistogram()
{
    // TODO Auto-generated destructor stub
}

void LVRHistogram::setPointBuffer(PointBufferPtr points)
{
    m_points = points;
}

void LVRHistogram::sethistogram()
{  
    size_t n;    
    size_t n_spec, n_channels;

    floatArr spec = m_points->getPointSpectralChannelsArray(n_spec, n_channels);
       
    floatArr data2 =floatArr(new float[n_channels]);

    #pragma omp parallel for
    for (int chan=0;chan<n_channels;chan++)
    {
        data2[chan]=0;
       
        for (int i = 0; i < n_spec; i++)
        {
            int specIndex = n_channels * i+chan;
            data2[chan]+= spec[specIndex];           
        }                                       
        data2[chan]=data2[chan]/n_spec;
        //cout<<"channel "<<chan<<" : "<< data2[chan]<<endl;
    }
    m_plotter->setPoints(data2, n_channels);        
}

}