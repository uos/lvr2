/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "LVRHistogram.hpp"

namespace lvr2
{

LVRHistogram::LVRHistogram(QWidget* parent, PointBufferPtr points)
    : QDialog(parent)
{
    m_histogram.setupUi(this);
    //Set Plotmode to create a bar chart
    m_histogram.plotter->setPlotMode(PlotMode::BAR);
    m_histogram.plotter->setXRange(*points->getIntAtomic("spectral_wavelength_min"), *points->getIntAtomic("spectral_wavelength_max"));

    size_t n;
    size_t n_spec;

    //Get Array with Spectraldata
    //floatArr spec = points->getPointSpectralChannelsArray(n_spec, m_numChannels);
    UCharChannelOptional spec = points->getUCharChannel("spectral_channels");
    m_numChannels = spec->width();
    n_spec = spec->numElements();

    //New Array for Average channelintensity   
    m_data = floatArr(new float[m_numChannels]);

    #pragma omp parallel for
    //calculate average intensity of all Points for all channels
    for (size_t channel = 0; channel < m_numChannels; channel++)
    {
        m_data[channel] = 0;
       
        for (size_t i = 0; i < n_spec; i++)
        {
            m_data[channel] += (*spec)[i][channel];
        }                                       
        m_data[channel] /= n_spec;
    }

    refresh();
    
    //Connect scale checkbox
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
    //show Dialog
    show();
    raise();
    activateWindow();
}

} // namespace lvr2
