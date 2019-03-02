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

#include "LVRPointInfo.hpp"

namespace lvr2
{

LVRPointInfo::LVRPointInfo(QWidget* parent, PointBufferPtr pointBuffer, int pointId) :
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
    
    UCharChannelOptional spec_channels = pointBuffer->getUCharChannel("spectral_channels");
    size_t n_spec = spec_channels->numElements();
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
