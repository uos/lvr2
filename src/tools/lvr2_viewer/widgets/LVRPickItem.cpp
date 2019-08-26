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

/**
 * LVRPickItem.cpp
 *
 *  @date Feb 20, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPickItem.hpp"

#include "lvr2/geometry/BaseVector.hpp"

namespace lvr2
{

LVRPickItem::LVRPickItem(QTreeWidget* parent, int type) :
        QTreeWidgetItem(parent, type)
{
    m_start     = 0;
    m_end       = 0;
    m_arrow     = 0;
    setText(0, "Empty");
    setText(1, "Empty");
}

LVRPickItem::~LVRPickItem()
{
   if(m_start)  delete[] m_start;
   if(m_end)    delete[] m_end;
}

void LVRPickItem::setStart(double* start)
{
    using Vec = BaseVector<float>;
    if (m_start) delete[] m_start;
    m_start = start;
    QString x, y, z, text;
    x.setNum(start[0], 'f');
    y.setNum(start[1], 'f');
    z.setNum(start[2], 'f');
    text = QString("%1 %2 %3").arg(x).arg(y).arg(z);
    setText(0, text);

    // Create new arrow if necessary
    if(m_start && m_end)
    {
        Vec start(m_start[0], m_start[1], m_start[2]);
        Vec end(m_end[0], m_end[1], m_end[2]);
        m_arrow = new LVRVtkArrow(start, end);
    }
}

double* LVRPickItem::getStart()
{
    return m_start;
}

double* LVRPickItem::getEnd()
{
    return m_end;
}

void LVRPickItem::setEnd(double* end)
{
    if(m_end)    delete[] m_end;
    m_end = end;
    QString x, y, z, text;
    x.setNum(end[0], 'f');
    y.setNum(end[1], 'f');
    z.setNum(end[2], 'f');
    text = QString("%1 %2 %3").arg(x).arg(y).arg(z);
    setText(1, text);

    // Create new arrow if necessary
    if(m_start && m_end)
    {
        Vec start(m_start[0], m_start[1], m_start[2]);
        Vec end(m_end[0], m_end[1], m_end[2]);
        m_arrow = new LVRVtkArrow(start, end);
    }
}

LVRVtkArrow* LVRPickItem::getArrow()
{
    return m_arrow;
}

} /* namespace lvr2 */
