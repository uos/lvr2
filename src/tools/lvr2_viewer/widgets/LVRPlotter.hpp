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

#ifndef LVRPLOTTER_HPP_
#define LVRPLOTTER_HPP_

#include <QWidget>

#include "lvr2/io/DataStruct.hpp"

namespace lvr2
{

enum class PlotMode
{
	LINE,
	BAR
};

class LVRPlotter : public QWidget
{
	Q_OBJECT

Q_SIGNALS:
    void mouseRelease();

public:

    LVRPlotter(QWidget* parent = (QWidget*)nullptr);
    virtual ~LVRPlotter();

	void setPlotMode(PlotMode mode);
	void setXRange(int min, int max);
    void setPoints(floatArr points, size_t numPoints);
    void setPoints(floatArr points, size_t numPoints, float min, float max);
	void removePoints();

protected:
	virtual void mouseReleaseEvent(QMouseEvent* event); 
    ///Create axes, labeling and draw graph or bar chart
	void paintEvent(QPaintEvent *event);

private:
    floatArr m_points;
	size_t   m_numPoints;
	float    m_min;
	float    m_max;
	PlotMode m_mode;
	int		 m_minX;
	int		 m_maxX;
};

} /* namespace lvr2 */

#endif /* LVRPLOTTER_HPP_ */
