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

#include "LVRPlotter.hpp"
#include "LVRPointInfo.hpp"
#include <QPainter>
//#include <../app/LVRMainWindow.hpp"

namespace lvr2
{

LVRPlotter::LVRPlotter(QWidget * parent)
	: QWidget(parent), m_mode(PlotMode::LINE), m_numPoints(0), m_minX(0), m_maxX(1000)
{
	m_points.reset();
}

LVRPlotter::~LVRPlotter()
{

}

void LVRPlotter::mouseReleaseEvent(QMouseEvent* event)
{
	Q_EMIT(mouseRelease());
}

void LVRPlotter::setPlotMode(PlotMode mode)
{
	m_mode = mode;
}

void LVRPlotter::setPoints(floatArr points, size_t numPoints)
{
	float max = points[0], min = points[0];
	for (int i = 0; i < numPoints; i++)
	{
		if (points[i] > max)
			max = points[i];
		if (points[i] < min)
			min = points[i];
	}
	setPoints(points, numPoints, min, max);
}

void LVRPlotter::setPoints(floatArr points, size_t numPoints, float min, float max)
{
	m_points = points;
	m_numPoints = numPoints;
	m_max = max;
	m_min = min;
	update();
}

void LVRPlotter::removePoints()
{
	m_points.reset();
	m_numPoints = 0;
	update();
}

void LVRPlotter::setXRange(int min, int max)
{
	m_minX = min;
	m_maxX = max;
}

void LVRPlotter::paintEvent(QPaintEvent *)
{
	if (!m_numPoints)
	{
		return;
	}
	//Create axes and labeling for the plotter
	QPainter painter(this);

	painter.setPen(QColor(255, 255, 255));
	painter.setBrush(QColor(255, 255, 255));

	painter.drawRect(0, 0, width(), height());

	painter.setPen(QColor(0, 0, 0));
	painter.setBrush(QColor(0, 0, 0));

	QRect rect;
	painter.drawText(0, 0, width(), height(), Qt::AlignTop, QString("%1").arg(m_max), &rect);
	int leftMargin = rect.width() + 1;

	painter.drawText(0, 0, width(), height(), Qt::AlignBottom, QString("%1").arg(m_min), &rect);
	leftMargin = rect.width() + 1 > leftMargin ? rect.width() + 1 : leftMargin;

	painter.drawLine(leftMargin, 0, leftMargin, height());

	float drawWidth = width() - leftMargin;

	//add wavelength to axes
	int count = width() / 70;
	for (int i = 0; i < count; i++)
	{
		float new_x = i * drawWidth / count + leftMargin;
		painter.drawText(new_x, height() - 20, width(), height(), Qt::AlignTop, QString("%1").arg(i * (m_maxX - m_minX) / count + m_minX), &rect);
	}
	
	int botMargin = rect.height() + 1;
	float drawHeight = height() - botMargin;

	painter.drawLine(0, drawHeight, width(), drawHeight);

	painter.setPen(QColor(255, 0, 0));
	painter.setBrush(QColor(255, 0, 0));

	float old_x = leftMargin;
	float old_y = (m_points[0] - m_min) / (m_max - m_min) * drawHeight;

	//Plot information
	for (int i = 1; i < m_numPoints; i++)
	{
		float new_x = i * drawWidth / m_numPoints + leftMargin;
		float new_y = (m_points[i] - m_min) / (m_max - m_min) * drawHeight;
		//Mode to draw Line for Point Preview and Pointview
		if(m_mode == PlotMode::LINE)
		{
			painter.drawLine(old_x, drawHeight - old_y, new_x, drawHeight - new_y);
		}
		//Mode to draw bar chart for histogram
		else if (m_mode == PlotMode::BAR)
		{
			painter.setPen(QColor(0, 0, 255));
			while(old_x <= new_x)
			{
				painter.drawLine(old_x, drawHeight - old_y, old_x, drawHeight);
				old_x++;
			}	
		}
		old_x = new_x;
		old_y = new_y;
	}

}

} // namespace lvr2
