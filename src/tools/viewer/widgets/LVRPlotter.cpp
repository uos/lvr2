
#include "LVRPlotter.hpp"
#include <QPainter>

namespace lvr
{

LVRPlotter::LVRPlotter(QWidget * parent, bool curve)
	: QWidget(parent), m_numPoints(0)
{
	m_points.reset();
	m_curve = curve;
}

LVRPlotter::~LVRPlotter()
{

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

void LVRPlotter::paintEvent(QPaintEvent *)
{
	if (!m_numPoints)
	{
		return;
	}

	QPainter painter(this);

	painter.setPen(QColor(0, 0, 0));
	painter.setBrush(QColor(0, 0, 0));

	QRect rect;
	painter.drawText(0, 0, width(), height(), Qt::AlignTop, QString("%1").arg(m_max), &rect);
	int leftMargin = rect.width() + 1;

	painter.drawText(0, 0, width(), height(), Qt::AlignBottom, QString("%1").arg(m_min), &rect);
	leftMargin = rect.width() + 1 > leftMargin ? rect.width() + 1 : leftMargin;

	painter.drawLine(leftMargin, 0, leftMargin, height());

	float drawWidth = width() - leftMargin;

	int count = width() / 70;
	for (int i = 0; i < count; i++)
	{
		float new_x = i * drawWidth / count + leftMargin;
		painter.drawText(new_x, height() - 20, width(), height(), Qt::AlignTop, QString("%1").arg(i * (1000 - 400) / count + 400), &rect);
	}
	
	int botMargin = rect.height() + 1;
	float drawHeight = height() - botMargin;

	painter.drawLine(0, drawHeight, width(), drawHeight);

	painter.setPen(QColor(255, 0, 0));
	painter.setBrush(QColor(255, 0, 0));

	float old_x = leftMargin;
	float old_y = (m_points[0] - m_min) / (m_max - m_min) * drawHeight;

	for (int i = 1; i < m_numPoints; i++)
	{
		float new_x = i * drawWidth / m_numPoints + leftMargin;
		float new_y = (m_points[i] - m_min) / (m_max - m_min) * drawHeight;
		
		if(m_curve)
		{
			painter.drawLine(old_x, drawHeight - old_y, new_x, drawHeight - new_y);
		}
		else
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

}