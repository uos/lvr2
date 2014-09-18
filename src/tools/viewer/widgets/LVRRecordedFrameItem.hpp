/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

#ifndef LVRRECORDEDFRAMEITEM_H_
#define LVRRECORDEDFRAMEITEM_H_

#include <QString>
#include <QColor>
#include <QListWidgetItem>
#include <QTextStream>

#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkCameraRepresentation.h>

namespace lvr
{

class LVRRecordedFrameItem : public QListWidgetItem
{
public:
    LVRRecordedFrameItem(vtkSmartPointer<vtkCameraRepresentation> pathCamera, QString name = "");
    virtual ~LVRRecordedFrameItem();
    void writeToStream(QTextStream &stream);
    static LVRRecordedFrameItem* createFromStream(QTextStream &stream);

public Q_SLOTS:
    vtkSmartPointer<vtkCamera>	getFrame();

protected:
    LVRRecordedFrameItem(QString name = "");
	vtkSmartPointer<vtkCamera>  m_recordedFrame;
    QString                     m_name;
};

} /* namespace lvr */

#endif /* LVRRECORDEDFRAMEITEM_H_ */
