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

#include "LVRRecordedFrameItem.hpp"
#include "LVRItemTypes.hpp"

namespace lvr
{

// Create a camera from the current frame
LVRRecordedFrameItem::LVRRecordedFrameItem(vtkSmartPointer<vtkCameraRepresentation> pathCamera, QString name) :
    m_name(name)
{
    // Setup item properties
    setText(m_name);

    m_recordedFrame = vtkSmartPointer<vtkCamera>::New();
    m_recordedFrame->DeepCopy(pathCamera->GetCamera());
}

// Create a camera without setting its properties
LVRRecordedFrameItem::LVRRecordedFrameItem(QString name) :
    m_name(name)
{
    // Setup item properties
    setText(m_name);

    m_recordedFrame = vtkSmartPointer<vtkCamera>::New();
}

LVRRecordedFrameItem::~LVRRecordedFrameItem()
{
    // TODO Auto-generated destructor stub
}

vtkSmartPointer<vtkCamera> LVRRecordedFrameItem::getFrame()
{
    return m_recordedFrame;
}

void LVRRecordedFrameItem::writeToStream(QTextStream &out)
{
    // Save the position, the focal point and the view up to the current text stream
    out << "C:" << m_name << ";";
    double* position = m_recordedFrame->GetPosition();
    out << position[0] << "," << position[1] << "," << position[2] << ";";
    double* focalPoint = m_recordedFrame->GetFocalPoint();
    out << focalPoint[0] << "," << focalPoint[1] << "," << focalPoint[2] << ";";
    double* viewUp = m_recordedFrame->GetViewUp();
    out << viewUp[0] << "," << viewUp[1] << "," << viewUp[2] << endl;
}

LVRRecordedFrameItem* LVRRecordedFrameItem::createFromStream(QTextStream &in)
{
    QString line = in.readLine();
    // TODO: Surround with try and catch to prevent errors
    // Very basic file validity checking
    if(!line.startsWith("C:"))
    {
        cout << "Couldn't read frame from file!" << endl;
        return NULL;
    }

    line.remove(0,2);
    QStringList parameters = line.trimmed().split(";");

    QString name = parameters[0];
    LVRRecordedFrameItem* recordedFrameItem = new LVRRecordedFrameItem(name);

    QStringList position = parameters[1].split(",");
    recordedFrameItem->getFrame()->SetPosition(position[0].toDouble(), position[1].toDouble(), position[2].toDouble());

    QStringList focalPoint = parameters[2].split(",");
    recordedFrameItem->getFrame()->SetFocalPoint(focalPoint[0].toDouble(), focalPoint[1].toDouble(), focalPoint[2].toDouble());

    QStringList viewUp = parameters[3].split(",");
    recordedFrameItem->getFrame()->SetViewUp(viewUp[0].toDouble(), viewUp[1].toDouble(), viewUp[2].toDouble());

    return recordedFrameItem;
}

} /* namespace lvr */
