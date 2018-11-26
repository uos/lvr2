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

#include "LVRRecordedFrameItem.hpp"
#include "LVRItemTypes.hpp"

namespace lvr2
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

} /* namespace lvr2 */
