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

LVRRecordedFrameItem::LVRRecordedFrameItem(vtkSmartPointer<vtkCameraRepresentation> pathCamera, QString name) :
    m_name(name)
{
    // Setup item properties
    setText(m_name);

    m_recordedFrame = vtkSmartPointer<vtkCamera>::New();
    m_recordedFrame->DeepCopy(pathCamera->GetCamera());
}

LVRRecordedFrameItem::LVRRecordedFrameItem(QString name) :
    m_name(name)
{
    // Setup item properties
    setText(m_name);
}

LVRRecordedFrameItem::~LVRRecordedFrameItem()
{
    // TODO Auto-generated destructor stub
}

vtkSmartPointer<vtkCamera> LVRRecordedFrameItem::getFrame()
{
    return m_recordedFrame;
}

void LVRRecordedFrameItem::writeToStream(QTextStream &stream)
{

}

LVRRecordedFrameItem* createFromStream(QTextStream &stream)
{
    LVRRecordedFrameItem* recordedFrameItem = new LVRRecordedFrameItem("Test");
    return recordedFrameItem;
}

} /* namespace lvr */
