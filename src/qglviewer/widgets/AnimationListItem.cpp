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

/**
 * AnimationListItem.cpp
 *
 *  @date 20.12.2011
 *  @author Thomas Wiemann
 */
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
using namespace std;

#include "AnimationListItem.hpp"

AnimationListItem::AnimationListItem(qglviewer::Frame* frame, QListWidget * parent, float timeOffset, int type)
    : QListWidgetItem(parent, type), m_parent(parent)
{
    // Copy frame data
    m_frame = qglviewer::Frame(*frame);

    m_duration = timeOffset;

    AnimationListItem* prev = getPrev();
    AnimationListItem* next = getNext();

    if(prev)
    {
        m_time = prev->time() + prev->duration();
    }
    else
    {
        m_time = 0;
    }

    // Update other list items
    updateLabel();
    updateFollowingTimes(m_duration);

}

void AnimationListItem::updateLabel()
{
    // Create new label string and update label
     stringstream ss;
     ss << fixed << setprecision(2) << "[" << m_time << " @ " << m_duration << "]";
     string label = ss.str();
     setText(QString(label.c_str()));
}



float AnimationListItem::getNextTime()
{
    AnimationListItem* item = getNext();
    if(item)
    {
        return item->time();
    }
    else
    {
        return -1.0;
    }
}


float AnimationListItem::getPrevTime()
{
    AnimationListItem* item = getPrev();
   if(item)
   {
       return item->time();
   }
   else
   {
       return -1.0;
   }
}

AnimationListItem*  AnimationListItem::getNext()
{
    int nextPosition = m_parent->row(this) + 1;

    // Get time of next item
    QListWidgetItem* next = m_parent->item(nextPosition);

    // Check type and cast
    if(next)
    {
        if(next->type() == PlayListItem)
        {
            return static_cast<AnimationListItem*>(next);
        }
    }

    // Return 0 if something went wrong
    return 0;
}


AnimationListItem*  AnimationListItem::getPrev()
{
    int prevPosition = m_parent->row(this) - 1;

    if(prevPosition >= 0)
    {
        QListWidgetItem* next = m_parent->item(prevPosition);
        if(next)
        {
            return static_cast<AnimationListItem*>(next);
        }
    }

    return 0;
}


void AnimationListItem::updateFollowingTimes(float offset)
{
    // Update next position
    AnimationListItem* item = getNext();
    if(item)
    {
        item->m_time += offset;

        // Continue updating
        item->updateLabel();
        item->updateFollowingTimes(offset);
    }

}

void AnimationListItem::setTime(float time)
{
    // Check if given time is between previous and last time
    if( (time > getPrevTime()) && (time < getNextTime()) )
    {
        m_time = time;
        updateLabel();
    }
}



AnimationListItem::~AnimationListItem()
{
    // TODO Auto-generated destructor stub
}

