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
 * AnimationListItem.hpp
 *
 *  @date 20.12.2011
 *  @author Thomas Wiemann
 */

#ifndef ANIMATIONLISTITEM_HPP_
#define ANIMATIONLISTITEM_HPP_

#include <QtGui>
#include <QGLViewer/qglviewer.h>

enum
{
    PlayListItem = 1001
};


/**
 * @brief   A list item to represent a time for animations.
 */
class AnimationListItem : public QListWidgetItem
{
public:
    AnimationListItem(qglviewer::Frame* frame, QListWidget * parent = 0, float timeOffset = 1.0, int type = PlayListItem);
    virtual ~AnimationListItem();

    /**
     * @brief   Returns the current position in the timeline of the animation
     */
    float   time() { return m_time;}

    /**
     * @brief   Returns the duration of the current item
     */
    float   duration() { return m_duration;}

    /**
     * @brief   Sets the current position in the timeline.
     */
    void    setTime(float time);

    /**
     * @brief   Gets the time of the next item in the list. A negative
     *          return value indicates the end of the list.
     */
    float   getNextTime();

    /**
     * @brief   Gets the time of the previous item in the list. A negative
     *          return value indicated the beginning of the list.
     */
    float   getPrevTime();

    /**
     * @brief   Updates the times of the items after the current item
     */
    void    updateFollowingTimes(float offset);


    /**
     * @brief   Returns a pointer to the previous item
     */
    AnimationListItem*  getPrev();

    /**
     * @brief   Returns a pointer to the next item
     */
    AnimationListItem*  getNext();

    void setDuration(float d) { m_duration = d;}

    /**
     * @brief   UPdates the label in the list
     */
    void updateLabel();

    qglviewer::Frame frame() { return m_frame;}

private:
    /// The time index of this item
    float               m_time;

    /// The duration of this animation step
    float               m_duration;

    /// The list widget that contains all animation items
    QListWidget*        m_parent;

    qglviewer::Frame    m_frame;
};


#endif /* ANIMATIONLISTITEM_HPP_ */
