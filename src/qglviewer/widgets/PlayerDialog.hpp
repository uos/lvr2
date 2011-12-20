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
 * PlayerDialog.h
 *
 *  @date 19.12.2011
 *  @author Thomas Wiemann
 */

#ifndef PLAYERDIALOG_H_
#define PLAYERDIALOG_H_

#include "PlayerDialogUI.h"

#include "AnimationListItem.hpp"
#include "../viewers/Viewer.h"

using Ui::PlayerDialogUI;

namespace lssr
{

class PlayerDialog : public QObject
{
    Q_OBJECT

public:
    PlayerDialog(Viewer* parent = 0);
    virtual ~PlayerDialog() {};

public Q_SLOTS:
    void addItem();
    void removeItem();
    void updateTimes(double d);
    void selectNext();
    void selectPrev();
    void selectFirst();
    void selectLast();
    void play();
    void updateSelectedItem(QListWidgetItem* item);

public:
    AnimationListItem* item() { return m_item;}
private:

    void connectEvents();

    AnimationListItem*      m_item;
    Viewer*                 m_parent;
    PlayerDialogUI*         m_ui;
};

} /* namespace lssr */

#endif /* PLAYERDIALOG_H_ */
