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
 * LVRCorrespondanceDialog.cpp
 *
 *  @date Feb 18, 2014
 *  @author Thomas Wiemann
 */
#include "LVRCorrespondanceDialog.hpp"
#include "LVRItemTypes.hpp"


namespace lvr
{

LVRCorrespondanceDialog::LVRCorrespondanceDialog(QTreeWidget* parent) :
    QDialog(parent), m_treeWidget(parent)
{
    setupUi(this);
    fillComboBoxes();
}

void LVRCorrespondanceDialog::fillComboBoxes()
{
    // Clear contends
    comboBoxModel->clear();
    comboBoxData->clear();

    // Iterator over all items
    QTreeWidgetItemIterator it(m_treeWidget);
    while (*it)
    {
        if ( (*it)->type() == LVRPointCloudItemType)
        {
            comboBoxData->addItem((*it)->parent()->text(0));
            comboBoxModel->addItem((*it)->parent()->text(0));
        }
        ++it;
    }
}

LVRCorrespondanceDialog::~LVRCorrespondanceDialog()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
