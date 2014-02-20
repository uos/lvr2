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
#include "LVRPointCloudItem.hpp"
#include "LVRPickItem.hpp"
#include "LVRItemTypes.hpp"

#include <QMessageBox>

namespace lvr
{

LVRCorrespondanceDialog::LVRCorrespondanceDialog(QTreeWidget* treeWidget) :
    m_treeWidget(treeWidget)
{
    m_dialog = new QDialog(treeWidget);
    m_ui = new Ui_CorrespondenceDialog;
    m_ui->setupUi(m_dialog);


    m_dataSelectionColor = QColor(0, 255, 255, 0);
    m_modelSelectionColor = QColor(255, 255, 0, 0);
    m_defaultColor = QColor(255, 255, 255, 0);

    fillComboBoxes();
    QObject::connect(m_ui->comboBoxModel, SIGNAL(currentIndexChanged(QString)), this, SLOT(updateModelSelection(QString)));
    QObject::connect(m_ui->comboBoxData, SIGNAL(currentIndexChanged(QString)), this, SLOT(updateDataSelection(QString)));
    QObject::connect(m_ui->buttonNew, SIGNAL(pressed()), this, SLOT(insertNewItem()));
    QObject::connect(m_ui->buttonDelete, SIGNAL(pressed()), this, SLOT(deleteItem()));
}

void LVRCorrespondanceDialog::insertNewItem()
{
    LVRPickItem* item = new LVRPickItem( m_ui->treeWidget);
    QList<QTreeWidgetItem*> items =  m_ui->treeWidget->selectedItems();

    // De-select all previously selected items
    QList<QTreeWidgetItem*>::iterator it;
    for(it = items.begin(); it != items.end(); it++)
    {
        m_ui->treeWidget->setItemSelected(*it,false);
    }

    m_ui->treeWidget->addTopLevelItem(item);
    m_ui->treeWidget->setItemSelected(item, true);
}

void LVRCorrespondanceDialog::deleteItem()
{
    QList<QTreeWidgetItem*> items =  m_ui->treeWidget->selectedItems();
    if(items.size())
    {
        QTreeWidgetItem* it = items.first();
        if(it->type() == LVRPickItemType)
        {
           int index = m_ui->treeWidget->indexOfTopLevelItem(it);
           QTreeWidgetItem* i = m_ui->treeWidget->takeTopLevelItem(index);
           if(i) delete i;
        }
    }
}

void LVRCorrespondanceDialog::fillComboBoxes()
{
    // Clear contends
    m_ui->comboBoxModel->clear();
    m_ui->comboBoxData->clear();

    // Iterator over all items
    QTreeWidgetItemIterator it(m_treeWidget);
    while (*it)
    {
        if ( (*it)->type() == LVRPointCloudItemType)
        {
            m_ui->comboBoxData->addItem((*it)->parent()->text(0));
            m_ui->comboBoxModel->addItem((*it)->parent()->text(0));
        }
        ++it;
    }
    Q_EMIT(render());
}

void LVRCorrespondanceDialog::updateModelSelection(QString str)
{
    m_modelSelection = str;
    QTreeWidgetItemIterator it(m_treeWidget);
    while (*it)
    {
        if ( (*it)->type() == LVRPointCloudItemType )
        {
            LVRPointCloudItem* item = static_cast<LVRPointCloudItem*>(*it);
            if(item->parent()->text(0) == str)
            {
                item->setSelectionColor(m_modelSelectionColor);
            }
            else if(item->parent()->text(0) != m_dataSelection)
            {
                item->setSelectionColor(m_defaultColor);
            }
        }
        ++it;
    }
    Q_EMIT(render());
}

void LVRCorrespondanceDialog::updateDataSelection(QString str)
{
    m_dataSelection = str;
    QTreeWidgetItemIterator it(m_treeWidget);
     while (*it)
     {
         if ( (*it)->type() == LVRPointCloudItemType )
         {
             LVRPointCloudItem* item = static_cast<LVRPointCloudItem*>(*it);
             if(item->parent()->text(0) == str)
             {
                 item->setSelectionColor(m_dataSelectionColor);
             }
             else if(item->parent()->text(0) != m_modelSelection)
             {
                 item->setSelectionColor(m_defaultColor);
             }
         }
         ++it;
     }
     Q_EMIT(render());
}

LVRCorrespondanceDialog::~LVRCorrespondanceDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRCorrespondanceDialog::firstPointPicked(double* pos)
{
    QList<QTreeWidgetItem*> items =  m_ui->treeWidget->selectedItems();
    if(items.size())
    {
        QTreeWidgetItem* it = items.first();
        if(it->type() == LVRPickItemType)
        {
            LVRPickItem* item = static_cast<LVRPickItem*>(it);
            item->setStart(pos);
        }
    }
    else
    {
        /*QMessageBox msgBox;
        msgBox.setText("No item to edit selected.");
        msgBox.setInformativeText("Please select one or add a new item to the list. Press 'q' to quit pick mode.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();*/
    }
}

void LVRCorrespondanceDialog::secondPointPicked(double* pos)
{
    QList<QTreeWidgetItem*> items =  m_ui->treeWidget->selectedItems();
    if(items.size())
    {
        QTreeWidgetItem* it = items.first();
        if(it->type() == LVRPickItemType)
        {
            LVRPickItem* item = static_cast<LVRPickItem*>(it);
            item->setEnd(pos);
        }
    }
    else
    {
        /*
        QMessageBox msgBox;
        msgBox.setText("No item to edit selected.");
        msgBox.setInformativeText("Please select one or add a new item to the list. Press 'q' to quit pick mode.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();*/
    }
}

} /* namespace lvr */
