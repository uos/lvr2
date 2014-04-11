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
#include <QFont>
#include <QFileDialog>

#include <fstream>
using std::ifstream;
using std::ofstream;

namespace lvr
{

LVRCorrespondanceDialog::LVRCorrespondanceDialog(QTreeWidget* treeWidget) :
    m_treeWidget(treeWidget)
{
    m_dialog = new QDialog(treeWidget);
    m_ui = new Ui_CorrespondenceDialog;
    m_ui->setupUi(m_dialog);


    m_dataSelectionColor = QColor(0, 0, 255, 0);      // Blue
    m_modelSelectionColor = QColor(255, 255, 0, 0);     // Yellow
    m_defaultColor = QColor(255, 255, 255, 0);

    fillComboBoxes();

    m_ui->comboBoxModel->setAutoFillBackground( true );
    m_ui->comboBoxModel->setStyleSheet("QComboBox { background-color: blue; } QComboBox QAbstractItemView {border: 2px solid darkgray; selection-background-color: lightgray;}");



    m_ui->comboBoxData->setAutoFillBackground( true );
    m_ui->comboBoxData->setStyleSheet("QComboBox { background-color: yellow; }");

    QObject::connect(m_ui->comboBoxModel, SIGNAL(activated(int)), this, SLOT(updateModelSelection(int)));
    QObject::connect(m_ui->comboBoxData, SIGNAL(activated(int)), this, SLOT(updateDataSelection(int)));
    QObject::connect(m_ui->buttonNew, SIGNAL(pressed()), this, SLOT(insertNewItem()));
    QObject::connect(m_ui->buttonDelete, SIGNAL(pressed()), this, SLOT(deleteItem()));
    QObject::connect(m_ui->buttonLoad, SIGNAL(pressed()), this, SLOT(loadCorrespondences()));
    QObject::connect(m_ui->buttonSave, SIGNAL(pressed()), this, SLOT(saveCorrespondences()));
    QObject::connect(m_ui->buttonSwap, SIGNAL(pressed()), this, SLOT(swapItemPositions()));
    QObject::connect(m_ui->buttonDelete, SIGNAL(pressed()), this, SLOT(deleteItem()));
    QObject::connect(m_ui->treeWidget, SIGNAL(currentItemChanged (QTreeWidgetItem*, QTreeWidgetItem*)), this, SLOT(treeItemSelected(QTreeWidgetItem*, QTreeWidgetItem*)));
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
    m_ui->treeWidget->setCurrentItem(item);

    Q_EMIT(enableCorrespondenceSearch());
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
           LVRPickItem* i = static_cast<LVRPickItem*>(m_ui->treeWidget->takeTopLevelItem(index));
           if(i->getArrow())
           {
               Q_EMIT(removeArrow(i->getArrow()));
           }
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

void LVRCorrespondanceDialog::updateModelSelection(int index)
{
    QString str = m_ui->comboBoxModel->currentText();
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
           /* else if(item->parent()->text(0) != m_dataSelection)
            {
                item->setSelectionColor(m_defaultColor);
            }*/
        }
        ++it;
    }
    Q_EMIT(render());
}

void LVRCorrespondanceDialog::updateDataSelection(int index)
{
    QString str = m_ui->comboBoxData->currentText();
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
       /*      else if(item->parent()->text(0) != m_modelSelection)
             {
                 item->setSelectionColor(m_defaultColor);
             }*/
         }
         ++it;
     }
     Q_EMIT(render());
}

bool LVRCorrespondanceDialog::doICP()
{
    return m_ui->checkBoxICP->isChecked();
}

double LVRCorrespondanceDialog::getEpsilon()
{
    return m_ui->spinBoxEpsilon->value();
}

double LVRCorrespondanceDialog::getMaxDistance()
{
    return m_ui->spinBoxDistance->value();
}

int LVRCorrespondanceDialog::getMaxIterations()
{
    return m_ui->spinBoxIterations->value();
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

            // Handle arrow displayment: Check if an arrow existed,
            // remove it from renderer and create a new one (and pray
            // that there are no concurrency problems...
            LVRVtkArrow* arrow = item->getArrow();
            if(arrow)
            {
                Q_EMIT(removeArrow(arrow));
            }
            item->setStart(pos);
            arrow = item->getArrow();
            if(arrow)
            {
                Q_EMIT(addArrow(arrow));
            }


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

            // Handle arrow displayment: Check if an arrow existed,
            // remove it from renderer and create a new one (and pray
            // that there are no concurrency problems...
            LVRVtkArrow* arrow = item->getArrow();
            if(arrow)
            {
                Q_EMIT(removeArrow(arrow));
            }
            item->setEnd(pos);
            arrow = item->getArrow();
            if(arrow)
            {
                Q_EMIT(addArrow(arrow));
            }
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

void LVRCorrespondanceDialog::swapItemPositions()
{
    QList<QTreeWidgetItem*> items =  m_ui->treeWidget->selectedItems();
    if(items.size())
    {
        QTreeWidgetItem* it = items.first();
        if(it->type() == LVRPickItemType)
        {
            LVRPickItem* item = static_cast<LVRPickItem*>(it);
            double* end = item->getEnd();
            double* start = item->getStart();
            item->setStart(end);
            item->setEnd(start);
        }
    }
}

void LVRCorrespondanceDialog::saveCorrespondences()
{
    QString fileName = QFileDialog::getSaveFileName(m_treeWidget,
            tr("Save Correspondences"), "./", tr("Correspondence Files (*.cor)"));

    if(fileName != "")
    {
        ofstream outfile(fileName.toStdString().c_str());
        QTreeWidgetItemIterator it(m_ui->treeWidget);
        while(*it)
        {
            if( (*it)->type() == LVRPickItemType)
            {
                LVRPickItem* item = static_cast<LVRPickItem*>(*it);
                double* start = item->getStart();
                double* end = item->getEnd();
                outfile << start[0] << " " << start[1] << " " << start[2] << " ";
                outfile << end[0] << " " << end[1] << " " << end[2] << endl;
                cout << start << " " << end << endl;
            }
            ++it;
        }
        outfile.close();
    }
}

void LVRCorrespondanceDialog::loadCorrespondences()
{
    QString fileName = QFileDialog::getOpenFileName(m_treeWidget,
            tr("Load Correspondences"), "./", tr("Correspondence Files (*.cor)"));

    if(fileName != "")
    {
        ifstream infile(fileName.toStdString().c_str());
        while(infile.good())
        {
            double* start = new double[3];
            double* end   = new double[3];
            infile >> start[0] >> start[1] >> start[2];
            infile >> end[0] >> end[1] >> end[2];
            // Check if we reached Eof with last read
            if(infile.good())
            {
                LVRPickItem* item = new LVRPickItem( m_ui->treeWidget);
                item->setStart(start);
                item->setEnd(end);

                LVRVtkArrow* arrow = item->getArrow();
                if(arrow)
                {
                    Q_EMIT(addArrow(arrow));
                }

                m_ui->treeWidget->addTopLevelItem(item);
                m_ui->treeWidget->setItemSelected(item, false);
                m_ui->treeWidget->setCurrentItem(item);
            }
        }
        // Disable search to prevent change of last loaded correspondes
        // when the user clicks into the window
        Q_EMIT(disableCorrespondenceSearch());
    }
}

void LVRCorrespondanceDialog::treeItemSelected(QTreeWidgetItem* current, QTreeWidgetItem* prev)
{
    // Enable picking
    Q_EMIT(enableCorrespondenceSearch());

    // Set color of current item to red
    if(current)
    {
        if(current->type() == LVRPickItemType)
        {
            LVRPickItem* item = static_cast<LVRPickItem*>(current);
            if(item->getArrow())
            {
                item->getArrow()->setTmpColor(1.0, 0.2, 0.2);
            }
        }
    }

    // Reset previous item to default color
    if(prev)
    {
        if(prev->type() == LVRPickItemType)
        {
            LVRPickItem* item = static_cast<LVRPickItem*>(prev);
            if(item->getArrow())
            {
                item->getArrow()->restoreColor();
            }
        }
    }
    Q_EMIT(render());
}


Matrix4f LVRCorrespondanceDialog::getTransformation()
{
    PointPairVector pairs;
    Vertexf centroid1;
    Vertexf centroid2;
    Matrix4f matrix;

    QTreeWidgetItemIterator it(m_ui->treeWidget);
    while (*it)
    {
        if( (*it)->type() == LVRPickItemType)
        {
            LVRPickItem* item = static_cast<LVRPickItem*>(*it);
            if(item->getStart() && item->getEnd())
            {
                double* s = item->getStart();
                double* e = item->getEnd();

                // Convert to left handed coordinates!
                Vertexf start(s[0], s[1], s[2]);
                Vertexf end(e[0], e[1], e[2]);

                centroid1 += start;
                centroid2 += end;

                pairs.push_back(std::pair<Vertexf, Vertexf>(start, end));
            }
        }
        ++it;
    }

    if(pairs.size() > 3)
    {
        centroid1 /= pairs.size();
        centroid2 /= pairs.size();

        EigenSVDPointAlign align;
        align.alignPoints(pairs, centroid1, centroid2, matrix);
    }
    else
    {
        cout << "Need at least 4 corresponding points" << endl;
    }
    return matrix;
}

QString LVRCorrespondanceDialog::getModelName()
{
    return m_ui->comboBoxModel->currentText();
}

QString LVRCorrespondanceDialog::getDataName()
{
    return m_ui->comboBoxData->currentText();
}

} /* namespace lvr */


