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

namespace lvr2
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

void LVRCorrespondanceDialog::clearAllItems()
{
    // For some reason we have to hack here. Code below
    // is not working properly
    while(m_ui->treeWidget->topLevelItemCount ())
    {
        m_ui->treeWidget->selectAll();
        deleteItem();
    }

    /*   NOT WORKING! Don't know why though...
    QTreeWidgetItemIterator it( m_ui->treeWidget );
    while(*it)
    {
        QTreeWidgetItem* item = *it;
        if(item->type() == LVRPickItemType)
        {
            int index = m_ui->treeWidget->indexOfTopLevelItem(item);
            LVRPickItem* i = static_cast<LVRPickItem*>(m_ui->treeWidget->takeTopLevelItem(index));
            if(i->getArrow())
            {
                Q_EMIT(removeArrow(i->getArrow()));
            }
            //if(i) delete i;
        }
        ++it;
    }
    */

    Q_EMIT(disableCorrespondenceSearch());
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
           //if(i) delete i;
        }
    }
}

void LVRCorrespondanceDialog::fillComboBoxes()
{
    // Clear contends
    m_ui->comboBoxModel->clear();
    m_ui->comboBoxData->clear();

    int index = 0;

    // Iterator over all items
    QTreeWidgetItemIterator it(m_treeWidget);
    while (*it)
    {
        if ( (*it)->type() == LVRPointCloudItemType)
        {
            QString text = (*it)->parent()->text(0);
            m_ui->comboBoxData->addItem(text);
            m_ui->comboBoxModel->addItem(text);
            
            if (index == 0)
            {
                m_ui->comboBoxModel->setCurrentText(text);
            }
            else if (index == 1)
            {
                m_ui->comboBoxData->setCurrentText(text);
            }
            index++;
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
    delete m_ui;
    delete m_dialog;
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

            // Handle arrow displayment: Check if an arrow exists,
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
        // Disable search to prevent change of last loaded correspondence
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

boost::optional<Transformf> LVRCorrespondanceDialog::getTransformation()
{
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> pairs;
    Eigen::Vector3f centroid_m = Eigen::Vector3f::Zero();
    Eigen::Vector3f centroid_d = Eigen::Vector3f::Zero();

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

                Vector3f start(s[0], s[1], s[2]);
                Vector3f end(e[0], e[1], e[2]);

                centroid_m += start.cast<float>();
                centroid_d += end.cast<float>();

                pairs.push_back(make_pair(start, end));
            }
        }
        ++it;

    }

    if(pairs.size() > 3)
    {
        centroid_m /= pairs.size();
        centroid_d /= pairs.size();

        Transformf matrix;
        EigenSVDPointAlign<float> align;
        align.alignPoints(pairs, centroid_m, centroid_d, matrix);

        return boost::make_optional(matrix);
    }
    else
    {
        cout << "Need at least 4 corresponding points" << endl;
        return boost::none;
    }
}

QString LVRCorrespondanceDialog::getModelName()
{
    return m_ui->comboBoxModel->currentText();
}

QString LVRCorrespondanceDialog::getDataName()
{
    return m_ui->comboBoxData->currentText();
}

} /* namespace lvr2 */


