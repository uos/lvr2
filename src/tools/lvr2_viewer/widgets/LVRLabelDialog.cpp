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
#include "LVRLabelDialog.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRItemTypes.hpp"

#include <vtkSmartPointer.h>
#include <vtkCubeSource.h>

#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>

#include <QMessageBox>
#include <QFont>
#include <QFileDialog>
#include <QInputDialog>
#include <QColorDialog>
#include <QButtonGroup>
#include <vtkSelectEnclosedPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkPointData.h>
#include <vector>
#include <algorithm>

#include <vtkLookupTable.h>
#include <vtkExtractGeometry.h>
#include "lvr2/io/descriptions/HDF5Kernel.hpp"

#include <fstream>
using std::ifstream;
using std::ofstream;

namespace lvr2
{

LVRLabelDialog::LVRLabelDialog(QTreeWidget* treeWidget) :
    m_treeWidget(treeWidget)
{
    m_dialog = new QDialog(treeWidget);
    m_ui = new Ui_LabelDialog;
    m_ui->setupUi(m_dialog);
   // m_ui->treeWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    QObject::connect(m_ui->newLabelButton, SIGNAL(pressed()), this, SLOT(addNewLabel()));
    QObject::connect(m_ui->newInstanceButton, SIGNAL(pressed()), this, SLOT(addNewInstance()));
    QObject::connect(m_ui->treeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(cellSelected(QTreeWidgetItem*, int)));
    QObject::connect(m_ui->selectedLabelComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(comboBoxIndexChanged(int)));
    QObject::connect(m_ui->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(visibilityChanged(QTreeWidgetItem*, int)));
//    QObject::connect(m_ui->labelSelectedPoints, SIGNAL(pressed()), this, SLOT(labelPoints()));
}

LVRLabelDialog::~LVRLabelDialog()
{
    delete m_ui;
    delete m_dialog;
    // TODO Auto-generated destructor stub
}

void LVRLabelDialog::visibilityChanged(QTreeWidgetItem* changedItem, int column)
{
    if(column != LABEL_VISIBLE_COLUMN)
    {
        return;
    }

    //check if Instance or hole label changed
    if (changedItem->parent())
    {
        //parent exists item is an instance
        Q_EMIT(hidePoints(changedItem->data(LABEL_ID_COLUMN,0).toInt(), changedItem->checkState(LABEL_VISIBLE_COLUMN)));
    } else
    {
        for (int i = 0; i < changedItem->childCount(); i++)
        {
            QTreeWidgetItem* childItem = changedItem->child(i);

            //sets child elements checkbox on toplevel box value if valuechanged a singal will be emitted and handeled
            childItem->setCheckState(LABEL_VISIBLE_COLUMN, changedItem->checkState(LABEL_VISIBLE_COLUMN));
        }
    }


}

void LVRLabelDialog::cellSelected(QTreeWidgetItem* item, int column)
{
    if(column == LABEL_NAME_COLUMN)
    {
        //Edit Label name
        bool accepted;
        QString label_name = QInputDialog::getText(m_dialog, tr("Select Label Name"),
            tr("Label name:"), QLineEdit::Normal,
            item->text(LABEL_NAME_COLUMN), &accepted);
        if (accepted && !label_name.isEmpty())
        {
            item->setText(LABEL_NAME_COLUMN, label_name);
            if (!item->parent())
            {
                //Toplevel item nothing else to do
                return;
            }
            int comboBoxPos = m_ui->selectedLabelComboBox->findData(item->data(LABEL_ID_COLUMN, 0).toInt());
            if (comboBoxPos >= 0)
            {
                m_ui->selectedLabelComboBox->setItemText(comboBoxPos, label_name);

            }
            return;
        }
    }else if(column == LABEL_ID_COLUMN && item->parent())
    {
        //Change 
        QColor label_color = QColorDialog::getColor(Qt::red, m_dialog, tr("Choose Label Color"));
        if (label_color.isValid())
        {
            item->setData(LABEL_ID_COLUMN, 1, label_color);

            //Update Color In picker
            Q_EMIT(labelAdded(item));
            return;
            }
    }
}

void LVRLabelDialog::updatePointCount(int selectedPointCount)
{

    int topItemCount = m_ui->treeWidget->topLevelItemCount();
    for (int i = 0; i < topItemCount; i++)
    {
        QTreeWidgetItem* topLevelItem = m_ui->treeWidget->topLevelItem(i);
        int childCount = topLevelItem->childCount();
        for (int j = 0; j < childCount; j++)
        {
            if(m_ui->selectedLabelComboBox->currentData().toInt() == topLevelItem->child(j)->data(LABEL_ID_COLUMN, 0).toInt())
            {       
                int pointCountDifference = selectedPointCount - topLevelItem->child(j)->text(LABELED_POINT_COLUMN).toInt();
                topLevelItem->child(j)->setText(LABELED_POINT_COLUMN, QString::number(selectedPointCount));
                //Add points to toplevel points
                topLevelItem->setText(LABELED_POINT_COLUMN, QString::number(pointCountDifference + topLevelItem->text(LABELED_POINT_COLUMN).toInt()));
                return;
            }
        }
    }
}

void LVRLabelDialog::responseLabels(std::vector<uint16_t> labeledPoints)
{
    std::map<uint16_t,std::vector<int>> idMap;

    for (int i = 0; i < labeledPoints.size(); i++)
    {
        
        if(idMap.find(labeledPoints[i]) == idMap.end())
        {
            //first occurence of id add new entry
            idMap[labeledPoints[i]] = {};
        }
        idMap[labeledPoints[i]].push_back(i);
    }
    
    QFileDialog dialog;
    dialog.setFileMode(QFileDialog::AnyFile);
    QString strFile = dialog.getSaveFileName(m_dialog, "Creat New HDF5 File","","");
    HDF5Kernel label_hdf5kernel((strFile + QString(".h5")).toStdString());
    std::cout << strFile.toStdString() << std::endl;
    int topItemCount = m_ui->treeWidget->topLevelItemCount();
    for (int i = 0; i < topItemCount; i++)
    {
        QTreeWidgetItem* topLevelItem = m_ui->treeWidget->topLevelItem(i);
        std::string topLabel = topLevelItem->text(LABEL_NAME_COLUMN).toStdString();
        int childCount = topLevelItem->childCount();
        for (int j = 0; j < childCount; j++)
        {
            int childID = topLevelItem->child(j)->data(LABEL_ID_COLUMN, 0).toInt();
            int* sharedArrayData = new int[idMap[childID].size()];
            std::memcpy(sharedArrayData, idMap[childID].data(), idMap[childID].size() * sizeof(int));
            boost::shared_array<int> data(sharedArrayData);
            std::vector<size_t> dimension = {idMap[childID].size()};
            if(idMap.find(childID) != idMap.end())
            { 
                label_hdf5kernel.saveArray(topLabel, (topLevelItem->child(j)->text(LABEL_NAME_COLUMN)).toStdString(), dimension, data);
            }
                
        }
    }
    std::vector<std::string> subGroup= {"Subgroup1", "Subgroup2", "Subgroup3"};
    boost::shared_array<int> data(new int[5]);
    std::vector<size_t> dimension = {5};
    label_hdf5kernel.subGroupNames("TopGroup", subGroup);
    label_hdf5kernel.saveArray("Subgroup1", "more data", dimension, data);
    label_hdf5kernel.saveArray("Subgroup1", "other data", dimension, data);
    label_hdf5kernel.saveArray("other data", "others data", dimension, data);

    //add unlabeled Points
    if (idMap.find(0) != idMap.end())
    {
        int* sharedArrayData = new int[idMap[0].size()];
        std::memcpy(sharedArrayData, idMap[0].data(), idMap[0].size() * sizeof(int));
        boost::shared_array<int> data(sharedArrayData);
        std::vector<size_t> dimension = {idMap[0].size()};
        if(idMap.find(0) != idMap.end())
        { 
            label_hdf5kernel.saveArray("Labels", "Unlabeled", dimension, data);
        }
    }
}


void LVRLabelDialog::addNewLabel()
{
    //Ask For the Label name 
    bool accepted;
    QString label_name = QInputDialog::getText(m_dialog, tr("Select Label Name"),
    tr("Label name:"), QLineEdit::Normal,
                    tr("LabelName") , &accepted);
    if (!accepted || label_name.isEmpty())
    {
            //No valid Input
            return;
    }

    QColor label_color = QColorDialog::getColor(Qt::red, m_dialog, tr("Choose Label Color for first instance"));
    if (!label_color.isValid())
    {
            //Non Valid Color Return 
            return;
    }

    if (m_ui->treeWidget->topLevelItemCount() == 0)
    {
        //Added first Top Level item enable instance button
        m_ui->newInstanceButton->setEnabled(true);
    }

    int id = m_id_hack++;
    //Setting up new Toplevel item
    QTreeWidgetItem * item = new QTreeWidgetItem();
    item->setText(0, label_name);
    item->setText(LABELED_POINT_COLUMN, QString::number(0));
    item->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);

    //Setting up new child item
    QTreeWidgetItem * childItem = new QTreeWidgetItem();
    childItem->setText(LABEL_NAME_COLUMN, label_name + QString::number(1));
    childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
    childItem->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    childItem->setData(LABEL_ID_COLUMN, 1, label_color);
    childItem->setData(LABEL_ID_COLUMN, 0, id);
    item->addChild(childItem);
    m_ui->treeWidget->addTopLevelItem(item);    
    
    //TODO generate a gloabal id field that isnt bound to table position 
    //TODO Think about a better way than that hacky data color solution

    //Add label to combo box 
    m_ui->selectedLabelComboBox->addItem(childItem->text(LABEL_NAME_COLUMN), id);

    Q_EMIT(labelAdded(childItem));
}

void LVRLabelDialog::addNewInstance()
{
    QInputDialog topLevelDialog;
    QStringList topLabels;
    if (m_ui->treeWidget->topLevelItemCount() == 0)
    {
        return;
    }

    for (int i = 0; i < m_ui->treeWidget->topLevelItemCount(); i++)
    {
        topLabels << m_ui->treeWidget->topLevelItem(i)->text(LABEL_NAME_COLUMN);
    } 
    topLevelDialog.setComboBoxItems(topLabels);
    topLevelDialog.setWindowTitle("Create new Instance");
    if (QDialog::Accepted != topLevelDialog.exec())
    {
        return;
    }

    QString choosenLabel = topLevelDialog.textValue();
    QList<QTreeWidgetItem*> selectedTopLevelItem = m_ui->treeWidget->findItems(choosenLabel, Qt::MatchExactly);
    if (selectedTopLevelItem.count() != 1)
    {
    std::cout << "hallo was" << std::endl;
        return;
    }
    std::cout << "hallo" << std::endl;

    

    bool accepted;
    QString instance_name = QInputDialog::getText(m_dialog, tr("Choose Name for new Instance"),
    tr("Instance name:"), QLineEdit::Normal,
                    QString(choosenLabel + QString::number(selectedTopLevelItem[0]->childCount() + 1)) , &accepted);
    if (!accepted || instance_name.isEmpty())
    {
            //No valid Input
            return;
    }

    QColor label_color = QColorDialog::getColor(Qt::red, m_dialog, tr("Choose Label Color for first instance"));
    if (!label_color.isValid())
    {
            //Non Valid Color Return 
            return;
    }

    int id = m_id_hack++;
    QTreeWidgetItem * childItem = new QTreeWidgetItem();
    childItem->setText(LABEL_NAME_COLUMN, instance_name);
    childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
    childItem->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    childItem->setData(LABEL_ID_COLUMN, 1, label_color);
    childItem->setData(LABEL_ID_COLUMN, 0, id);
    selectedTopLevelItem[0]->addChild(childItem);


    //Add label to combo box 
    m_ui->selectedLabelComboBox->addItem(childItem->text(LABEL_NAME_COLUMN), id);
    Q_EMIT(labelAdded(childItem));

}

void LVRLabelDialog::comboBoxIndexChanged(int index)
{
	Q_EMIT(labelChanged(m_ui->selectedLabelComboBox->itemData(index).toInt()));
}

}
 /* namespace lvr2 */


