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
    QObject::connect(m_ui->loadLabeledPoints, SIGNAL(pressed()), this, SLOT(loadLabels()));
    QObject::connect(m_ui->newInstanceButton, SIGNAL(pressed()), this, SLOT(addNewInstance()));
    QObject::connect(m_ui->treeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(cellSelected(QTreeWidgetItem*, int)));
    QObject::connect(m_ui->selectedLabelComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(comboBoxIndexChanged(int)));
    QObject::connect(m_ui->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(visibilityChanged(QTreeWidgetItem*, int)));
//    QObject::connect(m_ui->labelSelectedPoints, SIGNAL(pressed()), this, SLOT(labelPoints()));
//
}

LVRLabelDialog::~LVRLabelDialog()
{
    delete m_ui;
    delete m_dialog;
    // TODO Auto-generated destructor stub
}

void LVRLabelDialog::showEvent()
{


}
void LVRLabelDialog::visibilityChanged(QTreeWidgetItem* changedItem, int column)
{
    if(column != LABEL_VISIBLE_COLUMN)
    {
        return;
    }

    //check if Instance or whole label changed
    if (changedItem->parent())
    {
        //parent exists item is an instance
        Q_EMIT(hidePoints(changedItem->data(LABEL_ID_COLUMN,0).toInt(), changedItem->checkState(LABEL_VISIBLE_COLUMN)));
    } else
    {
        //Check if unlabeled item
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
    }else if(column == LABEL_ID_COLUMN)
    {
        //Change 
        QColor label_color = QColorDialog::getColor(Qt::red, m_dialog, tr("Choose Label Color"));
        if (label_color.isValid())
        {
            item->setData(LABEL_ID_COLUMN, 1, label_color);
            if(item->parent())
            {
                //Update Color In picker
                Q_EMIT(labelAdded(item));
                return;
            }
            else
            {
                //ask if all childs Should be updated
    		QMessageBox colorUpdateDialog;
		colorUpdateDialog.setText("Labelclass default color changed. Shall all instance colors be updated?");
		colorUpdateDialog.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
	        colorUpdateDialog.setDefaultButton(QMessageBox::Yes);
                int returnValue = colorUpdateDialog.exec();
                if (returnValue == QMessageBox::Yes)
                {
                    //update All Childs 
                    for (int i = 0; i < item->childCount(); i++)
                    {
                        item->child(i)->setData(LABEL_ID_COLUMN, 1, label_color);
                        Q_EMIT(labelAdded(item->child(i)));
                    }
                }
	
            }
            }
    }
}

void LVRLabelDialog::updatePointCount(uint16_t id, int selectedPointCount)
{


    int topItemCount = m_ui->treeWidget->topLevelItemCount();
    for (int i = 0; i < topItemCount; i++)
    {
        QTreeWidgetItem* topLevelItem = m_ui->treeWidget->topLevelItem(i);
        int childCount = topLevelItem->childCount();
        for (int j = 0; j < childCount; j++)
        {
            if(id == topLevelItem->child(j)->data(LABEL_ID_COLUMN, 0).toInt())
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

void LVRLabelDialog::loadLabels()
{

    //TODO: What should be done if elements exists?
    QString fileName = QFileDialog::getOpenFileName(m_dialog,
                tr("Open HDF5 File"), QDir::homePath(), tr("HDF5 files (*.h5)"));
    if(!QFile::exists(fileName))
    {
        return;
    }

    HDF5Kernel kernel(fileName.toStdString());
    std::vector<std::string> pointCloudNames;
    kernel.subGroupNames("pointclouds", pointCloudNames);
    for (auto pointcloudName : pointCloudNames)
    {
        //pointclouds
        boost::filesystem::path classGroup = (boost::filesystem::path("pointclouds") / boost::filesystem::path(pointcloudName) / boost::filesystem::path("labels"));
        std::vector<std::string> labelClasses;
        kernel.subGroupNames(classGroup.string(), labelClasses);
        for (auto labelClass : labelClasses)
        {
            //Get TopLevel Item for tree view
            QTreeWidgetItem * item = new QTreeWidgetItem();
            item->setText(0, QString::fromStdString(labelClass));
            item->setText(LABELED_POINT_COLUMN, QString::number(0));
            item->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);

            if (m_ui->treeWidget->topLevelItemCount() == 0)
            {
                m_ui->newInstanceButton->setEnabled(true);
            }
            m_ui->treeWidget->addTopLevelItem(item);   


            //pointclouds/$name/labels/$labelname
            boost::filesystem::path instanceGroup = (classGroup / boost::filesystem::path(labelClass));
            std::vector<std::string> labelInstances;
            kernel.subGroupNames(instanceGroup.string(), labelInstances);
            for (auto instance : labelInstances)
            {

                int id = 0;
                boost::filesystem::path finalGroup = instanceGroup;
                //pointclouds/$name/labels/$labelname/instance
                finalGroup = (instanceGroup / boost::filesystem::path(instance));
                if (labelClass != "Unlabeled")
                {
                    id = m_id_hack++;

                } 

                //Get Color and IDs
                boost::shared_array<int> rgbData;
                std::vector<size_t> rgbDim;
                boost::shared_array<int> idData;
                std::vector<size_t> idDim;
                idData = kernel.loadArray<int>(finalGroup.string(), "IDs", idDim);
                rgbData = kernel.loadArray<int>(finalGroup.string(), "Color", rgbDim);

                //Add Child to top Level
                QTreeWidgetItem * childItem = new QTreeWidgetItem();
                childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
                childItem->setText(0, QString::fromStdString(instance));
                QColor label_color(rgbData[0], rgbData[1], rgbData[2]);
                childItem->setData(LABEL_ID_COLUMN, 1, label_color);
                childItem->setData(LABEL_ID_COLUMN, 0, id);
                item->addChild(childItem);
                Q_EMIT(labelAdded(childItem));
                std::vector<int> out(idData.get(), idData.get() + idDim[0]);
                Q_EMIT(labelLoaded(id, out));
                if (labelClass != "Unlabeled")
                {
                    m_ui->selectedLabelComboBox->addItem(childItem->text(LABEL_NAME_COLUMN), id);
                }
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
    dialog.setDirectory(QDir::homePath());
    dialog.setFileMode(QFileDialog::AnyFile);
    QString strFile = dialog.getSaveFileName(m_dialog, "Creat New HDF5 File","","");

    HDF5Kernel label_hdf5kernel((strFile + QString(".h5")).toStdString());
    int topItemCount = m_ui->treeWidget->topLevelItemCount();


    //TODO This should be for all Pointclouds
    boost::filesystem::path pointcloudName(m_points.begin()->first);
    auto points = m_points.begin()->second;

    double* pointsData = new double[points->GetNumberOfPoints() * 3];
   // const size_t point_number = points->GetNumberOfPoints();
   // std::array<std::array<float, 3>, point_number> test;
    
    for (int i = 0; i < points->GetNumberOfPoints(); i++)
    {
	auto point = points->GetPoint(i);
        pointsData[(3 * i)] = point[0];
        pointsData[(3 * i) + 1] = point[1];
        pointsData[(3 * i) + 2] = point[2];

    }

    std::vector<size_t> pointsDimension = {3, points->GetNumberOfPoints()};
    boost::shared_array<double> sharedPoints(pointsData);

    //Unlabeled top item
    QTreeWidgetItem* unlabeledItem;
    
    boost::filesystem::path pointGroup = (boost::filesystem::path("pointclouds") / pointcloudName);
    label_hdf5kernel.saveDoubleArray(pointGroup.string(), "Points" , pointsDimension, sharedPoints);
    for (int i = 0; i < topItemCount; i++)
    {
        QTreeWidgetItem* topLevelItem = m_ui->treeWidget->topLevelItem(i);
        if(topLevelItem->text(LABEL_NAME_COLUMN) == "Unlabeled")
        {
            unlabeledItem = topLevelItem;
        }
        boost::filesystem::path topLabel = topLevelItem->text(LABEL_NAME_COLUMN).toStdString();
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
                boost::filesystem::path childLabel = (topLevelItem->child(j)->text(LABEL_NAME_COLUMN)).toStdString();
                boost::filesystem::path completeGroup = (pointGroup / boost::filesystem::path("labels") / topLabel / childLabel);

                label_hdf5kernel.saveArray(completeGroup.string(), "IDs" , dimension, data);
                int* rgbSharedData = new int[3];
                (topLevelItem->child(j)->data(LABEL_ID_COLUMN, 1)).value<QColor>().getRgb(&rgbSharedData[0], &rgbSharedData[1], &rgbSharedData[2]);
                boost::shared_array<int> rgbData(rgbSharedData);
                std::vector<size_t> rgbDimension = {3};
                label_hdf5kernel.saveArray(completeGroup.string(), "Color" , rgbDimension, rgbData);
            }
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

    QColor label_color = QColorDialog::getColor(Qt::red, m_dialog, tr("Choose default Label Color for label Class(willbe used for first isntance)"));
    if (!label_color.isValid())
    {
            //Non Valid Color Return 
            return;
    }

    if (m_ui->treeWidget->topLevelItemCount() == 0)
    {

        //Setting up Top Label
        QTreeWidgetItem * item = new QTreeWidgetItem();
        item->setText(0, "Unlabeled");
        item->setText(LABELED_POINT_COLUMN, QString::number(0));
        item->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
        item->setData(LABEL_ID_COLUMN, 1, QColor(Qt::red));

        //Setting up new child item
        QTreeWidgetItem * childItem = new QTreeWidgetItem();
        childItem->setText(LABEL_NAME_COLUMN, QString("Unlabeled") + QString::number(1));
        childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
        childItem->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
        childItem->setData(LABEL_ID_COLUMN, 1, QColor(Qt::red));
        childItem->setData(LABEL_ID_COLUMN, 0, 0);
        item->addChild(childItem);
        m_ui->treeWidget->addTopLevelItem(item);    
        //Added first Top Level item enable instance button
        m_ui->newInstanceButton->setEnabled(true);
        Q_EMIT(labelAdded(childItem));
    }

    int id = m_id_hack++;
    //Setting up new Toplevel item
    QTreeWidgetItem * item = new QTreeWidgetItem();
    item->setText(0, label_name);
    item->setText(LABELED_POINT_COLUMN, QString::number(0));
    item->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    item->setData(LABEL_ID_COLUMN, 1, label_color);

    //Setting up new child item
    QTreeWidgetItem * childItem = new QTreeWidgetItem();
    childItem->setText(LABEL_NAME_COLUMN, label_name + QString::number(1));
    childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
    childItem->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    childItem->setData(LABEL_ID_COLUMN, 1, label_color);
    childItem->setData(LABEL_ID_COLUMN, 0, id);
    item->addChild(childItem);
    m_ui->treeWidget->addTopLevelItem(item);    
    
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
        if (m_ui->treeWidget->topLevelItem(i)->text(LABEL_NAME_COLUMN) != "Unlabeled")
        {
            topLabels << m_ui->treeWidget->topLevelItem(i)->text(LABEL_NAME_COLUMN);
        }
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
        return;
    }

    

    bool accepted;
    QString instance_name = QInputDialog::getText(m_dialog, tr("Choose Name for new Instance"),
    tr("Instance name:"), QLineEdit::Normal,
                    QString(choosenLabel + QString::number(selectedTopLevelItem[0]->childCount() + 1)) , &accepted);
    if (!accepted || instance_name.isEmpty())
    {
            //No valid Input
            return;
    }

    QColor label_color = QColorDialog::getColor(selectedTopLevelItem[0]->data(3,1).value<QColor>(), m_dialog, tr("Choose Label Color for first instance"));
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

void LVRLabelDialog::setPoints(const std::string pointcloudName, const vtkSmartPointer<vtkPolyData> points)
{
    m_points[pointcloudName] = points;
}
}

 /* namespace lvr2 */


