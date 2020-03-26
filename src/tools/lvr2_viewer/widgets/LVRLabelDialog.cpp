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

#include <QMessageBox>
#include <QFont>
#include <QFileDialog>
#include <QInputDialog>
#include <QColorDialog>
#include <vtkSelectEnclosedPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkPointData.h>
#include <vector>

#include <vtkLookupTable.h>
#include <vtkExtractGeometry.h>

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
    m_ui->tableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    QObject::connect(m_ui->newLabelButton, SIGNAL(pressed()), this, SLOT(addNewLabel()));
    QObject::connect(m_ui->tableWidget, SIGNAL( cellDoubleClicked (int, int) ), this, SLOT( cellSelected( int, int )));
    QObject::connect(m_ui->selectedLabelComboBox, SIGNAL( currentIndexChanged(int)), this, SLOT( comboBoxIndexChanged(int)));

}

LVRLabelDialog::~LVRLabelDialog()
{
	/*
    delete m_ui;
    delete m_dialog;
    */
    // TODO Auto-generated destructor stub
}
void LVRLabelDialog::cellSelected(int row, int column)
{
	if(column == LABEL_NAME_COLUMN)
	{
		//Edit Label name
		bool accepted;
		QTableWidgetItem* item = m_ui->tableWidget->item(row, column);
		QString label_name = QInputDialog::getText(m_dialog, tr("Select Label Name"),
		tr("Label name:"), QLineEdit::Normal,
				item->text(), &accepted);
		if (accepted && !label_name.isEmpty())
		{
			item->setText(label_name);
			int comboBoxPos = m_ui->selectedLabelComboBox->findData(m_ui->tableWidget->item(row, LABEL_ID_COLUMN)->text().toInt());
			if (comboBoxPos >= 0)
			{
				m_ui->selectedLabelComboBox->setItemText(comboBoxPos, label_name);

			}
			return;
		}
	}else if(column == LABEL_COLOR_COLUMN || column == LABEL_ID_COLUMN)
	{
		QColor label_color = QColorDialog::getColor(Qt::red, m_dialog, tr("Choose Label Color"));
		if (label_color.isValid())
		{
			m_ui->tableWidget->item(row, LABEL_COLOR_COLUMN)->setBackground(label_color);
			m_ui->tableWidget->item(row, LABEL_ID_COLUMN)->setData(1,label_color);

			//Update Color In picker
			Q_EMIT(labelAdded(m_ui->tableWidget->item(row, LABEL_ID_COLUMN)));
			return;
		}
	}
	
}

void LVRLabelDialog::updatePointCount(int selectedPointCount)
{

	int rows = m_ui->tableWidget->rowCount();
	for (int i = 0; i < rows; i++)
	{
		if(m_ui->selectedLabelComboBox->currentData().toInt() == m_ui->tableWidget->item(i, 4)->text().toInt())
		{
			m_ui->tableWidget->item(i, 2)->setText(QString::number(selectedPointCount));
		}
	}


}
void LVRLabelDialog::labelPoints()
{
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

	QColor label_color = QColorDialog::getColor(Qt::red, m_dialog, tr("Choose Label Color"));
	if (!label_color.isValid())
	{
		//Non Vlaid Color Return 
		return;
	}
	int rows = m_ui->tableWidget->rowCount();
	
	//generate new Table row
	m_ui->tableWidget->insertRow(rows);
	m_ui->tableWidget->setItem(rows, LABEL_NAME_COLUMN, new QTableWidgetItem(label_name));
	m_ui->tableWidget->setItem(rows, LABEL_COLOR_COLUMN, new QTableWidgetItem(" "));
	m_ui->tableWidget->item(rows, LABEL_COLOR_COLUMN)->setBackground(label_color);
	m_ui->tableWidget->setItem(rows, LABELED_POINT_COLUMN, new QTableWidgetItem("0"));
	m_ui->tableWidget->setItem(rows, LABEL_ID_COLUMN, new QTableWidgetItem(QString::number(rows)));
	m_ui->tableWidget->item(rows, LABEL_ID_COLUMN)->setData(1,label_color);
	//TODO generate a gloabal id field that isnt bound to table position 
	//TODO Add visible box and maybe link visibility to point cloud check with Thomas 
	//TODO Think about a better way than that hacky data color solution
	

	//Add label to combo box 
	m_ui->selectedLabelComboBox->addItem(label_name, rows);

	Q_EMIT(labelAdded(m_ui->tableWidget->item(rows, LABEL_ID_COLUMN)));
}

void LVRLabelDialog::comboBoxIndexChanged(int index)
{
	Q_EMIT(labelChanged(m_ui->selectedLabelComboBox->itemData(index).toInt()));
}
}
 /* namespace lvr2 */


