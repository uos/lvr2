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
	m_ui->comboBox->addItem("Test");

}

LVRLabelDialog::~LVRLabelDialog()
{
    delete m_ui;
    delete m_dialog;
    // TODO Auto-generated destructor stub
}

void LVRLabelDialog::labelPoints()
{
	bool ok;
	QString text = QInputDialog::getText(0, "Insert New Label", "New Label:", QLineEdit::Normal, "", &ok);
	vtkSmartPointer<vtkSelectEnclosedPoints> selectEnclosedPoints = vtkSmartPointer<vtkSelectEnclosedPoints>::New();
}
void LVRLabelDialog::insertNewCluster(double* bounds)
{
	bool ok;
	QString text = QInputDialog::getText(0, "BLABLA", "New Label:", QLineEdit::Normal, "", &ok);
	std::cout << bounds[0]<< " "  << bounds [1] << " "<<bounds[2] << " " <<  bounds[3]<<bounds[4] << " " <<  bounds[5]<<std::endl;
	vtkSmartPointer<vtkSelectEnclosedPoints> selectEnclosedPoints = vtkSmartPointer<vtkSelectEnclosedPoints>::New();
	
	vtkSmartPointer<vtkCubeSource> cubeSource = 
		vtkSmartPointer<vtkCubeSource>::New();
	cubeSource->SetBounds(bounds);
	cubeSource->Update();
	vtkPolyData* cube = cubeSource->GetOutput();


	vtkSmartPointer<vtkPolyData> pointsPolydata = 
		vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPoints> points = 
		vtkSmartPointer<vtkPoints>::New();
	QTreeWidgetItemIterator it(m_treeWidget);
	LVRPointCloudItem* citem = static_cast<LVRPointCloudItem*>(*it);
	
	while (*it)
	{
        	QTreeWidgetItem* item = *it;

		if ( item->type() == LVRPointCloudItemType)
		{
			LVRPointCloudItem* citem = static_cast<LVRPointCloudItem*>(*it);
			points->SetData(citem->getPointBufferBridge()->getPointCloudActor()->GetMapper()->GetInput()->GetPointData()->GetScalars());

			selectEnclosedPoints->SetInputData(citem->getPointBufferBridge()->getPointCloudActor()->GetMapper()->GetInput());
			pointsPolydata->SetPoints(points);
			selectEnclosedPoints->SetInputData(pointsPolydata);
		}
		it++;
	}
	//selectEnclosedPoints->SetInputData(pointsPolydata->GetInput());
	selectEnclosedPoints->SetSurfaceData(cube);
	selectEnclosedPoints->Update();
 vtkDataArray* insideArray = vtkDataArray::SafeDownCast(selectEnclosedPoints->GetOutput()->GetPointData()->GetArray("SelectedPoints"));
	std::cout << "hallo " << insideArray->GetNumberOfTuples() <<std::endl;  	
 
  	std::vector<int> ids;


	//citem->getPointBuffer()->getU
	unsigned char color[3];
	color[0] = 1;
	color[1] = 1;
	color[2] = 0;

  	for(vtkIdType i = 0; i < insideArray->GetNumberOfTuples(); i++)
    { 
	    if (insideArray->GetComponent(i,0))
	    {
		    ids.push_back(i);

	    }
	    
     //std::cout << i << " : " << insideArray->GetComponent(i,0) << std::endl;
    }


	m_dialog->hide();

}

}
 /* namespace lvr2 */


