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
 * MainWindow.hpp
 *
 *  @date Jan 31, 2014
 *  @author Thomas Wiemann
 */
#ifndef MAINWINDOW_HPP_
#define MAINWINDOW_HPP_

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>

#include <QtGui>
#include "LVRMainWindowUI.h"
#include "LVRTreeWidgetHelper.hpp"
#include "../widgets/LVRCorrespondanceDialog.hpp"
#include "../vtkBridge/LVRPickingInteractor.hpp"
#include "../vtkBridge/LVRVtkArrow.hpp"

#include <iostream>
using std::cout;
using std::endl;

namespace lvr
{

class LVRMainWindow : public QMainWindow, public Ui::MainWindow
{
    Q_OBJECT
public:

    /**
     * @brief   MainWindow
     */
    LVRMainWindow();


    virtual ~LVRMainWindow();

public Q_SLOTS:
    void loadModel();
    void manualICP();
    void showTransformationDialog();
    void showTreeContextMenu(const QPoint&);
    void showColorDialog();
    void deleteModelItem();
    void changePointSize(int pointSize);
    void changeTransparency(int transparencyValue);
    void changeShading(int shader);
    void togglePoints(bool checkboxState);
    void toggleMeshes(bool checkboxState);
    void refreshView();
    void updateView();
    void saveCamera();
    void loadCamera();
    void removeArrow(LVRVtkArrow*);
    void addArrow(LVRVtkArrow*);
    void alignPointClouds();
    void exportSelectedModel();

protected Q_SLOTS:
    void onItemChange(QTreeWidgetItem* treeWidgetItem, int column);
    void restoreSliders(QTreeWidgetItem* treeWidgetItem, int column);

Q_SIGNALS:
    void correspondenceDialogOpened();

private:
    void setupQVTK();
    void connectSignalsAndSlots();

    LVRCorrespondanceDialog*            m_correspondanceDialog;
    vtkSmartPointer<vtkRenderer>        m_renderer;
    vtkSmartPointer<vtkCamera>			m_camera;
    QMenu*				                m_treeContextMenu;

    // Toolbar item "File"
	QAction*							m_actionOpen;
	QAction*							m_actionExport;
	QAction*							m_actionQuit;
	// Toolbar item "Views"
	QAction*							m_actionReset_Camera;
	QAction*							m_actionStore_Current_View;
	QAction*							m_actionRecall_Stored_View;
	// QToolbar below toolbar
	QAction*							m_actionShow_Points;
	QAction*							m_actionShow_Normals;
	QAction*							m_actionShow_Mesh;
	QAction*							m_actionShow_Wireframe;
    // Sliders below tree widget
    QSlider*							m_horizontalSliderPointSize;
    QSlider*							m_horizontalSliderContrastLow;
    QSlider*							m_horizontalSliderContrastHigh;
    QSlider*							m_horizontalSliderTransparency;
    // Combo boxes below sliders
    QComboBox*							m_comboBoxGradient;
    QComboBox*							m_comboBoxShading;
    // Buttons below combo boxes
    QPushButton*						m_buttonRecordPath;
    QPushButton*						m_buttonCreateMesh;
    QPushButton*						m_buttonExportData;
    QPushButton*						m_buttonTransformModel;

	QAction*				            m_actionShowColorDialog;
	QAction*			                m_actionDeleteModelItem;
	QAction*                            m_actionExportModelTransformed;

    LVRPickingInteractor*               m_pickingInteractor;
    LVRTreeWidgetHelper*                m_treeWidgetHelper;
};

} /* namespace lvr */

#endif /* MAINWINDOW_HPP_ */
