/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *action_Open;
    QAction *action_Quit;
    QAction *actionShow_entire_scene;
    QAction *actionCenter_on_selected_object;
    QAction *actionReset_Postion;
    QAction *actionGo_to_last_postion;
    QAction *actionAdd_current_view_to_path;
    QAction *actionAnimate_Path;
    QAction *actionEdit_path;
    QAction *actionSend_Commands;
    QAction *actionSave_Screenshot;
    QAction *actionLoaded_Objects;
    QAction *action_Robot_control;
    QAction *actionXZ_ortho_projection;
    QAction *actionYZ_ortho_projection;
    QAction *actionXY_ortho_projection;
    QAction *actionPerspective_projection;
    QAction *actionOpen_a_scan_in_slam6d_format;
    QAction *actionToggle_fog;
    QAction *actionFog_settings;
    QAction *actionWireframeView;
    QAction *actionVertexView;
    QAction *actionPointCloudView;
    QAction *actionSurfaceView;
    QAction *actionGenerateMesh;
    QAction *actionMatchPointClouds;
    QAction *actionShowSelection;
    QAction *actionPointNormalView;
    QAction *actionRenderingSettings;
    QAction *action_Kinect;
    QWidget *centralwidget;
    QMenuBar *menubar;
    QMenu *menu_File;
    QMenu *menu_View;
    QMenu *menu_Tools;
    QMenu *menuView;
    QMenu *menu_Scene;
    QMenu *menuData_Sources;
    QStatusBar *statusbar;
    QToolBar *toolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1044, 840);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        MainWindow->setMinimumSize(QSize(320, 200));
        MainWindow->setMaximumSize(QSize(100000, 100000));
        action_Open = new QAction(MainWindow);
        action_Open->setObjectName(QString::fromUtf8("action_Open"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/qv_import.png"), QSize(), QIcon::Normal, QIcon::Off);
        action_Open->setIcon(icon);
        action_Quit = new QAction(MainWindow);
        action_Quit->setObjectName(QString::fromUtf8("action_Quit"));
        actionShow_entire_scene = new QAction(MainWindow);
        actionShow_entire_scene->setObjectName(QString::fromUtf8("actionShow_entire_scene"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/qv_showall.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_entire_scene->setIcon(icon1);
        actionCenter_on_selected_object = new QAction(MainWindow);
        actionCenter_on_selected_object->setObjectName(QString::fromUtf8("actionCenter_on_selected_object"));
        actionCenter_on_selected_object->setEnabled(false);
        actionReset_Postion = new QAction(MainWindow);
        actionReset_Postion->setObjectName(QString::fromUtf8("actionReset_Postion"));
        actionGo_to_last_postion = new QAction(MainWindow);
        actionGo_to_last_postion->setObjectName(QString::fromUtf8("actionGo_to_last_postion"));
        actionGo_to_last_postion->setEnabled(false);
        actionAdd_current_view_to_path = new QAction(MainWindow);
        actionAdd_current_view_to_path->setObjectName(QString::fromUtf8("actionAdd_current_view_to_path"));
        actionAdd_current_view_to_path->setEnabled(false);
        actionAnimate_Path = new QAction(MainWindow);
        actionAnimate_Path->setObjectName(QString::fromUtf8("actionAnimate_Path"));
        actionAnimate_Path->setEnabled(false);
        actionEdit_path = new QAction(MainWindow);
        actionEdit_path->setObjectName(QString::fromUtf8("actionEdit_path"));
        actionEdit_path->setEnabled(false);
        actionSend_Commands = new QAction(MainWindow);
        actionSend_Commands->setObjectName(QString::fromUtf8("actionSend_Commands"));
        actionSend_Commands->setEnabled(false);
        actionSave_Screenshot = new QAction(MainWindow);
        actionSave_Screenshot->setObjectName(QString::fromUtf8("actionSave_Screenshot"));
        actionSave_Screenshot->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/qv_screenshot.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSave_Screenshot->setIcon(icon2);
        actionLoaded_Objects = new QAction(MainWindow);
        actionLoaded_Objects->setObjectName(QString::fromUtf8("actionLoaded_Objects"));
        action_Robot_control = new QAction(MainWindow);
        action_Robot_control->setObjectName(QString::fromUtf8("action_Robot_control"));
        actionXZ_ortho_projection = new QAction(MainWindow);
        actionXZ_ortho_projection->setObjectName(QString::fromUtf8("actionXZ_ortho_projection"));
        actionXZ_ortho_projection->setEnabled(true);
        actionYZ_ortho_projection = new QAction(MainWindow);
        actionYZ_ortho_projection->setObjectName(QString::fromUtf8("actionYZ_ortho_projection"));
        actionYZ_ortho_projection->setEnabled(true);
        actionXY_ortho_projection = new QAction(MainWindow);
        actionXY_ortho_projection->setObjectName(QString::fromUtf8("actionXY_ortho_projection"));
        actionXY_ortho_projection->setEnabled(true);
        actionPerspective_projection = new QAction(MainWindow);
        actionPerspective_projection->setObjectName(QString::fromUtf8("actionPerspective_projection"));
        actionOpen_a_scan_in_slam6d_format = new QAction(MainWindow);
        actionOpen_a_scan_in_slam6d_format->setObjectName(QString::fromUtf8("actionOpen_a_scan_in_slam6d_format"));
        actionOpen_a_scan_in_slam6d_format->setEnabled(false);
        actionToggle_fog = new QAction(MainWindow);
        actionToggle_fog->setObjectName(QString::fromUtf8("actionToggle_fog"));
        actionToggle_fog->setCheckable(true);
        actionToggle_fog->setEnabled(false);
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/qv_fog.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionToggle_fog->setIcon(icon3);
        actionFog_settings = new QAction(MainWindow);
        actionFog_settings->setObjectName(QString::fromUtf8("actionFog_settings"));
        actionFog_settings->setEnabled(false);
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/qv_fogsettings.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionFog_settings->setIcon(icon4);
        actionWireframeView = new QAction(MainWindow);
        actionWireframeView->setObjectName(QString::fromUtf8("actionWireframeView"));
        actionWireframeView->setCheckable(true);
        actionWireframeView->setEnabled(false);
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/qv_wireframe.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionWireframeView->setIcon(icon5);
        actionVertexView = new QAction(MainWindow);
        actionVertexView->setObjectName(QString::fromUtf8("actionVertexView"));
        actionVertexView->setCheckable(true);
        actionVertexView->setEnabled(false);
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/qv_vertices.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionVertexView->setIcon(icon6);
        actionPointCloudView = new QAction(MainWindow);
        actionPointCloudView->setObjectName(QString::fromUtf8("actionPointCloudView"));
        actionPointCloudView->setEnabled(false);
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/qv_points.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPointCloudView->setIcon(icon7);
        actionSurfaceView = new QAction(MainWindow);
        actionSurfaceView->setObjectName(QString::fromUtf8("actionSurfaceView"));
        actionSurfaceView->setCheckable(true);
        actionSurfaceView->setEnabled(false);
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/qv_surfaces.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSurfaceView->setIcon(icon8);
        actionGenerateMesh = new QAction(MainWindow);
        actionGenerateMesh->setObjectName(QString::fromUtf8("actionGenerateMesh"));
        actionMatchPointClouds = new QAction(MainWindow);
        actionMatchPointClouds->setObjectName(QString::fromUtf8("actionMatchPointClouds"));
        actionShowSelection = new QAction(MainWindow);
        actionShowSelection->setObjectName(QString::fromUtf8("actionShowSelection"));
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/qv_showSelection.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShowSelection->setIcon(icon9);
        actionPointNormalView = new QAction(MainWindow);
        actionPointNormalView->setObjectName(QString::fromUtf8("actionPointNormalView"));
        actionPointNormalView->setCheckable(true);
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/qv_pointNormals.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPointNormalView->setIcon(icon10);
        actionRenderingSettings = new QAction(MainWindow);
        actionRenderingSettings->setObjectName(QString::fromUtf8("actionRenderingSettings"));
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/qv_rendering.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionRenderingSettings->setIcon(icon11);
        action_Kinect = new QAction(MainWindow);
        action_Kinect->setObjectName(QString::fromUtf8("action_Kinect"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1044, 22));
        menu_File = new QMenu(menubar);
        menu_File->setObjectName(QString::fromUtf8("menu_File"));
        menu_View = new QMenu(menubar);
        menu_View->setObjectName(QString::fromUtf8("menu_View"));
        menu_Tools = new QMenu(menubar);
        menu_Tools->setObjectName(QString::fromUtf8("menu_Tools"));
        menuView = new QMenu(menubar);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        menu_Scene = new QMenu(menubar);
        menu_Scene->setObjectName(QString::fromUtf8("menu_Scene"));
        menuData_Sources = new QMenu(menubar);
        menuData_Sources->setObjectName(QString::fromUtf8("menuData_Sources"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName(QString::fromUtf8("toolBar"));
        toolBar->setEnabled(true);
        QPalette palette;
        QBrush brush(QColor(255, 255, 255, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush1(QColor(91, 149, 242, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        toolBar->setPalette(palette);
        toolBar->setAutoFillBackground(false);
        toolBar->setIconSize(QSize(32, 32));
        toolBar->setToolButtonStyle(Qt::ToolButtonIconOnly);
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBar);

        menubar->addAction(menu_File->menuAction());
        menubar->addAction(menuData_Sources->menuAction());
        menubar->addAction(menu_Tools->menuAction());
        menubar->addAction(menuView->menuAction());
        menubar->addAction(menu_View->menuAction());
        menubar->addAction(menu_Scene->menuAction());
        menu_File->addAction(action_Open);
        menu_File->addAction(actionOpen_a_scan_in_slam6d_format);
        menu_File->addSeparator();
        menu_File->addAction(action_Quit);
        menu_View->addSeparator();
        menu_View->addAction(actionShow_entire_scene);
        menu_View->addAction(actionCenter_on_selected_object);
        menu_View->addAction(actionGo_to_last_postion);
        menu_View->addSeparator();
        menu_View->addAction(actionAdd_current_view_to_path);
        menu_View->addAction(actionEdit_path);
        menu_View->addAction(actionAnimate_Path);
        menu_View->addSeparator();
        menu_View->addAction(actionXZ_ortho_projection);
        menu_View->addAction(actionYZ_ortho_projection);
        menu_View->addAction(actionXY_ortho_projection);
        menu_View->addAction(actionPerspective_projection);
        menu_Tools->addAction(actionSave_Screenshot);
        menu_Tools->addAction(actionGenerateMesh);
        menu_Tools->addAction(actionMatchPointClouds);
        menu_Scene->addAction(actionToggle_fog);
        menu_Scene->addAction(actionFog_settings);
        menuData_Sources->addAction(action_Kinect);
        toolBar->addAction(action_Open);
        toolBar->addSeparator();
        toolBar->addAction(actionPointCloudView);
        toolBar->addAction(actionPointNormalView);
        toolBar->addAction(actionVertexView);
        toolBar->addAction(actionWireframeView);
        toolBar->addAction(actionSurfaceView);
        toolBar->addAction(actionRenderingSettings);
        toolBar->addAction(actionToggle_fog);
        toolBar->addAction(actionFog_settings);
        toolBar->addSeparator();
        toolBar->addAction(actionShowSelection);
        toolBar->addAction(actionShow_entire_scene);
        toolBar->addAction(actionSave_Screenshot);
        toolBar->addSeparator();

        retranslateUi(MainWindow);
        QObject::connect(action_Quit, SIGNAL(activated()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0, QApplication::UnicodeUTF8));
        action_Open->setText(QApplication::translate("MainWindow", "&Open...", 0, QApplication::UnicodeUTF8));
        action_Quit->setText(QApplication::translate("MainWindow", "&Quit", 0, QApplication::UnicodeUTF8));
        actionShow_entire_scene->setText(QApplication::translate("MainWindow", "Show entire scene", 0, QApplication::UnicodeUTF8));
        actionCenter_on_selected_object->setText(QApplication::translate("MainWindow", "Show selected object", 0, QApplication::UnicodeUTF8));
        actionReset_Postion->setText(QApplication::translate("MainWindow", "Reset Postion", 0, QApplication::UnicodeUTF8));
        actionGo_to_last_postion->setText(QApplication::translate("MainWindow", "Go to last postion", 0, QApplication::UnicodeUTF8));
        actionAdd_current_view_to_path->setText(QApplication::translate("MainWindow", "Add current view to path", 0, QApplication::UnicodeUTF8));
        actionAnimate_Path->setText(QApplication::translate("MainWindow", "Animate path", 0, QApplication::UnicodeUTF8));
        actionEdit_path->setText(QApplication::translate("MainWindow", "Edit path...", 0, QApplication::UnicodeUTF8));
        actionSend_Commands->setText(QApplication::translate("MainWindow", "Send Commands...", 0, QApplication::UnicodeUTF8));
        actionSave_Screenshot->setText(QApplication::translate("MainWindow", "Save Screenshot...", 0, QApplication::UnicodeUTF8));
        actionLoaded_Objects->setText(QApplication::translate("MainWindow", "&Loaded objects", 0, QApplication::UnicodeUTF8));
        action_Robot_control->setText(QApplication::translate("MainWindow", "&Robot control", 0, QApplication::UnicodeUTF8));
        actionXZ_ortho_projection->setText(QApplication::translate("MainWindow", "XZ ortho projection", 0, QApplication::UnicodeUTF8));
        actionYZ_ortho_projection->setText(QApplication::translate("MainWindow", "YZ ortho projection", 0, QApplication::UnicodeUTF8));
        actionXY_ortho_projection->setText(QApplication::translate("MainWindow", "XY ortho projection", 0, QApplication::UnicodeUTF8));
        actionPerspective_projection->setText(QApplication::translate("MainWindow", "Perspective projection", 0, QApplication::UnicodeUTF8));
        actionOpen_a_scan_in_slam6d_format->setText(QApplication::translate("MainWindow", "Load a slam6d scan...", 0, QApplication::UnicodeUTF8));
        actionToggle_fog->setText(QApplication::translate("MainWindow", "Toggle fog", 0, QApplication::UnicodeUTF8));
        actionFog_settings->setText(QApplication::translate("MainWindow", "Fog settings...", 0, QApplication::UnicodeUTF8));
        actionWireframeView->setText(QApplication::translate("MainWindow", "actionWireframeView", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionWireframeView->setToolTip(QApplication::translate("MainWindow", "Render objects in wireframe mode", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionVertexView->setText(QApplication::translate("MainWindow", "actionVertexView", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionVertexView->setToolTip(QApplication::translate("MainWindow", "Renders vertices only", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionPointCloudView->setText(QApplication::translate("MainWindow", "actionPointCloudView", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionPointCloudView->setToolTip(QApplication::translate("MainWindow", "Renders loaded point clouds only", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionSurfaceView->setText(QApplication::translate("MainWindow", "actionSurfaceView", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionSurfaceView->setToolTip(QApplication::translate("MainWindow", "Render surfaces of loded meshes", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionGenerateMesh->setText(QApplication::translate("MainWindow", "Generate Mesh", 0, QApplication::UnicodeUTF8));
        actionMatchPointClouds->setText(QApplication::translate("MainWindow", "Match Point Clouds", 0, QApplication::UnicodeUTF8));
        actionShowSelection->setText(QApplication::translate("MainWindow", "Show Selection", 0, QApplication::UnicodeUTF8));
        actionPointNormalView->setText(QApplication::translate("MainWindow", "actionPointNormalView", 0, QApplication::UnicodeUTF8));
        actionRenderingSettings->setText(QApplication::translate("MainWindow", "Rendering Settings", 0, QApplication::UnicodeUTF8));
        action_Kinect->setText(QApplication::translate("MainWindow", "Kinect...", 0, QApplication::UnicodeUTF8));
        menu_File->setTitle(QApplication::translate("MainWindow", "&File", 0, QApplication::UnicodeUTF8));
        menu_View->setTitle(QApplication::translate("MainWindow", "&Camera", 0, QApplication::UnicodeUTF8));
        menu_Tools->setTitle(QApplication::translate("MainWindow", "&Tools", 0, QApplication::UnicodeUTF8));
        menuView->setTitle(QApplication::translate("MainWindow", "&Docks", 0, QApplication::UnicodeUTF8));
        menu_Scene->setTitle(QApplication::translate("MainWindow", "&Scene", 0, QApplication::UnicodeUTF8));
        menuData_Sources->setTitle(QApplication::translate("MainWindow", "Data Sources", 0, QApplication::UnicodeUTF8));
        toolBar->setWindowTitle(QApplication::translate("MainWindow", "toolBar", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MAINWINDOW_H
