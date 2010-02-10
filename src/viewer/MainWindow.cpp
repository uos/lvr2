/********************************************************************************
** Form generated from reading ui file 'viewer.ui'
**
** Created: Mon Jan 18 09:59:05 2010
**      by: Qt User Interface Compiler version 4.5.2
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_VIEWER_H
#define UI_VIEWER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QFrame>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *action_open;
    QAction *action_exit;
    QAction *action_topView;
    QAction *action_perspective;
    QAction *action_editObjects;
    QAction *action_transformObject;
    QAction *action_loadMatrix;
    QWidget *centralwidget;
    QFrame *dummyFrame;
    QMenuBar *menubar;
    QMenu *menu_File;
    QMenu *menuView;
    QMenu *menuTools;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(789, 597);
        MainWindow->setAcceptDrops(true);
        MainWindow->setDockOptions(QMainWindow::AllowNestedDocks|QMainWindow::AnimatedDocks);
        action_open = new QAction(MainWindow);
        action_open->setObjectName(QString::fromUtf8("action_open"));
        action_exit = new QAction(MainWindow);
        action_exit->setObjectName(QString::fromUtf8("action_exit"));
        action_topView = new QAction(MainWindow);
        action_topView->setObjectName(QString::fromUtf8("action_topView"));
        action_perspective = new QAction(MainWindow);
        action_perspective->setObjectName(QString::fromUtf8("action_perspective"));
        action_editObjects = new QAction(MainWindow);
        action_editObjects->setObjectName(QString::fromUtf8("action_editObjects"));
        action_transformObject = new QAction(MainWindow);
        action_transformObject->setObjectName(QString::fromUtf8("action_transformObject"));
        action_loadMatrix = new QAction(MainWindow);
        action_loadMatrix->setObjectName(QString::fromUtf8("action_loadMatrix"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        centralwidget->setGeometry(QRect(0, 28, 789, 548));
        dummyFrame = new QFrame(centralwidget);
        dummyFrame->setObjectName(QString::fromUtf8("dummyFrame"));
        dummyFrame->setGeometry(QRect(0, 0, 781, 551));
        dummyFrame->setAcceptDrops(true);
        dummyFrame->setFrameShape(QFrame::Box);
        dummyFrame->setFrameShadow(QFrame::Raised);
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 789, 28));
        menu_File = new QMenu(menubar);
        menu_File->setObjectName(QString::fromUtf8("menu_File"));
        menuView = new QMenu(menubar);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        menuTools = new QMenu(menubar);
        menuTools->setObjectName(QString::fromUtf8("menuTools"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        statusbar->setGeometry(QRect(0, 576, 789, 21));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menu_File->menuAction());
        menubar->addAction(menuView->menuAction());
        menubar->addAction(menuTools->menuAction());
        menu_File->addAction(action_open);
        menu_File->addSeparator();
        menu_File->addAction(action_exit);
        menuView->addAction(action_topView);
        menuView->addAction(action_perspective);
        menuTools->addAction(action_transformObject);
        menuTools->addAction(action_loadMatrix);
        menuTools->addSeparator();

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "3D Viewer", 0, QApplication::UnicodeUTF8));
        action_open->setText(QApplication::translate("MainWindow", "&Open", 0, QApplication::UnicodeUTF8));
        action_exit->setText(QApplication::translate("MainWindow", "E&xit", 0, QApplication::UnicodeUTF8));
        action_topView->setText(QApplication::translate("MainWindow", "Top View", 0, QApplication::UnicodeUTF8));
        action_perspective->setText(QApplication::translate("MainWindow", "Perspective ", 0, QApplication::UnicodeUTF8));
        action_editObjects->setText(QApplication::translate("MainWindow", "Edit Objects", 0, QApplication::UnicodeUTF8));
        action_transformObject->setText(QApplication::translate("MainWindow", "Enter Transformation Matrix...", 0, QApplication::UnicodeUTF8));
        action_loadMatrix->setText(QApplication::translate("MainWindow", "Load Transformation Matrix...", 0, QApplication::UnicodeUTF8));
        menu_File->setTitle(QApplication::translate("MainWindow", "&File", 0, QApplication::UnicodeUTF8));
        menuView->setTitle(QApplication::translate("MainWindow", "View", 0, QApplication::UnicodeUTF8));
        menuTools->setTitle(QApplication::translate("MainWindow", "Tools", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VIEWER_H
