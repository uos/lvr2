/********************************************************************************
** Form generated from reading UI file 'objectdialog.ui'
**
** Created: Tue Jun 29 10:44:50 2010
**      by: Qt User Interface Compiler version 4.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_OBJECTDIALOG_H
#define UI_OBJECTDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDockWidget>
#include <QtGui/QHeaderView>
#include <QtGui/QListWidget>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ObjectDialog
{
public:
    QWidget *dockWidgetContents;
    QListWidget *listWidget;

    void setupUi(QDockWidget *ObjectDialog)
    {
        if (ObjectDialog->objectName().isEmpty())
            ObjectDialog->setObjectName(QString::fromUtf8("ObjectDialog"));
        ObjectDialog->resize(200, 137);
        ObjectDialog->setMinimumSize(QSize(200, 130));
        ObjectDialog->setMaximumSize(QSize(16777215, 150));
        ObjectDialog->setLayoutDirection(Qt::LeftToRight);
        ObjectDialog->setFloating(false);
        ObjectDialog->setFeatures(QDockWidget::DockWidgetClosable|QDockWidget::DockWidgetVerticalTitleBar);
        ObjectDialog->setAllowedAreas(Qt::LeftDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        dockWidgetContents->setGeometry(QRect(20, 0, 180, 137));
        listWidget = new QListWidget(dockWidgetContents);
        listWidget->setObjectName(QString::fromUtf8("listWidget"));
        listWidget->setGeometry(QRect(0, 10, 171, 121));
        listWidget->setMinimumSize(QSize(0, 0));
        listWidget->setMaximumSize(QSize(16777215, 16777215));
        ObjectDialog->setWidget(dockWidgetContents);

        retranslateUi(ObjectDialog);

        QMetaObject::connectSlotsByName(ObjectDialog);
    } // setupUi

    void retranslateUi(QDockWidget *ObjectDialog)
    {
        ObjectDialog->setWindowTitle(QApplication::translate("ObjectDialog", "Loaded Objects", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ObjectDialog: public Ui_ObjectDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_OBJECTDIALOG_H
