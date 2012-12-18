/********************************************************************************
** Form generated from reading UI file 'SceneDockWidgetUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef SCENEDOCKWIDGETUI_H
#define SCENEDOCKWIDGETUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDockWidget>
#include <QtGui/QHeaderView>
#include <QtGui/QTreeWidget>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SceneDockWidgetUI
{
public:
    QAction *actionExport;
    QAction *actionChangeName;
    QWidget *dockWidgetContents;
    QTreeWidget *treeWidget;

    void setupUi(QDockWidget *SceneDockWidgetUI)
    {
        if (SceneDockWidgetUI->objectName().isEmpty())
            SceneDockWidgetUI->setObjectName(QString::fromUtf8("SceneDockWidgetUI"));
        SceneDockWidgetUI->resize(180, 207);
        SceneDockWidgetUI->setMinimumSize(QSize(180, 79));
        SceneDockWidgetUI->setMaximumSize(QSize(180, 207));
        SceneDockWidgetUI->setLayoutDirection(Qt::RightToLeft);
        SceneDockWidgetUI->setFeatures(QDockWidget::DockWidgetFeatureMask);
        SceneDockWidgetUI->setAllowedAreas(Qt::LeftDockWidgetArea);
        actionExport = new QAction(SceneDockWidgetUI);
        actionExport->setObjectName(QString::fromUtf8("actionExport"));
        actionChangeName = new QAction(SceneDockWidgetUI);
        actionChangeName->setObjectName(QString::fromUtf8("actionChangeName"));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        treeWidget = new QTreeWidget(dockWidgetContents);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(0, QString::fromUtf8("1"));
        treeWidget->setHeaderItem(__qtreewidgetitem);
        treeWidget->setObjectName(QString::fromUtf8("treeWidget"));
        treeWidget->setGeometry(QRect(0, 10, 151, 192));
        treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
        treeWidget->setLayoutDirection(Qt::LeftToRight);
        treeWidget->setSelectionBehavior(QAbstractItemView::SelectItems);
        treeWidget->setIndentation(10);
        treeWidget->setHeaderHidden(true);
        treeWidget->setColumnCount(1);
        treeWidget->header()->setVisible(false);
        SceneDockWidgetUI->setWidget(dockWidgetContents);

        retranslateUi(SceneDockWidgetUI);

        QMetaObject::connectSlotsByName(SceneDockWidgetUI);
    } // setupUi

    void retranslateUi(QDockWidget *SceneDockWidgetUI)
    {
        SceneDockWidgetUI->setWindowTitle(QApplication::translate("SceneDockWidgetUI", "Scene Objects", 0, QApplication::UnicodeUTF8));
        actionExport->setText(QApplication::translate("SceneDockWidgetUI", "Export selection", 0, QApplication::UnicodeUTF8));
        actionChangeName->setText(QApplication::translate("SceneDockWidgetUI", "Change Label", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SceneDockWidgetUI: public Ui_SceneDockWidgetUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // SCENEDOCKWIDGETUI_H
