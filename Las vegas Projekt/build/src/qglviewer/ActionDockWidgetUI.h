/********************************************************************************
** Form generated from reading UI file 'ActionDockWidgetUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef ACTIONDOCKWIDGETUI_H
#define ACTIONDOCKWIDGETUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCommandLinkButton>
#include <QtGui/QDockWidget>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ActionDockWidgetUI
{
public:
    QWidget *dockWidgetContents;
    QCommandLinkButton *buttonCreateMesh;
    QCommandLinkButton *buttonTransform;
    QCommandLinkButton *buttonExport;
    QCommandLinkButton *buttonDelete;
    QCommandLinkButton *buttonAnimation;

    void setupUi(QDockWidget *ActionDockWidgetUI)
    {
        if (ActionDockWidgetUI->objectName().isEmpty())
            ActionDockWidgetUI->setObjectName(QString::fromUtf8("ActionDockWidgetUI"));
        ActionDockWidgetUI->resize(180, 200);
        ActionDockWidgetUI->setMinimumSize(QSize(180, 200));
        ActionDockWidgetUI->setMaximumSize(QSize(180, 200));
        ActionDockWidgetUI->setLayoutDirection(Qt::RightToLeft);
        ActionDockWidgetUI->setFeatures(QDockWidget::DockWidgetFeatureMask);
        ActionDockWidgetUI->setAllowedAreas(Qt::LeftDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        buttonCreateMesh = new QCommandLinkButton(dockWidgetContents);
        buttonCreateMesh->setObjectName(QString::fromUtf8("buttonCreateMesh"));
        buttonCreateMesh->setGeometry(QRect(0, 80, 151, 41));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/qv_createMesh.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonCreateMesh->setIcon(icon);
        buttonCreateMesh->setIconSize(QSize(25, 25));
        buttonTransform = new QCommandLinkButton(dockWidgetContents);
        buttonTransform->setObjectName(QString::fromUtf8("buttonTransform"));
        buttonTransform->setGeometry(QRect(0, 0, 151, 41));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/qv_transform.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonTransform->setIcon(icon1);
        buttonTransform->setIconSize(QSize(25, 25));
        buttonExport = new QCommandLinkButton(dockWidgetContents);
        buttonExport->setObjectName(QString::fromUtf8("buttonExport"));
        buttonExport->setGeometry(QRect(0, 160, 151, 41));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/qv_export.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonExport->setIcon(icon2);
        buttonExport->setIconSize(QSize(25, 25));
        buttonDelete = new QCommandLinkButton(dockWidgetContents);
        buttonDelete->setObjectName(QString::fromUtf8("buttonDelete"));
        buttonDelete->setGeometry(QRect(0, 40, 151, 41));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/qv_delete.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonDelete->setIcon(icon3);
        buttonDelete->setIconSize(QSize(25, 25));
        buttonAnimation = new QCommandLinkButton(dockWidgetContents);
        buttonAnimation->setObjectName(QString::fromUtf8("buttonAnimation"));
        buttonAnimation->setGeometry(QRect(0, 120, 151, 41));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/qv_animation.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonAnimation->setIcon(icon4);
        buttonAnimation->setIconSize(QSize(25, 25));
        ActionDockWidgetUI->setWidget(dockWidgetContents);

        retranslateUi(ActionDockWidgetUI);

        QMetaObject::connectSlotsByName(ActionDockWidgetUI);
    } // setupUi

    void retranslateUi(QDockWidget *ActionDockWidgetUI)
    {
        ActionDockWidgetUI->setWindowTitle(QApplication::translate("ActionDockWidgetUI", "Actions", 0, QApplication::UnicodeUTF8));
        buttonCreateMesh->setText(QApplication::translate("ActionDockWidgetUI", "Create Mesh", 0, QApplication::UnicodeUTF8));
        buttonTransform->setText(QApplication::translate("ActionDockWidgetUI", "Transform", 0, QApplication::UnicodeUTF8));
        buttonExport->setText(QApplication::translate("ActionDockWidgetUI", "Export", 0, QApplication::UnicodeUTF8));
        buttonDelete->setText(QApplication::translate("ActionDockWidgetUI", "Delete Selection", 0, QApplication::UnicodeUTF8));
        buttonAnimation->setText(QApplication::translate("ActionDockWidgetUI", "Anmation", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ActionDockWidgetUI: public Ui_ActionDockWidgetUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // ACTIONDOCKWIDGETUI_H
