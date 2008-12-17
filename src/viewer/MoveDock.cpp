/********************************************************************************
** Form generated from reading ui file 'movedock.ui'
**
** Created: Wed Dec 17 11:01:03 2008
**      by: Qt User Interface Compiler version 4.4.3
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef MOVEDOCK_H
#define MOVEDOCK_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDockWidget>
#include <QtGui/QFrame>
#include <QtGui/QGroupBox>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MoveDock
{
public:
    QWidget *dockWidgetContents;
    QFrame *touchFrame;
    QGroupBox *groupBoxAction;
    QComboBox *comboBoxAction;

    void setupUi(QDockWidget *MoveDock)
    {
    if (MoveDock->objectName().isEmpty())
        MoveDock->setObjectName(QString::fromUtf8("MoveDock"));
    MoveDock->resize(200, 232);
    MoveDock->setMinimumSize(QSize(200, 0));
    MoveDock->setFeatures(QDockWidget::DockWidgetFeatureMask);
    MoveDock->setAllowedAreas(Qt::LeftDockWidgetArea);
    dockWidgetContents = new QWidget();
    dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
    dockWidgetContents->setGeometry(QRect(20, 0, 180, 232));
    touchFrame = new QFrame(dockWidgetContents);
    touchFrame->setObjectName(QString::fromUtf8("touchFrame"));
    touchFrame->setGeometry(QRect(0, 10, 171, 151));
    touchFrame->setAutoFillBackground(false);
    touchFrame->setFrameShape(QFrame::StyledPanel);
    touchFrame->setFrameShadow(QFrame::Sunken);
    groupBoxAction = new QGroupBox(dockWidgetContents);
    groupBoxAction->setObjectName(QString::fromUtf8("groupBoxAction"));
    groupBoxAction->setGeometry(QRect(0, 170, 171, 51));
    comboBoxAction = new QComboBox(groupBoxAction);
    comboBoxAction->setObjectName(QString::fromUtf8("comboBoxAction"));
    comboBoxAction->setGeometry(QRect(10, 20, 151, 22));
    MoveDock->setWidget(dockWidgetContents);

    retranslateUi(MoveDock);

    QMetaObject::connectSlotsByName(MoveDock);
    } // setupUi

    void retranslateUi(QDockWidget *MoveDock)
    {
    MoveDock->setWindowTitle(QApplication::translate("MoveDock", "Rotate Object", 0, QApplication::UnicodeUTF8));
    groupBoxAction->setTitle(QApplication::translate("MoveDock", "Action", 0, QApplication::UnicodeUTF8));
    Q_UNUSED(MoveDock);
    } // retranslateUi

};

namespace Ui {
    class MoveDock: public Ui_MoveDock {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MOVEDOCK_H
