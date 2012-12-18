/********************************************************************************
** Form generated from reading UI file 'DebugOutputDialogUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef DEBUGOUTPUTDIALOGUI_H
#define DEBUGOUTPUTDIALOGUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QHeaderView>
#include <QtGui/QPlainTextEdit>

QT_BEGIN_NAMESPACE

class Ui_DebugOutputDialogUI
{
public:
    QPlainTextEdit *plainTextEdit;

    void setupUi(QDialog *DebugOutputDialogUI)
    {
        if (DebugOutputDialogUI->objectName().isEmpty())
            DebugOutputDialogUI->setObjectName(QString::fromUtf8("DebugOutputDialogUI"));
        DebugOutputDialogUI->resize(864, 300);
        plainTextEdit = new QPlainTextEdit(DebugOutputDialogUI);
        plainTextEdit->setObjectName(QString::fromUtf8("plainTextEdit"));
        plainTextEdit->setGeometry(QRect(10, 10, 841, 281));

        retranslateUi(DebugOutputDialogUI);

        QMetaObject::connectSlotsByName(DebugOutputDialogUI);
    } // setupUi

    void retranslateUi(QDialog *DebugOutputDialogUI)
    {
        DebugOutputDialogUI->setWindowTitle(QApplication::translate("DebugOutputDialogUI", "Dialog", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class DebugOutputDialogUI: public Ui_DebugOutputDialogUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // DEBUGOUTPUTDIALOGUI_H
