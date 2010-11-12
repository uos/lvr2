/********************************************************************************
** Form generated from reading UI file 'matrixdialog.ui'
**
** Created: Fri Nov 12 11:19:15 2010
**      by: Qt User Interface Compiler version 4.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MATRIXDIALOG_H
#define UI_MATRIXDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_MatrixDialog
{
public:
    QGroupBox *groupBox;
    QDoubleSpinBox *doubleSpinBox00;
    QDoubleSpinBox *doubleSpinBox01;
    QDoubleSpinBox *doubleSpinBox02;
    QDoubleSpinBox *doubleSpinBox03;
    QDoubleSpinBox *doubleSpinBox04;
    QDoubleSpinBox *doubleSpinBox05;
    QDoubleSpinBox *doubleSpinBox06;
    QDoubleSpinBox *doubleSpinBox07;
    QDoubleSpinBox *doubleSpinBox08;
    QDoubleSpinBox *doubleSpinBox09;
    QDoubleSpinBox *doubleSpinBox10;
    QDoubleSpinBox *doubleSpinBox11;
    QDoubleSpinBox *doubleSpinBox12;
    QDoubleSpinBox *doubleSpinBox13;
    QDoubleSpinBox *doubleSpinBox14;
    QDoubleSpinBox *doubleSpinBox15;
    QPushButton *pushButtonCancel;
    QPushButton *pushButtonOK;

    void setupUi(QDialog *MatrixDialog)
    {
        if (MatrixDialog->objectName().isEmpty())
            MatrixDialog->setObjectName(QString::fromUtf8("MatrixDialog"));
        MatrixDialog->resize(311, 196);
        groupBox = new QGroupBox(MatrixDialog);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(10, 0, 291, 141));
        doubleSpinBox00 = new QDoubleSpinBox(groupBox);
        doubleSpinBox00->setObjectName(QString::fromUtf8("doubleSpinBox00"));
        doubleSpinBox00->setGeometry(QRect(10, 20, 62, 22));
        doubleSpinBox00->setCursor(QCursor(Qt::WaitCursor));
        doubleSpinBox00->setDecimals(4);
        doubleSpinBox00->setMinimum(-1e+09);
        doubleSpinBox00->setMaximum(1e+09);
        doubleSpinBox00->setValue(1);
        doubleSpinBox01 = new QDoubleSpinBox(groupBox);
        doubleSpinBox01->setObjectName(QString::fromUtf8("doubleSpinBox01"));
        doubleSpinBox01->setGeometry(QRect(80, 20, 62, 22));
        doubleSpinBox01->setDecimals(4);
        doubleSpinBox01->setMinimum(-1e+09);
        doubleSpinBox01->setMaximum(1e+09);
        doubleSpinBox02 = new QDoubleSpinBox(groupBox);
        doubleSpinBox02->setObjectName(QString::fromUtf8("doubleSpinBox02"));
        doubleSpinBox02->setGeometry(QRect(150, 20, 62, 22));
        doubleSpinBox02->setDecimals(4);
        doubleSpinBox02->setMinimum(-1e+09);
        doubleSpinBox02->setMaximum(1e+09);
        doubleSpinBox03 = new QDoubleSpinBox(groupBox);
        doubleSpinBox03->setObjectName(QString::fromUtf8("doubleSpinBox03"));
        doubleSpinBox03->setGeometry(QRect(220, 20, 62, 22));
        doubleSpinBox03->setDecimals(4);
        doubleSpinBox03->setMinimum(-1e+09);
        doubleSpinBox03->setMaximum(1e+09);
        doubleSpinBox04 = new QDoubleSpinBox(groupBox);
        doubleSpinBox04->setObjectName(QString::fromUtf8("doubleSpinBox04"));
        doubleSpinBox04->setGeometry(QRect(10, 50, 62, 22));
        doubleSpinBox04->setDecimals(4);
        doubleSpinBox04->setMinimum(-1e+09);
        doubleSpinBox04->setMaximum(1e+09);
        doubleSpinBox05 = new QDoubleSpinBox(groupBox);
        doubleSpinBox05->setObjectName(QString::fromUtf8("doubleSpinBox05"));
        doubleSpinBox05->setGeometry(QRect(80, 50, 62, 22));
        doubleSpinBox05->setDecimals(4);
        doubleSpinBox05->setMinimum(-1e+09);
        doubleSpinBox05->setMaximum(1e+09);
        doubleSpinBox05->setValue(1);
        doubleSpinBox06 = new QDoubleSpinBox(groupBox);
        doubleSpinBox06->setObjectName(QString::fromUtf8("doubleSpinBox06"));
        doubleSpinBox06->setGeometry(QRect(150, 50, 62, 22));
        doubleSpinBox06->setDecimals(4);
        doubleSpinBox06->setMinimum(-1e+09);
        doubleSpinBox06->setMaximum(1e+09);
        doubleSpinBox07 = new QDoubleSpinBox(groupBox);
        doubleSpinBox07->setObjectName(QString::fromUtf8("doubleSpinBox07"));
        doubleSpinBox07->setGeometry(QRect(220, 50, 62, 22));
        doubleSpinBox07->setDecimals(4);
        doubleSpinBox07->setMinimum(-1e+09);
        doubleSpinBox07->setMaximum(1e+09);
        doubleSpinBox08 = new QDoubleSpinBox(groupBox);
        doubleSpinBox08->setObjectName(QString::fromUtf8("doubleSpinBox08"));
        doubleSpinBox08->setGeometry(QRect(10, 80, 62, 22));
        doubleSpinBox08->setDecimals(4);
        doubleSpinBox08->setMinimum(-1e+09);
        doubleSpinBox08->setMaximum(1e+09);
        doubleSpinBox09 = new QDoubleSpinBox(groupBox);
        doubleSpinBox09->setObjectName(QString::fromUtf8("doubleSpinBox09"));
        doubleSpinBox09->setGeometry(QRect(80, 80, 62, 22));
        doubleSpinBox09->setDecimals(4);
        doubleSpinBox09->setMinimum(-1e+09);
        doubleSpinBox09->setMaximum(1e+09);
        doubleSpinBox10 = new QDoubleSpinBox(groupBox);
        doubleSpinBox10->setObjectName(QString::fromUtf8("doubleSpinBox10"));
        doubleSpinBox10->setGeometry(QRect(150, 80, 62, 22));
        doubleSpinBox10->setDecimals(4);
        doubleSpinBox10->setMinimum(-1e+09);
        doubleSpinBox10->setMaximum(1e+09);
        doubleSpinBox10->setValue(1);
        doubleSpinBox11 = new QDoubleSpinBox(groupBox);
        doubleSpinBox11->setObjectName(QString::fromUtf8("doubleSpinBox11"));
        doubleSpinBox11->setGeometry(QRect(220, 80, 62, 21));
        doubleSpinBox11->setDecimals(4);
        doubleSpinBox11->setMinimum(-1e+09);
        doubleSpinBox11->setMaximum(1e+09);
        doubleSpinBox12 = new QDoubleSpinBox(groupBox);
        doubleSpinBox12->setObjectName(QString::fromUtf8("doubleSpinBox12"));
        doubleSpinBox12->setGeometry(QRect(10, 110, 62, 22));
        doubleSpinBox12->setDecimals(4);
        doubleSpinBox12->setMinimum(-1e+09);
        doubleSpinBox12->setMaximum(1e+09);
        doubleSpinBox13 = new QDoubleSpinBox(groupBox);
        doubleSpinBox13->setObjectName(QString::fromUtf8("doubleSpinBox13"));
        doubleSpinBox13->setGeometry(QRect(80, 110, 62, 22));
        doubleSpinBox13->setDecimals(4);
        doubleSpinBox13->setMinimum(-1e+09);
        doubleSpinBox13->setMaximum(1e+09);
        doubleSpinBox14 = new QDoubleSpinBox(groupBox);
        doubleSpinBox14->setObjectName(QString::fromUtf8("doubleSpinBox14"));
        doubleSpinBox14->setGeometry(QRect(150, 110, 62, 22));
        doubleSpinBox14->setDecimals(4);
        doubleSpinBox14->setMinimum(-1e+09);
        doubleSpinBox14->setMaximum(1e+09);
        doubleSpinBox15 = new QDoubleSpinBox(groupBox);
        doubleSpinBox15->setObjectName(QString::fromUtf8("doubleSpinBox15"));
        doubleSpinBox15->setGeometry(QRect(220, 110, 62, 22));
        doubleSpinBox15->setDecimals(4);
        doubleSpinBox15->setMinimum(-1e+09);
        doubleSpinBox15->setMaximum(1e+09);
        doubleSpinBox15->setValue(1);
        pushButtonCancel = new QPushButton(MatrixDialog);
        pushButtonCancel->setObjectName(QString::fromUtf8("pushButtonCancel"));
        pushButtonCancel->setGeometry(QRect(110, 160, 81, 26));
        pushButtonOK = new QPushButton(MatrixDialog);
        pushButtonOK->setObjectName(QString::fromUtf8("pushButtonOK"));
        pushButtonOK->setGeometry(QRect(10, 160, 91, 26));

        retranslateUi(MatrixDialog);
        QObject::connect(pushButtonOK, SIGNAL(pressed()), MatrixDialog, SLOT(accept()));
        QObject::connect(pushButtonCancel, SIGNAL(pressed()), MatrixDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(MatrixDialog);
    } // setupUi

    void retranslateUi(QDialog *MatrixDialog)
    {
        MatrixDialog->setWindowTitle(QApplication::translate("MatrixDialog", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("MatrixDialog", "Enter Transformation Matrix:", 0, QApplication::UnicodeUTF8));
        pushButtonCancel->setText(QApplication::translate("MatrixDialog", "Cancel", 0, QApplication::UnicodeUTF8));
        pushButtonOK->setText(QApplication::translate("MatrixDialog", "OK", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MatrixDialog: public Ui_MatrixDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MATRIXDIALOG_H
