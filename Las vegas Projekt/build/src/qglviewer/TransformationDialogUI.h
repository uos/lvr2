/********************************************************************************
** Form generated from reading UI file 'TransformationDialogUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef TRANSFORMATIONDIALOGUI_H
#define TRANSFORMATIONDIALOGUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>

QT_BEGIN_NAMESPACE

class Ui_TransformationDialogUI
{
public:
    QDialogButtonBox *buttonBox;
    QGroupBox *groupBoxRotation;
    QSlider *sliderXRot;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QDoubleSpinBox *spinBoxXRot;
    QLabel *label_6;
    QLabel *label_4;
    QDoubleSpinBox *spinBoxYRot;
    QLabel *label_5;
    QSlider *sliderYRot;
    QLabel *label_7;
    QDoubleSpinBox *spinBoxZRot;
    QLabel *label_8;
    QSlider *sliderZRot;
    QLabel *label_9;
    QGroupBox *groupBoxTranslation;
    QLabel *label_10;
    QDoubleSpinBox *spinBoxXTrans;
    QLabel *label_11;
    QLabel *label_12;
    QDoubleSpinBox *spinBoxYTrans;
    QDoubleSpinBox *spinBoxZTrans;
    QPushButton *buttonReset;
    QPushButton *buttonSave;
    QGroupBox *groupBox;
    QDoubleSpinBox *spinBoxStep;

    void setupUi(QDialog *TransformationDialogUI)
    {
        if (TransformationDialogUI->objectName().isEmpty())
            TransformationDialogUI->setObjectName(QString::fromUtf8("TransformationDialogUI"));
        TransformationDialogUI->resize(407, 277);
        buttonBox = new QDialogButtonBox(TransformationDialogUI);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(209, 242, 191, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        groupBoxRotation = new QGroupBox(TransformationDialogUI);
        groupBoxRotation->setObjectName(QString::fromUtf8("groupBoxRotation"));
        groupBoxRotation->setGeometry(QRect(10, 10, 393, 112));
        groupBoxRotation->setFlat(false);
        groupBoxRotation->setCheckable(false);
        sliderXRot = new QSlider(groupBoxRotation);
        sliderXRot->setObjectName(QString::fromUtf8("sliderXRot"));
        sliderXRot->setGeometry(QRect(90, 26, 160, 29));
        sliderXRot->setMinimum(-18000000);
        sliderXRot->setMaximum(18000000);
        sliderXRot->setTracking(true);
        sliderXRot->setOrientation(Qt::Horizontal);
        sliderXRot->setInvertedAppearance(false);
        sliderXRot->setInvertedControls(false);
        sliderXRot->setTickPosition(QSlider::NoTicks);
        sliderXRot->setTickInterval(10);
        label = new QLabel(groupBoxRotation);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(10, 30, 21, 17));
        label_2 = new QLabel(groupBoxRotation);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(40, 30, 41, 17));
        label_3 = new QLabel(groupBoxRotation);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(260, 30, 41, 17));
        spinBoxXRot = new QDoubleSpinBox(groupBoxRotation);
        spinBoxXRot->setObjectName(QString::fromUtf8("spinBoxXRot"));
        spinBoxXRot->setGeometry(QRect(310, 30, 75, 21));
        spinBoxXRot->setMinimum(-180);
        spinBoxXRot->setMaximum(180);
        spinBoxXRot->setSingleStep(1);
        label_6 = new QLabel(groupBoxRotation);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setGeometry(QRect(40, 60, 41, 17));
        label_4 = new QLabel(groupBoxRotation);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setGeometry(QRect(260, 60, 41, 17));
        spinBoxYRot = new QDoubleSpinBox(groupBoxRotation);
        spinBoxYRot->setObjectName(QString::fromUtf8("spinBoxYRot"));
        spinBoxYRot->setGeometry(QRect(310, 60, 75, 21));
        spinBoxYRot->setMinimum(-180);
        spinBoxYRot->setMaximum(180);
        spinBoxYRot->setSingleStep(0.1);
        label_5 = new QLabel(groupBoxRotation);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setGeometry(QRect(10, 60, 16, 17));
        sliderYRot = new QSlider(groupBoxRotation);
        sliderYRot->setObjectName(QString::fromUtf8("sliderYRot"));
        sliderYRot->setGeometry(QRect(91, 55, 160, 29));
        sliderYRot->setMinimum(-18000000);
        sliderYRot->setMaximum(18000000);
        sliderYRot->setTracking(false);
        sliderYRot->setOrientation(Qt::Horizontal);
        sliderYRot->setInvertedAppearance(false);
        sliderYRot->setInvertedControls(false);
        sliderYRot->setTickPosition(QSlider::NoTicks);
        sliderYRot->setTickInterval(10);
        label_7 = new QLabel(groupBoxRotation);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setGeometry(QRect(260, 90, 41, 17));
        spinBoxZRot = new QDoubleSpinBox(groupBoxRotation);
        spinBoxZRot->setObjectName(QString::fromUtf8("spinBoxZRot"));
        spinBoxZRot->setGeometry(QRect(310, 90, 75, 21));
        spinBoxZRot->setMinimum(-180);
        spinBoxZRot->setMaximum(180);
        spinBoxZRot->setSingleStep(0.1);
        label_8 = new QLabel(groupBoxRotation);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setGeometry(QRect(10, 90, 21, 17));
        sliderZRot = new QSlider(groupBoxRotation);
        sliderZRot->setObjectName(QString::fromUtf8("sliderZRot"));
        sliderZRot->setGeometry(QRect(91, 84, 160, 29));
        sliderZRot->setMinimum(-18000000);
        sliderZRot->setMaximum(18000000);
        sliderZRot->setTracking(false);
        sliderZRot->setOrientation(Qt::Horizontal);
        sliderZRot->setInvertedAppearance(false);
        sliderZRot->setInvertedControls(false);
        sliderZRot->setTickPosition(QSlider::NoTicks);
        sliderZRot->setTickInterval(10);
        label_9 = new QLabel(groupBoxRotation);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setGeometry(QRect(40, 90, 41, 17));
        groupBoxTranslation = new QGroupBox(TransformationDialogUI);
        groupBoxTranslation->setObjectName(QString::fromUtf8("groupBoxTranslation"));
        groupBoxTranslation->setGeometry(QRect(7, 123, 381, 62));
        label_10 = new QLabel(groupBoxTranslation);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setGeometry(QRect(10, 30, 21, 17));
        spinBoxXTrans = new QDoubleSpinBox(groupBoxTranslation);
        spinBoxXTrans->setObjectName(QString::fromUtf8("spinBoxXTrans"));
        spinBoxXTrans->setGeometry(QRect(30, 30, 89, 21));
        spinBoxXTrans->setMinimum(-1e+06);
        spinBoxXTrans->setMaximum(1e+06);
        spinBoxXTrans->setSingleStep(0.01);
        label_11 = new QLabel(groupBoxTranslation);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setGeometry(QRect(130, 30, 21, 17));
        label_12 = new QLabel(groupBoxTranslation);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        label_12->setGeometry(QRect(250, 30, 21, 17));
        spinBoxYTrans = new QDoubleSpinBox(groupBoxTranslation);
        spinBoxYTrans->setObjectName(QString::fromUtf8("spinBoxYTrans"));
        spinBoxYTrans->setGeometry(QRect(150, 30, 97, 21));
        spinBoxYTrans->setMinimum(-1e+06);
        spinBoxYTrans->setMaximum(1e+06);
        spinBoxYTrans->setSingleStep(0.01);
        spinBoxZTrans = new QDoubleSpinBox(groupBoxTranslation);
        spinBoxZTrans->setObjectName(QString::fromUtf8("spinBoxZTrans"));
        spinBoxZTrans->setGeometry(QRect(270, 30, 101, 21));
        spinBoxZTrans->setMinimum(-1e+06);
        spinBoxZTrans->setMaximum(1e+06);
        spinBoxZTrans->setSingleStep(0.01);
        buttonReset = new QPushButton(TransformationDialogUI);
        buttonReset->setObjectName(QString::fromUtf8("buttonReset"));
        buttonReset->setGeometry(QRect(5, 246, 97, 27));
        buttonSave = new QPushButton(TransformationDialogUI);
        buttonSave->setObjectName(QString::fromUtf8("buttonSave"));
        buttonSave->setGeometry(QRect(108, 244, 97, 27));
        groupBox = new QGroupBox(TransformationDialogUI);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(7, 185, 120, 48));
        spinBoxStep = new QDoubleSpinBox(groupBox);
        spinBoxStep->setObjectName(QString::fromUtf8("spinBoxStep"));
        spinBoxStep->setGeometry(QRect(5, 23, 87, 21));
        spinBoxStep->setMinimum(-1e+06);
        spinBoxStep->setMaximum(1e+06);
        spinBoxStep->setSingleStep(0.01);
        spinBoxStep->setValue(0.01);

        retranslateUi(TransformationDialogUI);
        QObject::connect(buttonBox, SIGNAL(accepted()), TransformationDialogUI, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), TransformationDialogUI, SLOT(reject()));

        QMetaObject::connectSlotsByName(TransformationDialogUI);
    } // setupUi

    void retranslateUi(QDialog *TransformationDialogUI)
    {
        TransformationDialogUI->setWindowTitle(QApplication::translate("TransformationDialogUI", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBoxRotation->setTitle(QApplication::translate("TransformationDialogUI", "Rotation:", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("TransformationDialogUI", "X:", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("TransformationDialogUI", "-180 \302\260", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("TransformationDialogUI", "+180 \302\260", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("TransformationDialogUI", "-180 \302\260", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("TransformationDialogUI", "+180 \302\260", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("TransformationDialogUI", "Y:", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("TransformationDialogUI", "+180 \302\260", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("TransformationDialogUI", "Z:", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("TransformationDialogUI", "-180 \302\260", 0, QApplication::UnicodeUTF8));
        groupBoxTranslation->setTitle(QApplication::translate("TransformationDialogUI", "Translation:", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("TransformationDialogUI", "X:", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("TransformationDialogUI", "Y:", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("TransformationDialogUI", "Z:", 0, QApplication::UnicodeUTF8));
        buttonReset->setText(QApplication::translate("TransformationDialogUI", "Reset", 0, QApplication::UnicodeUTF8));
        buttonSave->setText(QApplication::translate("TransformationDialogUI", "Save...", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("TransformationDialogUI", "Step:", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class TransformationDialogUI: public Ui_TransformationDialogUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // TRANSFORMATIONDIALOGUI_H
