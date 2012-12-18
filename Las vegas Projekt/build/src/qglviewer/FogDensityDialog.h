/********************************************************************************
** Form generated from reading UI file 'FogDensityDialog.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef FOGDENSITYDIALOG_H
#define FOGDENSITYDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>

QT_BEGIN_NAMESPACE

class Ui_Fogsettings
{
public:
    QGroupBox *groupBox;
    QRadioButton *radioButtonLinear;
    QRadioButton *radioButtonExp;
    QRadioButton *radioButtonExp2;
    QGroupBox *groupBox_2;
    QSlider *sliderDensity;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label;
    QPushButton *pushButtonOK;

    void setupUi(QDialog *Fogsettings)
    {
        if (Fogsettings->objectName().isEmpty())
            Fogsettings->setObjectName(QString::fromUtf8("Fogsettings"));
        Fogsettings->resize(350, 140);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(Fogsettings->sizePolicy().hasHeightForWidth());
        Fogsettings->setSizePolicy(sizePolicy);
        Fogsettings->setMinimumSize(QSize(350, 140));
        Fogsettings->setMaximumSize(QSize(350, 140));
        groupBox = new QGroupBox(Fogsettings);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 231, 51));
        radioButtonLinear = new QRadioButton(groupBox);
        radioButtonLinear->setObjectName(QString::fromUtf8("radioButtonLinear"));
        radioButtonLinear->setGeometry(QRect(10, 20, 109, 31));
        radioButtonLinear->setChecked(true);
        radioButtonExp = new QRadioButton(groupBox);
        radioButtonExp->setObjectName(QString::fromUtf8("radioButtonExp"));
        radioButtonExp->setGeometry(QRect(90, 26, 71, 21));
        radioButtonExp2 = new QRadioButton(groupBox);
        radioButtonExp2->setObjectName(QString::fromUtf8("radioButtonExp2"));
        radioButtonExp2->setGeometry(QRect(160, 25, 71, 21));
        groupBox_2 = new QGroupBox(Fogsettings);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 60, 231, 71));
        groupBox_2->setAutoFillBackground(false);
        groupBox_2->setFlat(true);
        sliderDensity = new QSlider(groupBox_2);
        sliderDensity->setObjectName(QString::fromUtf8("sliderDensity"));
        sliderDensity->setGeometry(QRect(10, 30, 211, 19));
        sliderDensity->setMaximum(2000);
        sliderDensity->setValue(1000);
        sliderDensity->setOrientation(Qt::Horizontal);
        sliderDensity->setInvertedAppearance(false);
        sliderDensity->setTickPosition(QSlider::TicksBelow);
        sliderDensity->setTickInterval(100);
        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(100, 50, 21, 17));
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(200, 50, 21, 17));
        label = new QLabel(groupBox_2);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(0, 50, 21, 17));
        pushButtonOK = new QPushButton(Fogsettings);
        pushButtonOK->setObjectName(QString::fromUtf8("pushButtonOK"));
        pushButtonOK->setGeometry(QRect(250, 10, 93, 27));

        retranslateUi(Fogsettings);
        QObject::connect(pushButtonOK, SIGNAL(clicked()), Fogsettings, SLOT(accept()));

        QMetaObject::connectSlotsByName(Fogsettings);
    } // setupUi

    void retranslateUi(QDialog *Fogsettings)
    {
        Fogsettings->setWindowTitle(QApplication::translate("Fogsettings", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("Fogsettings", "Fog Type", 0, QApplication::UnicodeUTF8));
        radioButtonLinear->setText(QApplication::translate("Fogsettings", "Linear", 0, QApplication::UnicodeUTF8));
        radioButtonExp->setText(QApplication::translate("Fogsettings", "Exp", 0, QApplication::UnicodeUTF8));
        radioButtonExp2->setText(QApplication::translate("Fogsettings", "Exp2", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("Fogsettings", "Fog Density", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("Fogsettings", "1.0", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("Fogsettings", "2.0", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("Fogsettings", "0.0", 0, QApplication::UnicodeUTF8));
        pushButtonOK->setText(QApplication::translate("Fogsettings", "OK", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class Fogsettings: public Ui_Fogsettings {};
} // namespace Ui

QT_END_NAMESPACE

#endif // FOGDENSITYDIALOG_H
