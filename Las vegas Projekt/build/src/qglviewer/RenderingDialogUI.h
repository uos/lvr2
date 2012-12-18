/********************************************************************************
** Form generated from reading UI file 'RenderingDialogUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef RENDERINGDIALOGUI_H
#define RENDERINGDIALOGUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QSpinBox>

QT_BEGIN_NAMESPACE

class Ui_RenderingDialogUI
{
public:
    QDialogButtonBox *buttonBox;
    QGroupBox *groupBox_2;
    QComboBox *comboBox;
    QLabel *label_3;
    QSpinBox *spinBoxBuckets;
    QGroupBox *groupBox_3;
    QCheckBox *checkBoxTwoSided;
    QCheckBox *checkBox;
    QGroupBox *groupBox;
    QLabel *label;
    QLabel *label_2;
    QDoubleSpinBox *spinBoxLineWidth;
    QDoubleSpinBox *spinBoxPointSize;

    void setupUi(QDialog *RenderingDialogUI)
    {
        if (RenderingDialogUI->objectName().isEmpty())
            RenderingDialogUI->setObjectName(QString::fromUtf8("RenderingDialogUI"));
        RenderingDialogUI->resize(400, 244);
        buttonBox = new QDialogButtonBox(RenderingDialogUI);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(50, 200, 341, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        groupBox_2 = new QGroupBox(RenderingDialogUI);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setEnabled(false);
        groupBox_2->setGeometry(QRect(20, 110, 351, 61));
        comboBox = new QComboBox(groupBox_2);
        comboBox->setObjectName(QString::fromUtf8("comboBox"));
        comboBox->setGeometry(QRect(0, 30, 221, 27));
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(0, 20, 121, 17));
        spinBoxBuckets = new QSpinBox(groupBox_2);
        spinBoxBuckets->setObjectName(QString::fromUtf8("spinBoxBuckets"));
        spinBoxBuckets->setGeometry(QRect(230, 30, 101, 27));
        spinBoxBuckets->setLayoutDirection(Qt::LeftToRight);
        groupBox_3 = new QGroupBox(RenderingDialogUI);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        groupBox_3->setEnabled(false);
        groupBox_3->setGeometry(QRect(210, 10, 141, 81));
        checkBoxTwoSided = new QCheckBox(groupBox_3);
        checkBoxTwoSided->setObjectName(QString::fromUtf8("checkBoxTwoSided"));
        checkBoxTwoSided->setGeometry(QRect(0, 30, 131, 22));
        checkBox = new QCheckBox(groupBox_3);
        checkBox->setObjectName(QString::fromUtf8("checkBox"));
        checkBox->setGeometry(QRect(0, 50, 141, 41));
        groupBox = new QGroupBox(RenderingDialogUI);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(20, 10, 171, 101));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(70, 60, 81, 17));
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(70, 30, 81, 17));
        spinBoxLineWidth = new QDoubleSpinBox(groupBox);
        spinBoxLineWidth->setObjectName(QString::fromUtf8("spinBoxLineWidth"));
        spinBoxLineWidth->setGeometry(QRect(0, 30, 62, 27));
        spinBoxPointSize = new QDoubleSpinBox(groupBox);
        spinBoxPointSize->setObjectName(QString::fromUtf8("spinBoxPointSize"));
        spinBoxPointSize->setGeometry(QRect(0, 60, 62, 27));

        retranslateUi(RenderingDialogUI);
        QObject::connect(buttonBox, SIGNAL(accepted()), RenderingDialogUI, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), RenderingDialogUI, SLOT(reject()));

        QMetaObject::connectSlotsByName(RenderingDialogUI);
    } // setupUi

    void retranslateUi(QDialog *RenderingDialogUI)
    {
        RenderingDialogUI->setWindowTitle(QApplication::translate("RenderingDialogUI", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("RenderingDialogUI", "Color Mapping", 0, QApplication::UnicodeUTF8));
        label_3->setText(QString());
        groupBox_3->setTitle(QApplication::translate("RenderingDialogUI", "Rendering", 0, QApplication::UnicodeUTF8));
        checkBoxTwoSided->setText(QApplication::translate("RenderingDialogUI", "Two sided light", 0, QApplication::UnicodeUTF8));
        checkBox->setText(QApplication::translate("RenderingDialogUI", "Texture Mapping", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("RenderingDialogUI", "Points and Lines", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("RenderingDialogUI", "Point Size", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("RenderingDialogUI", "Line Width", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class RenderingDialogUI: public Ui_RenderingDialogUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // RENDERINGDIALOGUI_H
