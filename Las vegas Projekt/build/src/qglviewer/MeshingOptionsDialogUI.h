/********************************************************************************
** Form generated from reading UI file 'MeshingOptionsDialogUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MESHINGOPTIONSDIALOGUI_H
#define MESHINGOPTIONSDIALOGUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFrame>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QSpinBox>

QT_BEGIN_NAMESPACE

class Ui_MeshingOptionsDialogUI
{
public:
    QDialogButtonBox *buttonBox;
    QGroupBox *groupBox;
    QDoubleSpinBox *spinBoxVoxelsize;
    QLabel *label;
    QLabel *label_2;
    QComboBox *comboBoxPCM;
    QLabel *label_3;
    QLabel *label_4;
    QSpinBox *spinBoxKd;
    QGroupBox *groupBox_2;
    QSpinBox *spinBoxKn;
    QSpinBox *spinBoxKi;
    QLabel *label_5;
    QLabel *label_6;
    QCheckBox *checkBoxRecalcNormals;
    QGroupBox *groupBox_3;
    QCheckBox *checkBoxOptimizePlanes;
    QLabel *label_7;
    QLabel *label_8;
    QSpinBox *spinBoxPlaneIterations;
    QDoubleSpinBox *spinBoxNormalThr;
    QLabel *label_9;
    QCheckBox *checkBoxFillHoles;
    QLabel *label_10;
    QDoubleSpinBox *spinBoxHoleSize;
    QDoubleSpinBox *spinBoxMinPlaneSize;
    QCheckBox *checkBoxRDA;
    QLabel *label_11;
    QSpinBox *spinBoxRDA;
    QCheckBox *checkBoxRemoveRegions;
    QLabel *label_12;
    QSpinBox *spinBoxRemoveRegions;
    QCheckBox *checkBoxRetesselate;
    QGroupBox *groupBox_4;
    QCheckBox *checkBoxColorRegions;
    QCheckBox *checkBoxGenerateTextures;
    QFrame *line;

    void setupUi(QDialog *MeshingOptionsDialogUI)
    {
        if (MeshingOptionsDialogUI->objectName().isEmpty())
            MeshingOptionsDialogUI->setObjectName(QString::fromUtf8("MeshingOptionsDialogUI"));
        MeshingOptionsDialogUI->resize(689, 411);
        buttonBox = new QDialogButtonBox(MeshingOptionsDialogUI);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(330, 370, 341, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        groupBox = new QGroupBox(MeshingOptionsDialogUI);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 321, 121));
        spinBoxVoxelsize = new QDoubleSpinBox(groupBox);
        spinBoxVoxelsize->setObjectName(QString::fromUtf8("spinBoxVoxelsize"));
        spinBoxVoxelsize->setGeometry(QRect(190, 30, 131, 27));
        spinBoxVoxelsize->setDecimals(6);
        spinBoxVoxelsize->setValue(5);
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(0, 30, 101, 17));
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(0, 100, 151, 17));
        comboBoxPCM = new QComboBox(groupBox);
        comboBoxPCM->setObjectName(QString::fromUtf8("comboBoxPCM"));
        comboBoxPCM->setGeometry(QRect(190, 90, 131, 27));
        comboBoxPCM->setMaxVisibleItems(1);
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(0, 90, 67, 17));
        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setGeometry(QRect(0, 60, 131, 17));
        spinBoxKd = new QSpinBox(groupBox);
        spinBoxKd->setObjectName(QString::fromUtf8("spinBoxKd"));
        spinBoxKd->setGeometry(QRect(190, 60, 131, 27));
        spinBoxKd->setMaximum(100000);
        spinBoxKd->setValue(5);
        groupBox_2 = new QGroupBox(MeshingOptionsDialogUI);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 150, 321, 111));
        spinBoxKn = new QSpinBox(groupBox_2);
        spinBoxKn->setObjectName(QString::fromUtf8("spinBoxKn"));
        spinBoxKn->setGeometry(QRect(250, 20, 71, 27));
        spinBoxKn->setMaximum(10000);
        spinBoxKn->setValue(10);
        spinBoxKi = new QSpinBox(groupBox_2);
        spinBoxKi->setObjectName(QString::fromUtf8("spinBoxKi"));
        spinBoxKi->setGeometry(QRect(250, 50, 71, 27));
        spinBoxKi->setMaximum(10000);
        spinBoxKi->setValue(10);
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setGeometry(QRect(0, 30, 181, 17));
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setGeometry(QRect(0, 60, 201, 17));
        checkBoxRecalcNormals = new QCheckBox(groupBox_2);
        checkBoxRecalcNormals->setObjectName(QString::fromUtf8("checkBoxRecalcNormals"));
        checkBoxRecalcNormals->setGeometry(QRect(10, 90, 171, 22));
        checkBoxRecalcNormals->setChecked(true);
        groupBox_3 = new QGroupBox(MeshingOptionsDialogUI);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        groupBox_3->setGeometry(QRect(360, 10, 321, 351));
        checkBoxOptimizePlanes = new QCheckBox(groupBox_3);
        checkBoxOptimizePlanes->setObjectName(QString::fromUtf8("checkBoxOptimizePlanes"));
        checkBoxOptimizePlanes->setGeometry(QRect(10, 30, 231, 22));
        checkBoxOptimizePlanes->setChecked(true);
        label_7 = new QLabel(groupBox_3);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setGeometry(QRect(40, 60, 201, 17));
        label_8 = new QLabel(groupBox_3);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setGeometry(QRect(40, 90, 161, 17));
        spinBoxPlaneIterations = new QSpinBox(groupBox_3);
        spinBoxPlaneIterations->setObjectName(QString::fromUtf8("spinBoxPlaneIterations"));
        spinBoxPlaneIterations->setGeometry(QRect(240, 80, 81, 27));
        spinBoxPlaneIterations->setMaximum(10000);
        spinBoxPlaneIterations->setValue(3);
        spinBoxNormalThr = new QDoubleSpinBox(groupBox_3);
        spinBoxNormalThr->setObjectName(QString::fromUtf8("spinBoxNormalThr"));
        spinBoxNormalThr->setGeometry(QRect(240, 110, 81, 27));
        spinBoxNormalThr->setDecimals(5);
        spinBoxNormalThr->setMaximum(1);
        spinBoxNormalThr->setSingleStep(0.001);
        spinBoxNormalThr->setValue(0.85);
        label_9 = new QLabel(groupBox_3);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setGeometry(QRect(40, 120, 151, 17));
        checkBoxFillHoles = new QCheckBox(groupBox_3);
        checkBoxFillHoles->setObjectName(QString::fromUtf8("checkBoxFillHoles"));
        checkBoxFillHoles->setGeometry(QRect(10, 150, 97, 22));
        checkBoxFillHoles->setChecked(true);
        label_10 = new QLabel(groupBox_3);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setGeometry(QRect(40, 180, 131, 17));
        spinBoxHoleSize = new QDoubleSpinBox(groupBox_3);
        spinBoxHoleSize->setObjectName(QString::fromUtf8("spinBoxHoleSize"));
        spinBoxHoleSize->setGeometry(QRect(240, 180, 81, 27));
        spinBoxHoleSize->setMaximum(10000);
        spinBoxHoleSize->setValue(30);
        spinBoxMinPlaneSize = new QDoubleSpinBox(groupBox_3);
        spinBoxMinPlaneSize->setObjectName(QString::fromUtf8("spinBoxMinPlaneSize"));
        spinBoxMinPlaneSize->setGeometry(QRect(240, 50, 81, 27));
        spinBoxMinPlaneSize->setDecimals(3);
        spinBoxMinPlaneSize->setSingleStep(1);
        spinBoxMinPlaneSize->setValue(7);
        checkBoxRDA = new QCheckBox(groupBox_3);
        checkBoxRDA->setObjectName(QString::fromUtf8("checkBoxRDA"));
        checkBoxRDA->setGeometry(QRect(10, 210, 221, 22));
        checkBoxRDA->setChecked(true);
        label_11 = new QLabel(groupBox_3);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setGeometry(QRect(40, 240, 191, 17));
        spinBoxRDA = new QSpinBox(groupBox_3);
        spinBoxRDA->setObjectName(QString::fromUtf8("spinBoxRDA"));
        spinBoxRDA->setGeometry(QRect(240, 240, 81, 27));
        spinBoxRDA->setMaximum(10000);
        spinBoxRDA->setValue(100);
        checkBoxRemoveRegions = new QCheckBox(groupBox_3);
        checkBoxRemoveRegions->setObjectName(QString::fromUtf8("checkBoxRemoveRegions"));
        checkBoxRemoveRegions->setGeometry(QRect(10, 270, 261, 22));
        label_12 = new QLabel(groupBox_3);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        label_12->setGeometry(QRect(40, 300, 191, 17));
        spinBoxRemoveRegions = new QSpinBox(groupBox_3);
        spinBoxRemoveRegions->setObjectName(QString::fromUtf8("spinBoxRemoveRegions"));
        spinBoxRemoveRegions->setGeometry(QRect(240, 290, 81, 27));
        spinBoxRemoveRegions->setMaximum(10000);
        spinBoxRemoveRegions->setValue(30);
        checkBoxRetesselate = new QCheckBox(groupBox_3);
        checkBoxRetesselate->setObjectName(QString::fromUtf8("checkBoxRetesselate"));
        checkBoxRetesselate->setGeometry(QRect(10, 330, 261, 22));
        checkBoxRetesselate->setChecked(true);
        groupBox_4 = new QGroupBox(MeshingOptionsDialogUI);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        groupBox_4->setGeometry(QRect(10, 270, 331, 80));
        checkBoxColorRegions = new QCheckBox(groupBox_4);
        checkBoxColorRegions->setObjectName(QString::fromUtf8("checkBoxColorRegions"));
        checkBoxColorRegions->setGeometry(QRect(10, 30, 281, 22));
        checkBoxColorRegions->setChecked(true);
        checkBoxGenerateTextures = new QCheckBox(groupBox_4);
        checkBoxGenerateTextures->setObjectName(QString::fromUtf8("checkBoxGenerateTextures"));
        checkBoxGenerateTextures->setGeometry(QRect(10, 60, 281, 22));
        checkBoxGenerateTextures->setChecked(true);
        line = new QFrame(MeshingOptionsDialogUI);
        line->setObjectName(QString::fromUtf8("line"));
        line->setGeometry(QRect(10, 360, 671, 20));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        retranslateUi(MeshingOptionsDialogUI);
        QObject::connect(buttonBox, SIGNAL(accepted()), MeshingOptionsDialogUI, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), MeshingOptionsDialogUI, SLOT(reject()));

        QMetaObject::connectSlotsByName(MeshingOptionsDialogUI);
    } // setupUi

    void retranslateUi(QDialog *MeshingOptionsDialogUI)
    {
        MeshingOptionsDialogUI->setWindowTitle(QApplication::translate("MeshingOptionsDialogUI", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("MeshingOptionsDialogUI", "General Options", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MeshingOptionsDialogUI", "Voxelsize", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MeshingOptionsDialogUI", "Point Cloud Manager", 0, QApplication::UnicodeUTF8));
        comboBoxPCM->clear();
        comboBoxPCM->insertItems(0, QStringList()
         << QApplication::translate("MeshingOptionsDialogUI", "STANN", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MeshingOptionsDialogUI", "PCL", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MeshingOptionsDialogUI", "FLANN", 0, QApplication::UnicodeUTF8)
        );
        label_3->setText(QString());
        label_4->setText(QApplication::translate("MeshingOptionsDialogUI", "Distance Values", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("MeshingOptionsDialogUI", "Normal Estimation", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("MeshingOptionsDialogUI", "Num Points for Estimation", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("MeshingOptionsDialogUI", "Num Points for Interpolation", 0, QApplication::UnicodeUTF8));
        checkBoxRecalcNormals->setText(QApplication::translate("MeshingOptionsDialogUI", "Recalc Normals", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("MeshingOptionsDialogUI", "Mesh Optimization", 0, QApplication::UnicodeUTF8));
        checkBoxOptimizePlanes->setText(QApplication::translate("MeshingOptionsDialogUI", "Optimize Planes", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("MeshingOptionsDialogUI", "Minimum plane size", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("MeshingOptionsDialogUI", "Number of Iterations", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("MeshingOptionsDialogUI", "Normal Threshold", 0, QApplication::UnicodeUTF8));
        checkBoxFillHoles->setText(QApplication::translate("MeshingOptionsDialogUI", "Fill Holes", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("MeshingOptionsDialogUI", "Maximum hole size", 0, QApplication::UnicodeUTF8));
        checkBoxRDA->setText(QApplication::translate("MeshingOptionsDialogUI", "Remove Dangling Artifacts", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("MeshingOptionsDialogUI", "Artifacts to remove", 0, QApplication::UnicodeUTF8));
        checkBoxRemoveRegions->setText(QApplication::translate("MeshingOptionsDialogUI", "Remove small regions", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("MeshingOptionsDialogUI", "Regions to remove", 0, QApplication::UnicodeUTF8));
        checkBoxRetesselate->setText(QApplication::translate("MeshingOptionsDialogUI", "Retesselate", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("MeshingOptionsDialogUI", "Coloring / Texture Generation", 0, QApplication::UnicodeUTF8));
        checkBoxColorRegions->setText(QApplication::translate("MeshingOptionsDialogUI", "Color Regions", 0, QApplication::UnicodeUTF8));
        checkBoxGenerateTextures->setText(QApplication::translate("MeshingOptionsDialogUI", "Generate Textures", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MeshingOptionsDialogUI: public Ui_MeshingOptionsDialogUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MESHINGOPTIONSDIALOGUI_H
