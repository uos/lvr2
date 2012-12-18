/********************************************************************************
** Form generated from reading UI file 'PlayerDialogUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef PLAYERDIALOGUI_H
#define PLAYERDIALOGUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QListWidget>
#include <QtGui/QPushButton>
#include <QtGui/QToolButton>

QT_BEGIN_NAMESPACE

class Ui_PlayerDialogUI
{
public:
    QListWidget *listWidget;
    QGroupBox *groupBoxControlls;
    QToolButton *buttonFirst;
    QToolButton *buttonPrev;
    QToolButton *buttonAnimate;
    QToolButton *buttonNext;
    QToolButton *buttonLast;
    QToolButton *buttonAddFrame;
    QToolButton *buttonDeleteFrame;
    QGroupBox *groupBoxTimeline;
    QLabel *labelStartTime;
    QDoubleSpinBox *spinBoxStartTime;
    QDoubleSpinBox *spinBoxCurrentTime;
    QLabel *labeCurrentTime;
    QDoubleSpinBox *spinBoxLastTime;
    QLabel *labelEndTime;
    QPushButton *buttonCancel;
    QPushButton *buttonCreateVideo;
    QPushButton *buttonLoad;
    QPushButton *buttonSave;

    void setupUi(QDialog *PlayerDialogUI)
    {
        if (PlayerDialogUI->objectName().isEmpty())
            PlayerDialogUI->setObjectName(QString::fromUtf8("PlayerDialogUI"));
        PlayerDialogUI->resize(508, 255);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(PlayerDialogUI->sizePolicy().hasHeightForWidth());
        PlayerDialogUI->setSizePolicy(sizePolicy);
        PlayerDialogUI->setMinimumSize(QSize(508, 255));
        PlayerDialogUI->setMaximumSize(QSize(508, 255));
        listWidget = new QListWidget(PlayerDialogUI);
        listWidget->setObjectName(QString::fromUtf8("listWidget"));
        listWidget->setGeometry(QRect(10, 10, 201, 201));
        groupBoxControlls = new QGroupBox(PlayerDialogUI);
        groupBoxControlls->setObjectName(QString::fromUtf8("groupBoxControlls"));
        groupBoxControlls->setGeometry(QRect(220, 10, 291, 71));
        buttonFirst = new QToolButton(groupBoxControlls);
        buttonFirst->setObjectName(QString::fromUtf8("buttonFirst"));
        buttonFirst->setGeometry(QRect(0, 30, 33, 33));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/qv_play_first.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonFirst->setIcon(icon);
        buttonFirst->setIconSize(QSize(32, 32));
        buttonPrev = new QToolButton(groupBoxControlls);
        buttonPrev->setObjectName(QString::fromUtf8("buttonPrev"));
        buttonPrev->setGeometry(QRect(40, 30, 33, 33));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/qv_play_prev.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonPrev->setIcon(icon1);
        buttonPrev->setIconSize(QSize(32, 32));
        buttonAnimate = new QToolButton(groupBoxControlls);
        buttonAnimate->setObjectName(QString::fromUtf8("buttonAnimate"));
        buttonAnimate->setGeometry(QRect(80, 30, 33, 33));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/qv_play_animate.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonAnimate->setIcon(icon2);
        buttonAnimate->setIconSize(QSize(32, 32));
        buttonNext = new QToolButton(groupBoxControlls);
        buttonNext->setObjectName(QString::fromUtf8("buttonNext"));
        buttonNext->setGeometry(QRect(120, 30, 33, 33));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/qv_play_next.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonNext->setIcon(icon3);
        buttonNext->setIconSize(QSize(32, 32));
        buttonLast = new QToolButton(groupBoxControlls);
        buttonLast->setObjectName(QString::fromUtf8("buttonLast"));
        buttonLast->setGeometry(QRect(160, 30, 33, 33));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/qv_play_last.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonLast->setIcon(icon4);
        buttonLast->setIconSize(QSize(32, 32));
        buttonAddFrame = new QToolButton(groupBoxControlls);
        buttonAddFrame->setObjectName(QString::fromUtf8("buttonAddFrame"));
        buttonAddFrame->setGeometry(QRect(210, 30, 33, 33));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/qv_play_add.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonAddFrame->setIcon(icon5);
        buttonAddFrame->setIconSize(QSize(32, 32));
        buttonDeleteFrame = new QToolButton(groupBoxControlls);
        buttonDeleteFrame->setObjectName(QString::fromUtf8("buttonDeleteFrame"));
        buttonDeleteFrame->setGeometry(QRect(250, 30, 33, 33));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/qv_play_delete.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonDeleteFrame->setIcon(icon6);
        buttonDeleteFrame->setIconSize(QSize(32, 32));
        groupBoxTimeline = new QGroupBox(PlayerDialogUI);
        groupBoxTimeline->setObjectName(QString::fromUtf8("groupBoxTimeline"));
        groupBoxTimeline->setGeometry(QRect(220, 90, 281, 111));
        labelStartTime = new QLabel(groupBoxTimeline);
        labelStartTime->setObjectName(QString::fromUtf8("labelStartTime"));
        labelStartTime->setGeometry(QRect(0, 30, 141, 17));
        spinBoxStartTime = new QDoubleSpinBox(groupBoxTimeline);
        spinBoxStartTime->setObjectName(QString::fromUtf8("spinBoxStartTime"));
        spinBoxStartTime->setEnabled(false);
        spinBoxStartTime->setGeometry(QRect(170, 20, 111, 27));
        spinBoxStartTime->setMaximum(1e+07);
        spinBoxCurrentTime = new QDoubleSpinBox(groupBoxTimeline);
        spinBoxCurrentTime->setObjectName(QString::fromUtf8("spinBoxCurrentTime"));
        spinBoxCurrentTime->setEnabled(true);
        spinBoxCurrentTime->setGeometry(QRect(170, 50, 111, 27));
        spinBoxCurrentTime->setMaximum(1e+07);
        spinBoxCurrentTime->setValue(1);
        labeCurrentTime = new QLabel(groupBoxTimeline);
        labeCurrentTime->setObjectName(QString::fromUtf8("labeCurrentTime"));
        labeCurrentTime->setGeometry(QRect(0, 60, 141, 17));
        spinBoxLastTime = new QDoubleSpinBox(groupBoxTimeline);
        spinBoxLastTime->setObjectName(QString::fromUtf8("spinBoxLastTime"));
        spinBoxLastTime->setEnabled(false);
        spinBoxLastTime->setGeometry(QRect(170, 80, 111, 27));
        spinBoxLastTime->setMaximum(1e+07);
        labelEndTime = new QLabel(groupBoxTimeline);
        labelEndTime->setObjectName(QString::fromUtf8("labelEndTime"));
        labelEndTime->setGeometry(QRect(0, 90, 141, 17));
        buttonCancel = new QPushButton(PlayerDialogUI);
        buttonCancel->setObjectName(QString::fromUtf8("buttonCancel"));
        buttonCancel->setGeometry(QRect(400, 220, 97, 27));
        buttonCreateVideo = new QPushButton(PlayerDialogUI);
        buttonCreateVideo->setObjectName(QString::fromUtf8("buttonCreateVideo"));
        buttonCreateVideo->setGeometry(QRect(210, 220, 97, 27));
        buttonLoad = new QPushButton(PlayerDialogUI);
        buttonLoad->setObjectName(QString::fromUtf8("buttonLoad"));
        buttonLoad->setGeometry(QRect(10, 220, 97, 27));
        buttonSave = new QPushButton(PlayerDialogUI);
        buttonSave->setObjectName(QString::fromUtf8("buttonSave"));
        buttonSave->setGeometry(QRect(110, 220, 97, 27));

        retranslateUi(PlayerDialogUI);
        QObject::connect(buttonCancel, SIGNAL(clicked()), PlayerDialogUI, SLOT(reject()));
        QObject::connect(buttonCreateVideo, SIGNAL(clicked()), PlayerDialogUI, SLOT(accept()));

        QMetaObject::connectSlotsByName(PlayerDialogUI);
    } // setupUi

    void retranslateUi(QDialog *PlayerDialogUI)
    {
        PlayerDialogUI->setWindowTitle(QApplication::translate("PlayerDialogUI", "Dialog", 0, QApplication::UnicodeUTF8));
        groupBoxControlls->setTitle(QApplication::translate("PlayerDialogUI", "Frame Controlls", 0, QApplication::UnicodeUTF8));
        buttonFirst->setText(QApplication::translate("PlayerDialogUI", "...", 0, QApplication::UnicodeUTF8));
        buttonPrev->setText(QApplication::translate("PlayerDialogUI", "...", 0, QApplication::UnicodeUTF8));
        buttonAnimate->setText(QApplication::translate("PlayerDialogUI", "...", 0, QApplication::UnicodeUTF8));
        buttonNext->setText(QApplication::translate("PlayerDialogUI", "...", 0, QApplication::UnicodeUTF8));
        buttonLast->setText(QApplication::translate("PlayerDialogUI", "...", 0, QApplication::UnicodeUTF8));
        buttonAddFrame->setText(QApplication::translate("PlayerDialogUI", "...", 0, QApplication::UnicodeUTF8));
        buttonDeleteFrame->setText(QApplication::translate("PlayerDialogUI", "...", 0, QApplication::UnicodeUTF8));
        groupBoxTimeline->setTitle(QApplication::translate("PlayerDialogUI", "Timeline", 0, QApplication::UnicodeUTF8));
        labelStartTime->setText(QApplication::translate("PlayerDialogUI", "Previous Time:", 0, QApplication::UnicodeUTF8));
        labeCurrentTime->setText(QApplication::translate("PlayerDialogUI", "Duration:", 0, QApplication::UnicodeUTF8));
        labelEndTime->setText(QApplication::translate("PlayerDialogUI", "Next Time:", 0, QApplication::UnicodeUTF8));
        buttonCancel->setText(QApplication::translate("PlayerDialogUI", "Cancel", 0, QApplication::UnicodeUTF8));
        buttonCreateVideo->setText(QApplication::translate("PlayerDialogUI", "Create Video", 0, QApplication::UnicodeUTF8));
        buttonLoad->setText(QApplication::translate("PlayerDialogUI", "Load Path", 0, QApplication::UnicodeUTF8));
        buttonSave->setText(QApplication::translate("PlayerDialogUI", "Save Path", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class PlayerDialogUI: public Ui_PlayerDialogUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // PLAYERDIALOGUI_H
