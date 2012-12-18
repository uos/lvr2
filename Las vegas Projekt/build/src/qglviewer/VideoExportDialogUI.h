/********************************************************************************
** Form generated from reading UI file 'VideoExportDialogUI.ui'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef VIDEOEXPORTDIALOGUI_H
#define VIDEOEXPORTDIALOGUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QFrame>
#include <QtGui/QHeaderView>

QT_BEGIN_NAMESPACE

class Ui_VideoExportDialog
{
public:
    QDialogButtonBox *buttonBox;
    QFrame *frame;

    void setupUi(QDialog *VideoExportDialog)
    {
        if (VideoExportDialog->objectName().isEmpty())
            VideoExportDialog->setObjectName(QString::fromUtf8("VideoExportDialog"));
        VideoExportDialog->resize(598, 375);
        buttonBox = new QDialogButtonBox(VideoExportDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(510, 10, 81, 241));
        buttonBox->setOrientation(Qt::Vertical);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        frame = new QFrame(VideoExportDialog);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setGeometry(QRect(10, 10, 491, 351));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);

        retranslateUi(VideoExportDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), VideoExportDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), VideoExportDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(VideoExportDialog);
    } // setupUi

    void retranslateUi(QDialog *VideoExportDialog)
    {
        VideoExportDialog->setWindowTitle(QApplication::translate("VideoExportDialog", "Video export settings", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class VideoExportDialog: public Ui_VideoExportDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // VIDEOEXPORTDIALOGUI_H
