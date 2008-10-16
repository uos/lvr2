/********************************************************************************
** Form generated from reading ui file 'test.ui'
**
** Created: Thu Sep 4 16:09:06 2008
**      by: Qt User Interface Compiler version 4.4.0
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef TEST_H
#define TEST_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDockWidget>
#include <QtGui/QPushButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Form
{
public:
    QWidget *dockWidgetContents;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QPushButton *pushButton_3;

    void setupUi(QDockWidget *Form)
    {
    if (Form->objectName().isEmpty())
        Form->setObjectName(QString::fromUtf8("Form"));
    Form->resize(200, 300);
    Form->setFeatures(QDockWidget::DockWidgetFeatureMask);
    dockWidgetContents = new QWidget();
    dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
    dockWidgetContents->setGeometry(QRect(20, 0, 180, 300));
    pushButton = new QPushButton(dockWidgetContents);
    pushButton->setObjectName(QString::fromUtf8("pushButton"));
    pushButton->setGeometry(QRect(10, 10, 75, 26));
    pushButton_2 = new QPushButton(dockWidgetContents);
    pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
    pushButton_2->setGeometry(QRect(10, 40, 75, 26));
    pushButton_3 = new QPushButton(dockWidgetContents);
    pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
    pushButton_3->setGeometry(QRect(10, 70, 75, 26));
    Form->setWidget(dockWidgetContents);

    retranslateUi(Form);

    QMetaObject::connectSlotsByName(Form);
    } // setupUi

    void retranslateUi(QDockWidget *Form)
    {
    Form->setWindowTitle(QApplication::translate("Form", "Form", 0, QApplication::UnicodeUTF8));
    pushButton->setText(QApplication::translate("Form", "PushButton", 0, QApplication::UnicodeUTF8));
    pushButton_2->setText(QApplication::translate("Form", "PushButton", 0, QApplication::UnicodeUTF8));
    pushButton_3->setText(QApplication::translate("Form", "PushButton", 0, QApplication::UnicodeUTF8));
    Q_UNUSED(Form);
    } // retranslateUi

};

namespace Ui {
    class Form: public Ui_Form {};
} // namespace Ui

QT_END_NAMESPACE

#endif // TEST_H
