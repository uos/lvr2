#ifndef LVRSCANPROJECTOPENDIALOG_HPP
#define LVRSCANPROJECTOPENDIALOG_HPP

#include "LVRFileAndDirectoryDialog.hpp"

#include "ui_LVRScanProjectOpenDialogUI.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QStringList>

#include <iostream>

using Ui::LVRScanProjectOpenDialogUI;

namespace lvr2
{

class LVRScanProjectOpenDialog : public QDialog
{
    Q_OBJECT

public:

    LVRScanProjectOpenDialog() = delete;
    LVRScanProjectOpenDialog(QWidget* parent);

public Q_SLOTS:
    void openPathDialog();
    void setPathMode(const QString& str);

private:
    void connectSignalsAndSlots();

    LVRScanProjectOpenDialogUI*     m_ui;
    QWidget*                        m_parent;
    LVRFileAndDirectoryDialog       m_fileDialog;

};

} // namespace std

#endif