#ifndef LVRPOINTCLOUDSELECTIONDIALOG_H
#define LVRPOINTCLOUDSELECTIONDIALOG_H

#include <QDialog>
#include <QDialogButtonBox>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>

class LVRPointcloudSelectionDialog : public QDialog
{
Q_OBJECT
public:
    LVRPointcloudSelectionDialog(QStringList names, QWidget *parent = 0);
    QStringList getChecked();
public Q_SLOTS:
    void highlightChecked(QListWidgetItem* item);
    void save();
private:
    QListWidget* widget;
    QDialogButtonBox* buttonBox;
    QGroupBox* viewBox;
    QPushButton* saveButton;
    QPushButton* closeButton;

};

#endif // LVRPOINTCLOUDSELECTIONDIALOG
