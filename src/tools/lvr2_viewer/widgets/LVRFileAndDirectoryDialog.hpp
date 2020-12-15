#ifndef LVR_FILEANDDIRECTORYDIALOG_HPP
#define LVR_FILEANDDIRECTORYDIALOG_HPP

#include <QFileDialog>
#include <QApplication>
#include <QWidget>
#include <QTreeWidget>
#include <QPushButton>
#include <QStringList>
#include <QModelIndex>
#include <QDir>
#include <QDebug>
#include <QFileInfo>

#include <iostream>

namespace lvr2
{

class LVRFileAndDirectoryDialog : public QFileDialog
{
    Q_OBJECT
public:
    explicit LVRFileAndDirectoryDialog(QWidget *parent = Q_NULLPTR);

    QStringList selected() const;

public Q_SLOTS:
    void openClicked();

private:
    QTreeView *treeView;
    QPushButton *openButton;
    QStringList selectedFilePaths;
};

} // namespace lvr2
#endif