#include "LVRFileAndDirectoryDialog.hpp"

namespace lvr2
{

LVRFileAndDirectoryDialog::LVRFileAndDirectoryDialog(QWidget *parent)
    : QFileDialog(parent)
{
    setOption(QFileDialog::DontUseNativeDialog);
    //setFileMode(QFileDialog::Directory);
    setFileMode(QFileDialog::ExistingFiles);
    for (auto *pushButton : findChildren<QPushButton *>())
    {
        qDebug() << pushButton->text();
        if (pushButton->text() == "&Open" || pushButton->text() == "&Choose")
        {
            openButton = pushButton;
            break;
        }
    }
    disconnect(openButton, SIGNAL(clicked(bool)));
    connect(openButton, &QPushButton::clicked, this, &LVRFileAndDirectoryDialog::openClicked);
    treeView = findChild<QTreeView *>();
}

QStringList LVRFileAndDirectoryDialog::selected() const
{
    return selectedFilePaths;
}

void LVRFileAndDirectoryDialog::openClicked()
{
    selectedFilePaths.clear();
    // qDebug() << treeView->selectionModel()->selection();
    for (const auto &modelIndex : treeView->selectionModel()->selectedIndexes())
    {
        qDebug() << modelIndex.column();
        if (modelIndex.column() == 0)
        {
            //qDebug() << modelIndex.data().toString();
            QFileInfo info(modelIndex.data().toString());
            if (info.isFile())
            {
                selectedFilePaths.append(directory().absolutePath() + "/" + modelIndex.data().toString());
            }
            else
            {
                 selectedFilePaths.append(directory().absolutePath());
            }
        }
    }
    qDebug() << selectedFilePaths;
    Q_EMIT filesSelected(selectedFilePaths);
    hide();
}

} // namespace lvr2