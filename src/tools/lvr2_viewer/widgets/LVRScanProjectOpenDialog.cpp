#include "LVRScanProjectOpenDialog.hpp"



namespace lvr2
{

LVRScanProjectOpenDialog::LVRScanProjectOpenDialog(QWidget* parent): QDialog(parent)
{
    m_parent = parent;
    m_ui = new LVRScanProjectOpenDialogUI;
    m_ui->setupUi(this);

    connectSignalsAndSlots();    
}

void LVRScanProjectOpenDialog::connectSignalsAndSlots()
{
     // Add connections
    QObject::connect(m_ui->pushButton, SIGNAL(pressed()), this, SLOT(openPathDialog()));
  //  QObject::connect(&m_fileDialog, &QFileDialog::currentChanged, this, &LVRScanProjectOpenDialog::setPathMode);
      
}

void LVRScanProjectOpenDialog::setPathMode(const QString& str)
{   
    // std::cout << "OK" << std::endl;
    // std::cout << str.toStdString() << std::endl;
    // // Quickfix to allow selection of files or directories:
    // // https://stackoverflow.com/questions/27520304/qfiledialog-that-accepts-a-single-file-or-a-single-directory  
    // QFileInfo info(str);
    // if(info.isFile())
    // {
    //     m_fileDialog.setFileMode(QFileDialog::ExistingFile);
    // }   
    // else if(info.isDir())
    // {
    //     m_fileDialog.setFileMode(QFileDialog::Directory);
    // }

}

void LVRScanProjectOpenDialog::openPathDialog()
{    
    QStringList fileNames;
    if (m_fileDialog.exec())
    {
        fileNames = m_fileDialog.selectedFiles();
    }
   
}

} // namespace lvr2