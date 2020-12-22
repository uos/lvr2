#include "LVRScanProjectOpenDialog.hpp"



namespace lvr2
{

LVRScanProjectOpenDialog::LVRScanProjectOpenDialog(QWidget* parent): 
    QDialog(parent),
    m_schema(nullptr),
    m_kernel(nullptr),
    m_projectType(NONE)
{
    m_parent = parent;
    m_ui = new LVRScanProjectOpenDialogUI;
    m_ui->setupUi(this);

    connectSignalsAndSlots();    
}

void LVRScanProjectOpenDialog::connectSignalsAndSlots()
{
     // Add connections
    QObject::connect(m_ui->toolButtonPath, SIGNAL(pressed()), this, SLOT(openPathDialog()));    
    QObject::connect(m_ui->comboBoxSchema, SIGNAL(currentIndexChanged(int)), this, SLOT(schemaSelectionChanged(int)));
}

void LVRScanProjectOpenDialog::schemaSelectionChanged(int index)
{
    if(index != -1 && m_kernel)
    {
        switch(m_projectType)
        {
            case DIR:
                updateDirectorySchema(index);
                break;
            case HDF5:
                updateHDF5Schema(index);
                break;
            default:
                break;
        }
    }
}

void LVRScanProjectOpenDialog::updateDirectorySchema(int index)
{
    switch(index)
    {
        case 0:
            m_schema = ScanProjectSchemaPtr(new ScanProjectSchemaHyperlib(m_kernel->fileResource()));
            break;
        case 1:
            m_schema = ScanProjectSchemaPtr(new ScanProjectSchemaSLAM(m_kernel->fileResource()));
            break;
    }
}
void LVRScanProjectOpenDialog::updateHDF5Schema(int index)
{
    switch(index)
    {
        case 0:
            m_schema = ScanProjectSchemaPtr(new ScanProjectSchemaHDF5V2());
            break;
    }
}

void LVRScanProjectOpenDialog::updateAvailableSchemas()
{
    // Clear all schema entries in combo box
    QComboBox* b = m_ui->comboBoxSchema;
    b->clear();

    // Check if kernel exists, than the current UI 
    // should be consistent. Otherwise, do not 
    // offer any schema
    if(!m_kernel)
    {
        b->addItem("None");
        if(m_schema)
        {
            m_schema.reset();
        }
        return;
    }

    // Add available schemas depending on selected 
    // project type
    switch(m_projectType)
    {
        case NONE:
            b->addItem("None");
            break;
        case HDF5:
            b->addItem("HDF5 Schema V2");
            break;
        case DIR:
            b->addItem("Hyperlib");
            b->addItem("SLAM 6D");
            break;    
        default:
            b->addItem("None");      
    }
}

void LVRScanProjectOpenDialog::openPathDialog()
{   
    // Directory mode
    if(m_ui->comboBoxProjectType->currentIndex() == 0)
    {
        // Set dialog properties
        QFileDialog dialog(m_parent);
        dialog.setFileMode(QFileDialog::DirectoryOnly);

        // Get selected directory
        QStringList fileNames;
        if (dialog.exec())
        {
            fileNames = dialog.selectedFiles();
            if(fileNames.count() > 0)
            {
                QString selectedDir = fileNames[0];
                m_ui->lineEditPath->setText(selectedDir);

                // Check if path is valid
                QFileInfo info(selectedDir);
                if(info.exists())
                {
                    // Update project type flag
                    m_projectType = DIR;

                    // Update kernel
                    m_kernel = FileKernelPtr(new DirectoryKernel(selectedDir.toStdString()));
                }
            }
        }
    }
    // HDF5 mode
    else if(m_ui->comboBoxProjectType->currentIndex() == 1)
    {
        // Set dialog properties
        QFileDialog dialog(m_parent);
        dialog.setFileMode(QFileDialog::AnyFile);
        dialog.setNameFilter(tr("HDF5 Files (*.h5 *.hdf5)"));
        
        // Get selected file
        QStringList fileNames;
        if (dialog.exec())
        {
            fileNames = dialog.selectedFiles();
            if(fileNames.count() > 0)
            {
                QString selectedFile = fileNames[0];
                m_ui->lineEditPath->setText(selectedFile);
             
                // Check if path is valid
                QFileInfo info(selectedFile);
                if(info.exists())
                {
                    // Update project type flag
                    m_projectType = HDF5;

                    // Update kernelelete m_kernel;
                    m_kernel = FileKernelPtr(new HDF5Kernel(selectedFile.toStdString()));
                }
            }
        }
    }
    else
    {
        std::cout << "LVRScanProjectDialog: Unknown item index: "
                  << m_ui->comboBoxProjectType->currentIndex() << std::endl;
    }

    // Update list of available schemas depending
    // on selected project type 
    updateAvailableSchemas();
}

} // namespace lvr2