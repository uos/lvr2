#ifndef LVRSCANPROJECTOPENDIALOG
#define LVRSCANPROJECTOPENDIALOG

#include "lvr2/io/kernels/HDF5Kernel.hpp"
#include "lvr2/io/kernels/FileKernel.hpp"
#include "lvr2/io/kernels/DirectoryKernel.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/schema/ScanProjectSchemaSlam6D.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHDF5.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"
#include "lvr2/registration/OctreeReduction.hpp"
#include "LVRReductionAlgorithmDialog.hpp"
#include "../vtkBridge/LVRScanProjectBridge.hpp"

#include "ui_LVRScanProjectOpenDialogUI.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QStringList>

#include <iostream>

using Ui::LVRScanProjectOpenDialogUI;

namespace lvr2
{
/**
 * @brief   Custom dialog that allows to open a scan project
 *          (directory or HDF5) and select a suitable schema.
 */
class LVRScanProjectOpenDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new LVRScanProjectOpenDialog object
     */
    LVRScanProjectOpenDialog() = delete;

    /**
     * @brief Construct a new LVRScanProjectOpenDialog object
     * @param parent    Parent widget
     */
    LVRScanProjectOpenDialog(QWidget* parent);

    /**
     * @brief Destroy the LVRScanProjectOpenDialog object
     */
    virtual ~LVRScanProjectOpenDialog() = default;

    /**
     * @brief Returns a pointer to the currently selected schema
     */
    ScanProjectSchemaPtr schema() {return m_schema;} 
   
    /**
     * @brief Returns a pointer to the current file kernel
     * 
     * @return FileKernelPtr 
     */
    FileKernelPtr kernel() {return m_kernel;}

    /// Used to represent the type
    /// of the currently selected scan project
    enum ProjectType{NONE, DIR, HDF5};

    /**
     * @brief Returns ProjectType
     * 
     * @return ProjectType
     */
    ProjectType projectType() {return m_projectType;}

    /**
     * @brief Returns ReductionPtr
     * 
     * @return ReductionPtr
     */
    ReductionAlgorithmPtr reductionPtr() {return m_reductionPtr;}

        /**
     * @brief Returns ProjectType
     * 
     * @return ProjectType
     */
    ProjectScale projectScale() {return m_projectScale;}

    /**
     *  @brief Return whether the dialog was finished with OK
     */
    bool successful();

public Q_SLOTS:
    /// Shows the QFileDialog
    void openPathDialog();

    /// Called to open the reduction algorithm dialog
    void openReductionDialog();

    /// Called when a diffent project type was selected
    void projectTypeSelectionChanged(int index);

    /// Called when a diffent schema was selected
    void schemaSelectionChanged(int index);

    /// Called when a diffent reduction was selected
    void projectScaleSelectionChanged(int index);

    /// Called when OK is pressed
    void acceptOpen();

private:

    /// Connects signals and slots
    void connectSignalsAndSlots();

    /// Updates the list of available schemas
    /// depending on selected scan project type
    void updateAvailableSchemas();

    /// adds scales to combobox and sets default selection
    void initAvailableScales();

    /// Sets the current (directory) schema based
    /// on the selected list index in the combo box
    void updateDirectorySchema(int index);

    /// Sets the current (HDF5) schema based on the 
    /// selected list index in the combo box
    void updateHDF5Schema(int index);

    /// Pointer to the UI
    LVRScanProjectOpenDialogUI*     m_ui;

    /// Parent widget
    QWidget*                        m_parent;

    /// Selected scan project schema
    ScanProjectSchemaPtr            m_schema;

    /// File kernel for selected project
    FileKernelPtr                   m_kernel;    

    /// Current scan project type
    ProjectType                     m_projectType;

    /// Current reduction type algorithm
    ReductionAlgorithmPtr           m_reductionPtr;

    /// Current project scale which affects
    /// scaling of scanner position cylinder
    ProjectScale                    m_projectScale;

    /// States that the dialog was submitted successfully
    bool                            m_successful;
};

} // namespace std

#endif // LVRSCANPROJECTOPENDIALOG
