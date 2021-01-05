#ifndef LVRSCANPROJECTOPENDIALOG_HPP
#define LVRSCANPROJECTOPENDIALOG_HPP

#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"


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

public Q_SLOTS:
    /// Shows the QFileDialog
    void openPathDialog();

    /// Called when a diffent schema was selected
    void schemaSelectionChanged(int index);

private:

    /// Connects signals and slots
    void connectSignalsAndSlots();

    /// Updates the list of available schemas
    /// depending on selected scan project type
    void updateAvailableSchemas();

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
};

} // namespace std

#endif