#ifndef LVRREDUCTIONALGORITHMDIALOG_HPP
#define LVRREDUCTIONALGORITHMDIALOG_HPP

#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaSLAM.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"
#include "lvr2/registration/OctreeReduction.hpp"


#include "ui_LVRReductionAlgorithmDialogUI.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QStringList>

#include <iostream>

using Ui::LVRReductionAlgorithmDialogUI;

namespace lvr2
{
/**
 * @brief   Custom dialog that allows to open a scan project
 *          (directory or HDF5) and select a suitable schema.
 */
class LVRReductionAlgorithmDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new LVRScanProjectOpenDialog object
     */
    LVRReductionAlgorithmDialog() = delete;

    /**
     * @brief Construct a new LVRScanProjectOpenDialog object
     * @param parent    Parent widget
     */
    LVRReductionAlgorithmDialog(QWidget* parent);

    /**
     * @brief Destroy the LVRScanProjectOpenDialog object
     */
    virtual ~LVRReductionAlgorithmDialog() = default;


    /**
     * @brief Returns ReductionPtr
     * 
     * @return ReductionPtr
     */
    ReductionAlgorithmPtr reductionPtr() {return m_reductionPtr;}

    int reductionName() {return m_reduction;}


    /**
     *  @brief Return whether the dialog was finished with OK
     */
    bool successful();

public Q_SLOTS:

    /// Called when a diffent reduction was selected
    void reductionSelectionChanged(int index);

    void updateVoxelValue(int value);
    void changeVoxelState(bool state);

    //Called when OK is pressed
    void acceptOpen();

private:

    /// Connects signals and slots
    void connectSignalsAndSlots();


    void addReductionTypes();

    /// Pointer to the UI
    LVRReductionAlgorithmDialogUI*     m_ui;

    /// Parent widget
    QWidget*                        m_parent;

    ReductionAlgorithmPtr           m_reductionPtr;

    bool                            m_successful;

    double                          m_voxelSize;

    int                             m_reduction;
};

} // namespace std

#endif