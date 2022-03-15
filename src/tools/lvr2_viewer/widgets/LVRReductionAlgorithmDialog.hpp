#ifndef LVRREDUCTIONALGORITHMDIALOG_HPP
#define LVRREDUCTIONALGORITHMDIALOG_HPP

#include <string>

#include "lvr2/io/kernels/FileKernel.hpp"
#include "lvr2/io/kernels/DirectoryKernel.hpp"
#include "lvr2/io/kernels/HDF5Kernel.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/schema/ScanProjectSchemaSlam6D.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHDF5V2.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"
#include "lvr2/registration/OctreeReduction.hpp"

#include "ui_LVRReductionAlgorithmDialogUI.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QStringList>

#include <QVBoxLayout>
#include <QString>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>

#include <iostream>

using Ui::LVRReductionAlgorithmDialogUI;

namespace lvr2
{
/**
 * @brief   Custom dialog that allows to choose a ReductionAlgorithm
 */
class LVRReductionAlgorithmDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new LVRReductionAlgorithmDialog object
     */
    LVRReductionAlgorithmDialog() = delete;

    /**
     * @brief Construct a new LVRReductionAlgorithmDialog object
     * @param parent    Parent widget
     */
    LVRReductionAlgorithmDialog(QWidget* parent);

    /**
     * @brief Destroy the LVRSReductionAlgorithmDialog object
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

    /// Called when number of fixed points is changed (FixedReduction)
    void setFixedNumberPoints(int value);

    /// Called when percent is changed (PercentageReduction)
    void setPercentPoints(int value);

    /// Called when voxel size is changed (OctreeReduction)
    void setVoxelSize(double value);

    /// Called when min num of points per leaf is changed (OctreeReduction)
    void setOctreeMinPoints(int value);

    /// Called when OK is pressed
    void acceptOpen();

    /// Remove widgets from parameters frame
    void cleanFrame();

    /// Resets dialog ui parameters
    void resetParameters();

    /// Adds a label as a tooltip to the frame
    void addLabelToFrame(QString labelText);

private:

    /// Connects signals and slots
    void connectSignalsAndSlots();

    /// Add items to reduction combo box
    void addReductionTypes();

    /// Pointer to the UI
    LVRReductionAlgorithmDialogUI*  m_ui;

    /// Parent widget
    QWidget*                        m_parent;

    /// Pointer to the selected reduction algorithm
    ReductionAlgorithmPtr           m_reductionPtr;

    /// States that the dialog was submitted successfully  
    bool                            m_successful;

    /// Size of the voxels used in the octree reduction
    double                          m_voxelSize;

    /// Size of the min points per leaf in the octree reduction
    int                             m_octreeMinPoints;

    /// Fixed amount of points in reduction
    int                             m_fixedNumberPoints;

    /// Percentage of points used in reduction
    int                             m_percentPoints;

    /// Current reduction type
    int                             m_reduction;
};

} // namespace std

#endif
