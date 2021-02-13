#ifndef LVRREDUCTIONALGORITHMDIALOG_HPP
#define LVRREDUCTIONALGORITHMDIALOG_HPP

#include <string>

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

    /// remove widgets from parameters frame
    void cleanFrame();


    void resetParameters();
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

    bool                            m_successful;

    double                          m_voxelSize;

    int                             m_octreeMinPoints;

    int                             m_fixedNumberPoints;

    int                             m_percentPoints;

    int                             m_reduction;
};

} // namespace std

#endif
