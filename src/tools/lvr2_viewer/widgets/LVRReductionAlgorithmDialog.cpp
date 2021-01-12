#include "LVRReductionAlgorithmDialog.hpp"



namespace lvr2
{

LVRReductionAlgorithmDialog::LVRReductionAlgorithmDialog(QWidget* parent): 
    QDialog(parent),
    m_reductionPtr(nullptr),
    m_successful(false),
    m_voxelSize(10),
    m_reduction(0)
{
    m_parent = parent;
    m_ui = new LVRReductionAlgorithmDialogUI;
    m_ui->setupUi(this);

    connectSignalsAndSlots();
    addReductionTypes();
    updateVoxelValue(m_ui->horizontalSlider->value());
}

void LVRReductionAlgorithmDialog::connectSignalsAndSlots()
{
     // Add connections
    QObject::connect(m_ui->buttonBox, SIGNAL(accepted()), this, SLOT(acceptOpen()));
    QObject::connect(m_ui->comboBoxReduction, SIGNAL(currentIndexChanged(int)), this, SLOT(reductionSelectionChanged(int)));
    QObject::connect(m_ui->horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(updateVoxelValue(int)));
}

void LVRReductionAlgorithmDialog::acceptOpen()
{
    m_successful = true;
    switch(m_reduction)
    {
        case 0:
            m_reductionPtr = ReductionAlgorithmPtr(new AllReductionAlgorithm());
            break;
        case 1:
            m_reductionPtr = ReductionAlgorithmPtr(new NoReductionAlgorithm());
            break;
        case 2:
            m_reductionPtr = ReductionAlgorithmPtr(new OctreeReductionAlgorithm(m_voxelSize/100, 5));
            break;
        default:
            break;
    }

}

bool LVRReductionAlgorithmDialog::successful()
{
    return m_successful;
}


void LVRReductionAlgorithmDialog::reductionSelectionChanged(int index)
{
    switch(index)
    {
        case 0:
            m_reduction = 0;
            changeVoxelState(false);
            break;
        case 1:
            m_reduction = 1;
            changeVoxelState(false);
            break;
        case 2:
            m_reduction = 2;
            changeVoxelState(true);
            break;
        default:
            break;
    }
}

void LVRReductionAlgorithmDialog::updateVoxelValue(int value)
{
    m_voxelSize = value;
    m_ui->textEdit->setText(QString::number(value));
    m_ui->textEdit->setAlignment(Qt::AlignCenter);
}



void LVRReductionAlgorithmDialog::addReductionTypes()
{
    QComboBox* b = m_ui->comboBoxReduction;
    b->clear();

    b->addItem("All Reduction");
    b->addItem("No Reduction");
    b->addItem("Octree Reduction");
}

void LVRReductionAlgorithmDialog::changeVoxelState(bool state)
{
    m_ui->horizontalSlider->setEnabled(state);
    m_ui->textEdit->setEnabled(state);
    m_ui->groupBoxVoxel->setEnabled(state);
}


} // namespace lvr2