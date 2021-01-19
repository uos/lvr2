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
    updateInputValue(m_ui->input1->value());
}

void LVRReductionAlgorithmDialog::connectSignalsAndSlots()
{
     // Add connections
    QObject::connect(m_ui->buttonBox, SIGNAL(accepted()), this, SLOT(acceptOpen()));
    QObject::connect(m_ui->comboBoxReduction, SIGNAL(currentIndexChanged(int)), this, SLOT(reductionSelectionChanged(int)));
    QObject::connect(m_ui->input1, SIGNAL(valueChanged(int)), this, SLOT(updateInputValue(int)));
}

void LVRReductionAlgorithmDialog::acceptOpen()
{
    m_successful = true;
    switch(m_reduction)
    {
        case 0:
            m_reductionPtr = ReductionAlgorithmPtr(new NoReductionAlgorithm());
            break;
        case 1:
            m_reductionPtr = ReductionAlgorithmPtr(new AllReductionAlgorithm());
            break;
        case 2:
            m_reductionPtr = ReductionAlgorithmPtr(new OctreeReductionAlgorithm(m_voxelSize/100, 5));
            break;
        case 3:
            m_reductionPtr = ReductionAlgorithmPtr(new FixedSizeReductionAlgorithm(m_voxelSize));
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
            changeInputState(false);
            break;
        case 1:
            m_reduction = 1;
            changeInputState(false);
            break;
        case 2:
            m_reduction = 2;
            changeInputState(true);
            break;
        case 3:
            m_reduction = 3;
            changeInputState(true);
            break;
        default:
            break;
    }
}

void LVRReductionAlgorithmDialog::updateInputValue(int value)
{
    m_voxelSize = value;
    //m_ui->textEdit->setText(QString::number(value));
    //m_ui->textEdit->setAlignment(Qt::AlignCenter);
}



void LVRReductionAlgorithmDialog::addReductionTypes()
{
    QComboBox* b = m_ui->comboBoxReduction;
    b->clear();

    b->addItem("No Reduction");
    b->addItem("All Reduction");
    b->addItem("Octree Reduction");
    b->addItem("Fixed Size");
}

void LVRReductionAlgorithmDialog::changeInputState(bool state)
{
    //m_ui->horizontalSlider->setEnabled(state);
    m_ui->input1->setEnabled(state);
    m_ui->groupBoxVoxel->setEnabled(state);
}


} // namespace lvr2