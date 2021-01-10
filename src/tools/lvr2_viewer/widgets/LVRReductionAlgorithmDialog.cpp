#include "LVRReductionAlgorithmDialog.hpp"



namespace lvr2
{

LVRReductionAlgorithmDialog::LVRReductionAlgorithmDialog(QWidget* parent): 
    QDialog(parent),
    m_reductionPtr(nullptr),
    m_successful(false)
{
    m_parent = parent;
    m_ui = new LVRReductionAlgorithmDialogUI;
    m_ui->setupUi(this);

    connectSignalsAndSlots();
    addReductionTypes();    
}

void LVRReductionAlgorithmDialog::connectSignalsAndSlots()
{
     // Add connections
    QObject::connect(m_ui->buttonBox, SIGNAL(accepted()), this, SLOT(acceptOpen()));
    QObject::connect(m_ui->comboBoxReduction, SIGNAL(currentIndexChanged(int)), this, SLOT(reductionSelectionChanged(int)));
}

void LVRReductionAlgorithmDialog::acceptOpen()
{
    m_successful = true;
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
            m_reductionPtr = ReductionAlgorithmPtr(new AllReductionAlgorithm());
            break;
        case 1:
            m_reductionPtr = ReductionAlgorithmPtr(new NoReductionAlgorithm());
            break;
        case 2:
            m_reductionPtr = ReductionAlgorithmPtr(new OctreeReductionAlgorithm(0.1, 5));
            break;
        default:
            break;
    }
}



void LVRReductionAlgorithmDialog::addReductionTypes()
{
    QComboBox* b = m_ui->comboBoxReduction;
    b->clear();

    b->addItem("All Reduction");
    b->addItem("No Reduction");
    b->addItem("Octree Reduction");

}


} // namespace lvr2