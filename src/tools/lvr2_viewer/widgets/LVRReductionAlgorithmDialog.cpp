#include "LVRReductionAlgorithmDialog.hpp"



namespace lvr2
{

LVRReductionAlgorithmDialog::LVRReductionAlgorithmDialog(QWidget* parent): 
    QDialog(parent),
    m_reductionPtr(nullptr),
    m_successful(false),
    m_voxelSize(1000.0),
    m_reduction(0)
{
    m_parent = parent;
    m_ui = new LVRReductionAlgorithmDialogUI;
    m_ui->setupUi(this);
    this->setFixedSize(this->size().width(), this->size().height());

    connectSignalsAndSlots();
    addReductionTypes();
    resetParameters();
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
    switch(m_reduction)
    {
        case 0:
            m_reductionPtr = ReductionAlgorithmPtr(new NoReductionAlgorithm());
            break;
        case 1:
            m_reductionPtr = ReductionAlgorithmPtr(new AllReductionAlgorithm());
            break;
        case 2:
            m_reductionPtr = ReductionAlgorithmPtr(new OctreeReductionAlgorithm(m_voxelSize, m_octreeMinPoints));
            break;
        case 3:
            m_reductionPtr = ReductionAlgorithmPtr(new FixedSizeReductionAlgorithm(m_fixedNumberPoints));
            break;
        case 4:
            m_reductionPtr = ReductionAlgorithmPtr(new PercentageReductionAlgorithm((float)m_percentPoints / 100.0));
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
    cleanFrame();
    resetParameters();
    switch(index)
    {
        case 0:
            // No reduction
            m_reduction = 0;
            addLabelToFrame(QString("Will load the complete point buffer"));
            break;
        case 1:
            // All reduction
            m_reduction = 1;
            addLabelToFrame(QString("Will load an empty point buffer"));
            break;
        case 2:
            {
                m_reduction = 2;
                addLabelToFrame(QString("Will perform an octree reduction\n"));
                addLabelToFrame(QString("Voxel size"));
                QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(m_ui->parameters_frame->layout());
                QDoubleSpinBox* inputVoxelSize = new QDoubleSpinBox(m_ui->parameters_frame);
                inputVoxelSize->setMaximum(999999999.0);
                inputVoxelSize->setValue(m_voxelSize);
                layout->insertWidget(m_ui->parameters_frame->layout()->count() - 1, inputVoxelSize);
                QObject::connect(inputVoxelSize, SIGNAL(valueChanged(double)), this, SLOT(setVoxelSize(double)));

                addLabelToFrame(QString("Min points per leaf"));
                QSpinBox* inputMinPoints = new QSpinBox(m_ui->parameters_frame);
                inputMinPoints->setMaximum(999999999);
                inputMinPoints->setMinimum(1);
                inputMinPoints->setValue(m_octreeMinPoints);
                layout->insertWidget(m_ui->parameters_frame->layout()->count() - 1, inputMinPoints);
                QObject::connect(inputMinPoints, SIGNAL(valueChanged(int)), this, SLOT(setOctreeMinPoints(int)));
                break;
            }
        case 3:
            {
                m_reduction = 3;
                addLabelToFrame(QString("Will load a fixed number of points\n"));
                addLabelToFrame(QString("Number of points"));
                QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(m_ui->parameters_frame->layout());
                QSpinBox* input = new QSpinBox(m_ui->parameters_frame);
                input->setMaximum(999999999);
                input->setValue(m_fixedNumberPoints);
                layout->insertWidget(m_ui->parameters_frame->layout()->count() - 1, input);
                QObject::connect(input, SIGNAL(valueChanged(int)), this, SLOT(setFixedNumberPoints(int)));
                break;
            }
        case 4:
            {
                m_reduction = 4;
                addLabelToFrame(QString("Will load a certain percentage of the point buffer\n"));
                addLabelToFrame(QString("Percentage (%)"));
                QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(m_ui->parameters_frame->layout());
                QSpinBox* input = new QSpinBox(m_ui->parameters_frame);
                input->setMaximum(100);
                input->setValue(m_percentPoints);
                layout->insertWidget(m_ui->parameters_frame->layout()->count() - 1, input);
                QObject::connect(input, SIGNAL(valueChanged(int)), this, SLOT(setPercentPoints(int)));
                break;
            }
        default:
            break;
    }
}

void LVRReductionAlgorithmDialog::cleanFrame() {
    if ( m_ui->parameters_frame->layout() != NULL )
    {
        if(m_ui->parameters_frame->layout()->count() > 1){
            QLayoutItem* item;
            while ( m_ui->parameters_frame->layout()->count() > 1 )
            {
                item = m_ui->parameters_frame->layout()->takeAt(0);
                delete item->widget();
                delete item;
            }
        }
    }
}

void LVRReductionAlgorithmDialog::resetParameters() {
    setFixedNumberPoints(10000);
    setPercentPoints(10);
    setVoxelSize(0.1);
    setOctreeMinPoints(5);
}


void LVRReductionAlgorithmDialog::addLabelToFrame(QString labelText)
{
    QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(m_ui->parameters_frame->layout());
    QLabel* label = new QLabel(labelText, m_ui->parameters_frame);
    layout->insertWidget(m_ui->parameters_frame->layout()->count() - 1, label);
}

void LVRReductionAlgorithmDialog::setVoxelSize(double value)
{
    m_voxelSize = value;
}

void LVRReductionAlgorithmDialog::setFixedNumberPoints(int value)
{
    m_fixedNumberPoints = value;
}

void LVRReductionAlgorithmDialog::setPercentPoints(int value)
{
    m_percentPoints = value;
}

void LVRReductionAlgorithmDialog::setOctreeMinPoints(int value)
{
    m_octreeMinPoints = value;
}

void LVRReductionAlgorithmDialog::addReductionTypes()
{
    QComboBox* b = m_ui->comboBoxReduction;
    b->clear();

    b->addItem("Complete Point Buffer");
    b->addItem("Empty Point Buffer");
    b->addItem("Octree Reduction");
    b->addItem("Fixed Size");
    b->addItem("Percentage");
}

} // namespace lvr2