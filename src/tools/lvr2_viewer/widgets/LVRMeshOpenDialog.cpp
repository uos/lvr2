#include "LVRMeshOpenDialog.hpp"
#include "ui_LVRMeshOpenDialogUI.h"

LVRMeshOpenDialog::LVRMeshOpenDialog(QWidget *parent) :
    QDialog(parent),
    m_ui(new Ui::LVRMeshOpenDialog),
    m_success(false),
    m_failureReason("unknown")
{
    m_ui->setupUi(this);
    this->connectSignalsAndSlots();
}

void LVRMeshOpenDialog::onAccept()
{
    m_success = true;
}

void LVRMeshOpenDialog::onCancel()
{
    m_failureReason = "canceled by user";
    m_success = false;
}

void LVRMeshOpenDialog::setAvailableMeshes(const std::vector<std::string>& names)
{
    for (auto& name: names)
    {
        // TODO: Add to list view
        m_ui->comboBox->addItem(name.c_str());
    }
}

void LVRMeshOpenDialog::onSelectedMeshNameChanged(QString name)
{
    m_selectedMeshName = name.toStdString();
}

std::string LVRMeshOpenDialog::getSelectedMesh() const
{
    return m_selectedMeshName;
}

void LVRMeshOpenDialog::connectSignalsAndSlots()
{
    QObject::connect(m_ui->buttonBox, SIGNAL(accepted()), this, SLOT(onAccept()));
    QObject::connect(m_ui->buttonBox, SIGNAL(rejected()), this, SLOT(onCancel()));
    QObject::connect(m_ui->comboBox, SIGNAL(currentTextChanged(QString)), this, SLOT(onSelectedMeshNameChanged(QString)));
}

LVRMeshOpenDialog::~LVRMeshOpenDialog()
{
    delete m_ui;
}
