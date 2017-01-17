#include <QFileDialog>
#include "LVRRenameDialog.hpp"

namespace lvr
{

LVRRenameDialog::LVRRenameDialog(LVRModelItem* item, QTreeWidget* treeWidget) :
   m_item(item), m_treeWidget(treeWidget)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new RenameDialog;
    m_dialog->setupUi(dialog);

    QString oldName = m_item->getName();
    m_dialog->label_dynamic->setText(oldName);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRRenameDialog::~LVRRenameDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRRenameDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(renameItem()));
}

void LVRRenameDialog::renameItem()
{
    QLineEdit* newName_box = m_dialog->lineEdit_name;
    QString newName = newName_box->text();

    m_item->setName(newName);
}

}
