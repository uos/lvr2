/**
 * TransformationDialog.cpp
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#include "TransformationDialog.h"

TransformationDialog::TransformationDialog(QWidget* parent, Renderable* r)
    : m_renderable(r)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(parent);
    m_dialog = new TransformationDialogUI;
    m_dialog->setupUi(dialog);

    dialog->show();
}

TransformationDialog::~TransformationDialog()
{
    // TODO Auto-generated destructor stub
}
