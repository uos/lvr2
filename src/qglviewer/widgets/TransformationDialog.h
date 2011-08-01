/**
 * TransformationDialog.h
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#ifndef TRANSFORMATIONDIALOG_H_
#define TRANSFORMATIONDIALOG_H_

#include "TransformationDialogUI.h"

#include "model3d/Renderable.h"

using Ui::TransformationDialogUI;

class TransformationDialog
{
public:
    TransformationDialog(QWidget* parent, Renderable* r);
    virtual ~TransformationDialog();

private:
    Renderable*                 m_renderable;
    TransformationDialogUI*     m_dialog;

};

#endif /* TRANSFORMATIONDIALOG_H_ */
