#ifndef RECONSTRUCTIONMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONMARCHINGCUBESDIALOG_H_

#include <vtkRenderWindow.h>

#include "LVRReconstructionMarchingCubesDialogUI.h"
#include "LVRModelItem.hpp"

using Ui::ReconstructViaMarchingCubesDialog;

namespace lvr
{

class LVRReconstructViaMarchingCubesDialog : public QObject
{
    Q_OBJECT

public:
    LVRReconstructViaMarchingCubesDialog(LVRModelItem* parent, vtkRenderWindow* renderer);
    virtual ~LVRReconstructViaMarchingCubesDialog();

public Q_SLOTS:
    void save();

private:
    void connectSignalsAndSlots();

    ReconstructViaMarchingCubesDialog*      m_dialog;
    LVRModelItem*                           m_parent;
    vtkRenderWindow*                        m_renderWindow;

};

} // namespace lvr

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */
