#include <QFileDialog>
#include "LVRRecordPathDialog.hpp"

namespace lvr
{

LVRRecordPathDialog::LVRRecordPathDialog(vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor, vtkSmartPointer<vtkCameraRepresentation> pathCamera, vtkSmartPointer<LVRTimerCallback> timerCallback, QTreeWidget* treeWidget) :
   m_renderWindowInteractor(renderWindowInteractor), m_pathCamera(pathCamera), m_timerCallback(timerCallback), m_treeWidget(treeWidget)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new RecordPathDialog;
    m_dialog->setupUi(dialog);

    m_dialog->manual_addFrame_button->setVisible(false);
    m_dialog->recording_frames_label->setVisible(false);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRRecordPathDialog::~LVRRecordPathDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRRecordPathDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->recordMode_box, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(changeMode(const QString&)));
    QObject::connect(m_dialog->automatic_recording_button, SIGNAL(pressed()), this, SLOT(recordPath()));
}

void LVRRecordPathDialog::changeMode(const QString &text)
{
    if(text == "Automatic")
    {
        m_dialog->automatic_timer_box->setVisible(true);
        m_dialog->automatic_timer_label->setVisible(true);
        m_dialog->automatic_recording_button->setVisible(true);
        m_dialog->manual_addFrame_button->setVisible(false);
    }
    else if(text == "Manual")
    {
        m_dialog->automatic_timer_box->setVisible(false);
        m_dialog->automatic_timer_label->setVisible(false);
        m_dialog->automatic_recording_button->setVisible(false);
        m_dialog->manual_addFrame_button->setVisible(true);
    }
}

void LVRRecordPathDialog::Path()
{
    QSpinBox* timer_b
    if(m_timerID <= 0)
    {
        m_pathCamera->InitializePath();
        m_timerCallback->reset();
        m_timerID = m_renderWindowInteractor->CreateRepeatingTimer(1000);
        m_dialog->recording_status_label->setText("Currently recording!");
        m_dialog->automatic_recording_button->setText("Stop recording");
    }
    // stop recording if timer ID is valid, destroy timer
    else
    {
        m_dialog->recording_status_label->setText("Currently not recording.");
        m_dialog->automatic_recording_button->setText("Start recording");
        m_renderWindowInteractor->DestroyTimer(m_timerID);
        m_timerID = -1;
    }
}

}
