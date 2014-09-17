#include <QFileDialog>
#include "LVRAnimationDialog.hpp"

namespace lvr
{

LVRAnimationDialog::LVRAnimationDialog(vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor, vtkSmartPointer<vtkCameraRepresentation> pathCamera, QTreeWidget* treeWidget) :
   m_renderWindowInteractor(renderWindowInteractor), m_pathCamera(pathCamera), m_treeWidget(treeWidget)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new AnimationDialog;
    m_dialog->setupUi(dialog);

    m_mainCamera = m_pathCamera->GetCamera();
    m_frameCounter = 0;

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRAnimationDialog::~LVRAnimationDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRAnimationDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->addFrame_button, SIGNAL(pressed()), this, SLOT(addFrame()));
    QObject::connect(m_dialog->removeFrame_button, SIGNAL(pressed()), this, SLOT(removeFrame()));
    QObject::connect(m_dialog->clearFrames_button, SIGNAL(pressed()), this, SLOT(clearFrames()));
    QObject::connect(m_dialog->play_button, SIGNAL(pressed()), this, SLOT(play()));
}

void LVRAnimationDialog::addFrame()
{
    cout << "Frame added." << endl;
    QString frameCount = QString("Frame no. %1").arg(++m_frameCounter);
    QListWidgetItem* currentFrame = new LVRRecordedFrameItem(m_pathCamera, frameCount);
    m_dialog->timeline_list->addItem(currentFrame);
}

void LVRAnimationDialog::removeFrame()
{
    QListWidgetItem* currentItem = m_dialog->timeline_list->currentItem();
    if(currentItem)
    {
        cout << "Deleting " << currentItem->text().toStdString() << "..." << endl;
        delete currentItem;
    }
}

void LVRAnimationDialog::clearFrames()
{
    m_dialog->timeline_list->clear();
    m_frameCounter = 0;
}

void LVRAnimationDialog::play()
{
    cout << "Animating " << m_dialog->timeline_list->count() << " frames..." << endl;
    m_pathCamera->InitializePath();
    for(int i = 0; i < m_dialog->timeline_list->count(); i++)
    {
        LVRRecordedFrameItem* recordedFrame = static_cast<LVRRecordedFrameItem*>(m_dialog->timeline_list->item(i));
        m_pathCamera->SetCamera(recordedFrame->getFrame());
        m_pathCamera->AddCameraToPath();
    }
    m_pathCamera->SetNumberOfFrames(m_dialog->timeline_list->count() * 30);
    m_pathCamera->SetCamera(m_mainCamera);
    m_pathCamera->AnimatePath(m_renderWindowInteractor);
}

}
