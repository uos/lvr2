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

    m_timeline = m_dialog->timeline_list;
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
    QObject::connect(m_dialog->interpolation_box, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(changeInterpolation(const QString&)));
    QObject::connect(m_dialog->savePath_button, SIGNAL(pressed()), this, SLOT(savePath()));
    QObject::connect(m_dialog->loadPath_button, SIGNAL(pressed()), this, SLOT(loadPath()));
    QObject::connect(m_dialog->play_button, SIGNAL(pressed()), this, SLOT(play()));
}

void LVRAnimationDialog::addFrame()
{
    cout << "Frame added." << endl;
    QString frameCount = QString("Frame no. %1").arg(++m_frameCounter);
    QListWidgetItem* currentFrame = new LVRRecordedFrameItem(m_pathCamera, frameCount);
    m_timeline->addItem(currentFrame);
}

void LVRAnimationDialog::removeFrame()
{
    QListWidgetItem* currentItem = m_timeline->currentItem();
    if(currentItem)
    {
        cout << "Deleting " << currentItem->text().toStdString() << "..." << endl;
        delete currentItem;
    }
}

void LVRAnimationDialog::clearFrames()
{
    m_timeline->clear();
    m_frameCounter = 0;
}

void LVRAnimationDialog::changeInterpolation(const QString& text)
{
    if(text == "Linear")
   {
        m_pathCamera->GetInterpolator()->SetInterpolationTypeToLinear();
   }
   else if(text == "Spline")
   {
       m_pathCamera->GetInterpolator()->SetInterpolationTypeToSpline();
   }
}

void LVRAnimationDialog::play()
{
    unsigned int frameCount = m_timeline->count();
    cout << "Animating " << frameCount << " frames..." << endl;
    m_pathCamera->InitializePath();
    for(int i = 0; i < frameCount; i++)
    {
        LVRRecordedFrameItem* recordedFrame = static_cast<LVRRecordedFrameItem*>(m_timeline->item(i));
        m_pathCamera->SetCamera(recordedFrame->getFrame());
        m_pathCamera->AddCameraToPath();
    }
    unsigned int frameMultiplier = m_dialog->frameMultiplier_box->value();
    m_pathCamera->SetNumberOfFrames(frameCount * frameMultiplier);
    m_pathCamera->SetCamera(m_mainCamera);
    m_pathCamera->AnimatePath(m_renderWindowInteractor);
}

void LVRAnimationDialog::savePath()
{
    QString filename = QFileDialog::getSaveFileName(m_treeWidget, tr("Save Path"), "", tr("BCP files (*.bcp)"));
    QFile pfile(filename);

    if (!pfile.open(QFile::WriteOnly))
    {
        return;
    }

    for(int row = 0; row < m_timeline->count(); row++)
    {
        LVRRecordedFrameItem* recordedFrame = static_cast<LVRRecordedFrameItem*>(m_timeline->item(row));
        recordedFrame->getFrame()->Print(std::cout);
    }

    pfile.close();
}

void LVRAnimationDialog::loadPath()
{
    QString filename = QFileDialog::getOpenFileName(m_treeWidget, tr("Load Path"), "", tr("BCP files (*.bcp)"));
    QFile pfile(filename);

    if (!pfile.open(QFile::ReadOnly))
    {
        return;
    }

    pfile.close();
}

}
