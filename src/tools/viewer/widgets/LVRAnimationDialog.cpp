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
    QObject::connect(m_dialog->saveVideo_button, SIGNAL(pressed()), this, SLOT(saveVideo()));
    QObject::connect(m_dialog->play_button, SIGNAL(pressed()), this, SLOT(play()));
}

void LVRAnimationDialog::addFrame()
{
    QString frameCount = QString("Frame no. %1").arg(++m_frameCounter);
    QListWidgetItem* currentFrame = new LVRRecordedFrameItem(m_pathCamera, frameCount);
    m_timeline->addItem(currentFrame);
}

void LVRAnimationDialog::removeFrame()
{
    QListWidgetItem* currentItem = m_timeline->currentItem();
    if(currentItem) delete currentItem;
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
   m_pathCamera->GetInterpolator()->Initialize();
}

void LVRAnimationDialog::play()
{
    unsigned int frameCount = m_timeline->count();

    // remove all cameras from the buffer and add every single one currently in the timeline
    m_pathCamera->InitializePath();
    for(int i = 0; i < frameCount; i++)
    {
        LVRRecordedFrameItem* recordedFrame = static_cast<LVRRecordedFrameItem*>(m_timeline->item(i));
        m_pathCamera->SetCamera(recordedFrame->getFrame());
        m_pathCamera->AddCameraToPath();
    }

    unsigned int frameMultiplier = m_dialog->frameMultiplier_box->value();
    m_pathCamera->SetNumberOfFrames(frameCount * frameMultiplier);

    // reset camera to main camera to play animation
    m_pathCamera->SetCamera(m_mainCamera);
    m_pathCamera->AnimatePath(m_renderWindowInteractor);
}

void LVRAnimationDialog::savePath()
{
    QString filename = QFileDialog::getSaveFileName(m_treeWidget, tr("Save Path"), "", tr("VCP files (*.vcp)"));
    QFile pfile(filename);

    if (!pfile.open(QFile::WriteOnly | QIODevice::Text))
    {
        return;
    }

    QTextStream out(&pfile);
    // save settings from the dialog to the first line
    QString interpolation = m_dialog->interpolation_box->currentText();
    QString transitionFrames = QString::number(m_dialog->frameMultiplier_box->value());
    out << "S:" << interpolation << ";" << transitionFrames << endl;

    // save settings for each camera in the timeline to a seperate line
    for(int row = 0; row < m_timeline->count(); row++)
    {
        LVRRecordedFrameItem* recordedFrame = static_cast<LVRRecordedFrameItem*>(m_timeline->item(row));
        recordedFrame->writeToStream(out);
    }

    pfile.close();
}

void LVRAnimationDialog::loadPath()
{
    QString filename = QFileDialog::getOpenFileName(m_treeWidget, tr("Load Path"), "", tr("VCP files (*.vcp)"));
    QFile pfile(filename);

    if (!pfile.open(QFile::ReadOnly | QIODevice::Text))
    {
        return;
    }

    QTextStream in(&pfile);
    QString line = in.readLine();

    // TODO: surround with try and catch to prevent errors
    // very basic file validity checking
    if(!line.startsWith("S:"))
    {
        cout << "Can't parse path settings from file!" << endl;
        cout << "Reverting to defaults..." << endl;
        m_dialog->frameMultiplier_box->setValue(30);
        m_dialog->interpolation_box->setCurrentIndex(1);
        m_pathCamera->GetInterpolator()->SetInterpolationTypeToSpline();
    }

    line.remove(0,2);
    QStringList parameters = line.trimmed().split(";");

    cout << parameters[0].toStdString() << endl;
    cout << parameters[1].toStdString() << endl;

    if(parameters[0] == "Linear")
    {
        m_pathCamera->GetInterpolator()->SetInterpolationTypeToLinear();
        m_dialog->interpolation_box->setCurrentIndex(0);
    }
    else if(parameters[0] == "Spline")
    {
        m_pathCamera->GetInterpolator()->SetInterpolationTypeToSpline();
        m_dialog->interpolation_box->setCurrentIndex(1);
    }

    int transitionFrames = parameters[1].toInt();
    m_dialog->frameMultiplier_box->setValue(transitionFrames);

    m_timeline->clear();

    // read each line in the file and create a camera from it; add it to the timeline
    while(!in.atEnd())
    {
        QListWidgetItem* recordedFrame = LVRRecordedFrameItem::createFromStream(in);
        m_timeline->addItem(recordedFrame);
    }

    pfile.close();

    m_frameCounter = m_timeline->count();
}

void LVRAnimationDialog::saveVideo()
{
    QString filename = QFileDialog::getSaveFileName(m_treeWidget, tr("Save Path"), "", tr("AVI files (*.avi)"));

    vtkSmartPointer<vtkFFMPEGWriter> videoWriter = vtkSmartPointer<vtkFFMPEGWriter>::New();
    videoWriter->SetQuality(2);
    videoWriter->SetRate(30);
    videoWriter->SetFileName(filename.toUtf8().constData());

    vtkSmartPointer<vtkWindowToImageFilter> w2i = vtkSmartPointer<vtkWindowToImageFilter>::New();
    w2i->SetInput(m_renderWindowInteractor->GetRenderWindow());
    videoWriter->SetInputConnection(w2i->GetOutputPort());

    videoWriter->Start();
    play(); // TODO: Capture video while playing!
    videoWriter->End();
}

}
