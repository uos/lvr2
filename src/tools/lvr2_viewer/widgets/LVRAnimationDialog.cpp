/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <QFileDialog>
#include "LVRAnimationDialog.hpp"

//#include <vtkFFMPEGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <cstring>

namespace lvr2
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

    QObject::connect(m_dialog->timeline_list, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(goToCamPosition(QListWidgetItem*)));
}

void LVRAnimationDialog::goToCamPosition(QListWidgetItem *item)
{
    LVRRecordedFrameItem* cam_item = static_cast<LVRRecordedFrameItem*>(item);
    if(cam_item)
    {
        vtkRenderWindow* window = m_renderWindowInteractor->GetRenderWindow();
        vtkRendererCollection* renderers = window->GetRenderers();
        renderers->InitTraversal();
        vtkRenderer* r = renderers->GetNextItem();
        while(r != 0)
        {
            vtkCamera* ac = r->GetActiveCamera();
            vtkCamera* frame = cam_item->getFrame();

            ac->SetPosition(frame->GetPosition());
            ac->SetFocalPoint(frame->GetFocalPoint());
            ac->SetViewUp(frame->GetViewUp());

            m_renderWindowInteractor->Render();
            r = renderers->GetNextItem();
        }

    }
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
    QFileDialog dialog(m_treeWidget);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setOption(QFileDialog::Option::DontUseNativeDialog, true);
    dialog.setNameFilter("*.vcp");

    if(!dialog.exec())
    {
        return;
    }

    QStringList files = dialog.selectedFiles();
    QString filename =  files.front();
    QFile pfile(filename);
    if (!pfile.open(QFile::ReadOnly | QIODevice::Text))
    {
        cout << "Error opening file " << filename.toStdString() << endl;
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
//#ifdef VTK_USE_FFMPEG_ENCODER
	QString filename = QFileDialog::getSaveFileName(m_treeWidget, tr("Save Path"), "", tr("AVI files (*.avi)"));

	this->m_renderWindowInteractor->GetRenderWindow()->SetOffScreenRendering( 1 );

	vtkCameraInterpolator* i =  m_pathCamera->GetInterpolator();

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

	double minT = i->GetMinimumT();
	double maxT = i->GetMaximumT();
    int n = frameMultiplier * frameCount;
	double step = (maxT - minT) / (double)n;
	vtkSmartPointer<vtkCamera> i_cam = vtkSmartPointer<vtkCamera>::New();




	vtkRendererCollection* collection = m_renderWindowInteractor->GetRenderWindow()->GetRenderers();
	vtkRenderer* renderer;
	char buffer[256];
	int c = 0;
	double range[2];
	for(double camT = 0; camT < maxT; camT += step)
	{
		i->InterpolateCamera(camT, i_cam);

		collection->InitTraversal();
		renderer = collection->GetNextItem();
		while(renderer)
		{
			renderer->GetActiveCamera()->DeepCopy(i_cam);
			m_renderWindowInteractor->GetRenderWindow()->Render();
			renderer = collection->GetNextItem();
		}

		vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
			    vtkSmartPointer<vtkWindowToImageFilter>::New();

			windowToImageFilter->SetInput(m_renderWindowInteractor->GetRenderWindow());
		windowToImageFilter->Update();

		  vtkSmartPointer<vtkPNGWriter> writer =
		    vtkSmartPointer<vtkPNGWriter>::New();

		  sprintf(buffer, "frame%04d.png", c);

		  writer->SetFileName(buffer);
		  writer->SetInputConnection(windowToImageFilter->GetOutputPort());
		  writer->Write();

        cout << c << " / " << frameCount * frameMultiplier << endl;
		c++;


	}



	  this->m_renderWindowInteractor->GetRenderWindow()->SetOffScreenRendering( 0 );



   /* vtkSmartPointer<vtkFFMPEGWriter> videoWriter = vtkSmartPointer<vtkFFMPEGWriter>::New();
    videoWriter->SetQuality(2);
    videoWriter->SetRate(30);
    videoWriter->SetFileName(filename.toUtf8().constData());

    vtkSmartPointer<vtkWindowToImageFilter> w2i = vtkSmartPointer<vtkWindowToImageFilter>::New();
    w2i->SetInput(m_renderWindowInteractor->GetRenderWindow());
    videoWriter->SetInputConnection(w2i->GetOutputPort());

    videoWriter->Start();
    play(); // TODO: Capture video while playing!
    videoWriter->End(); */
}

} // namespace lvr2
