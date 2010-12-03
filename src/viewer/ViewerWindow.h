/*
 * ViewerWindow.h
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#ifndef VIEWERWINDOW_H_
#define VIEWERWINDOW_H_

#include <QtGui>

#include "viewer.h"
#include "movedock.h"

#include "TouchPad.h"
#include "RenderFrame.h"
#include "EventHandler.h"
#include "ObjectHandler.h"


class EventHandler;
class RenderFrame;

class ViewerWindow : public Ui::MainWindow, public QMainWindow {

public:
	ViewerWindow();
	ViewerWindow(string filename);

	virtual ~ViewerWindow();
	virtual void resizeEvent(QResizeEvent* e);

	inline void renderScene();

	QListWidget* ListWidget(){return object_dialog_ui->listWidget;};

private:

	void setupDocks();
	void setupMainWindow();
	void setupConnections();


	QMainWindow* mainWindow;
	EventHandler* eventHandler;
	RenderFrame* renderFrame;

	QDockWidget* object_dialog;
	Ui::ObjectDialog* object_dialog_ui;
	TouchPad* touch_pad;

	bool first_resize;

};

inline void ViewerWindow::renderScene(){
}


#endif /* VIEWERWINDOW_H_ */
