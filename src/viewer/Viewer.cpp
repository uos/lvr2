#include <stdlib.h>
#include <QtGui/QApplication>

#include "ViewerWindow.h"
#include "EventHandler.h"

using Ui::MainWindow;

int main(int argc, char** argv){

	QApplication app(argc, argv);
	ViewerWindow* mainWindow = 0;

	if(argc == 1){
		mainWindow = new ViewerWindow();
	} else if (argc == 2){
		string filename(argv[1]);
		mainWindow = new ViewerWindow(filename);
	}


	app.exec();

	delete mainWindow;

	return 0;
}
