#include <iostream>
using namespace std;

#include <QApplication>
#include <QMainWindow>

#include "MainWindow.h"

#include "ViewerApplication.h"

using Ui::MainWindow;

int main(int argc, char** argv)
{

	// Create application object
	QApplication application(argc, argv);


	ViewerApplication vapp(argc, argv);

	// Run application
	return application.exec();
}
