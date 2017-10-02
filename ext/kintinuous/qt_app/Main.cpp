/*
 * Main.cpp
 *
 *  Created on: Jan 26, 2016
 *      Author: twiemann
 */

#include <QtGui>
#include "MainWindow.hpp"

using namespace std;

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    MainWindow mainWindow;
    mainWindow.show();
    return app.exec();
}
