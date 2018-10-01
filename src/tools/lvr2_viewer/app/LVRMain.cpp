#include <QtGui>
#include "LVRMainWindow.hpp"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    lvr2::LVRMainWindow mainWindow;
    mainWindow.parseCommandLine(argc, argv);
    mainWindow.show();

    return app.exec();
}
