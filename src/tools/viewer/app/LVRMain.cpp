#include <QtGui>
#include "LVRMainWindow.hpp"

#include "io/Progress.hpp"

using namespace lvr;

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    LVRMainWindow mainWindow;
    mainWindow.parseCommandLine(argc, argv);
    mainWindow.show();

    return app.exec();
}
