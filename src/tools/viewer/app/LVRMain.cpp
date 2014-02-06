#include <QtGui>
#include "LVRMainWindow.hpp"

using namespace lvr;

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    LVRMainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
