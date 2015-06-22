#include <QtGui>
#include "LVRMainWindow.hpp"

#include "io/Progress.hpp"

using namespace lvr;

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    lvr::ProgressBar::enableDialog();

    LVRMainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
