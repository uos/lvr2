#include "Options.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/kernels/DirectoryKernel.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/io/scanio/DirectoryIO.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRdbx.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;
using namespace lvr2::scanio;

int main(int argc, char** argv)
{
    std::string dir_in("/home/praktikum/Desktop/chemnitz_2022-07-19.PROJ");

    DirectoryKernelPtr kernel_in(new DirectoryKernel(dir_in));
    DirectorySchemaPtr schema_in(new ScanProjectSchemaRdbx(dir_in));
    DirectoryIO dirio_in(kernel_in, schema_in);

    auto scanProject = dirio_in.ScanProjectIO::load();
    auto scanPosition = dirio_in.loadScanPosition(1);
    auto scan = dirio_in.loadScan(1, 0, 0);

    scan->points_loader();

    //std::cout << scan->points->numPoints() << std::endl;

    std::string dir_out("/home/praktikum/Desktop/Schematest");

    DirectoryKernelPtr kernel_out(new DirectoryKernel(dir_out));
    DirectorySchemaPtr schema_out(new ScanProjectSchemaRawPly(dir_out));
    DirectoryIO dirio_out(kernel_out, schema_out);

    dirio_out.ScanProjectIO::save(scanProject);
    return 0;
}
