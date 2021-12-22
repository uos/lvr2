#include "Options.hpp"
#include "lvr2/io/scanio/FeatureBase.hpp"
#include "lvr2/io/scanio/DirectoryKernel.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/io/scanio/DirectoryIO.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;
using namespace lvr2::scanio;

int main(int argc, char** argv)
{
    std::string dir_in("/home/twiemann/data/example_scanproject_botgarden");

    DirectoryKernelPtr kernel_in(new DirectoryKernel(dir_in));
    DirectorySchemaPtr schema_in(new ScanProjectSchemaRawPly(dir_in));
    DirectoryIO dirio_in(kernel_in, schema_in);

    auto scanProject = dirio_in.ScanProjectIO::load();

    std::string dir_out("/home/twiemann/data/example_scanproject_botgarden_converted");
    DirectoryKernelPtr kernel_out(new DirectoryKernel(dir_out));
    DirectorySchemaPtr schema_out(new ScanProjectSchemaRaw(dir_out));
    DirectoryIO dirio_out(kernel_out, schema_out);

    dirio_out.ScanProjectIO::save(scanProject);

    return 0;
}
