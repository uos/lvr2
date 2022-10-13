#include "Options.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/kernels/DirectoryKernel.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/io/scanio/DirectoryIO.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRdbx.hpp"

#include "lvr2/util/ScanProjectUtils.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;
using namespace lvr2::scanio;

int main(int argc, char** argv)
{
    // Parse options
    scanproject_parser::Options options(argc, argv);
    options.printLogo();

    // Load scan project (without fetching data)
    ScanProjectPtr inputProject = loadScanProject(options.getInputSchema(), options.getInputSource());

    // Save to target in new format
    saveScanProject(inputProject, options.getOutputSchema(), options.getOutputSource());

    // DirectoryKernelPtr kernel_in(new DirectoryKernel(options.getInputSource()));
    // DirectorySchemaPtr schema_in(new ScanProjectSchemaRdbx(options.getInputSource()));
    // DirectoryIO dirio_in(kernel_in, schema_in);

    // auto scanProject = dirio_in.ScanProjectIO::load();

    // DirectoryKernelPtr kernel_out(new DirectoryKernel(options.getOutputSource()));
    // DirectorySchemaPtr schema_out(new ScanProjectSchemaRawPly(options.getOutputSource()));
    // DirectoryIO dirio_out(kernel_out, schema_out);

    // dirio_out.ScanProjectIO::save(scanProject);



    return 0;
}
