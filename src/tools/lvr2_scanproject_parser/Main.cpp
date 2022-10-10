#include "Options.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/util/ScanProjectUtils.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;

int main(int argc, char** argv)
{
    // Parse options
    scanproject_parser::Options options(argc, argv);
    options.printLogo();

    // Laod scan project (without fetching data)
    ScanProjectPtr inputProject = loadScanProject(options.getInputSchema(), options.getInputSource());

    // Save to target in new format
    saveScanProject(inputProject, options.getOutputSchema(), options.getOutputSource());

    return 0;
}
