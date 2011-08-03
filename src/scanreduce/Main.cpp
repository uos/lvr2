#include "Options.hpp"

#include "io/UosIO.hpp"

#include <iostream>

using namespace lssr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    reduce::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage()) return 0;

    ::std::cout<<options<<::std::endl;

    UosIO<float> io;
    io.setFirstScan(options.firstScan());
    io.setLastScan(options.lastScan());
    io.reduce(options.directory(), options.outputFile(), options.reduction());

	return 0;
}

