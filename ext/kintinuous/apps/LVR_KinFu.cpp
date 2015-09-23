#include <boost/program_options.hpp>
// Program options for this tool
#include <kfusion/Options.hpp>
#include "KinFuApp.hpp"

namespace po = boost::program_options;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
	
    kfusion::Options* options;
    try
    {
		// Parse command line arguments
		options = new kfusion::Options(argc, argv);

		// Exit if options had to generate a usage message
		// (this means required parameters are missing)
		if ( options->printUsage() )
		{
			return 0;
		}

		::std::cout << *options << ::std::endl;
    }
    catch(exception& e)
    {
		cout << e.what() << endl;
		return 0;
    }
    
    OpenNISource capture;
    string device = options->getInputDevice();
    if(device.find_first_not_of("0123456789") == std::string::npos)
	{
		cuda::setDevice (atoi(device.c_str()));
		cuda::printShortCudaDeviceInfo (atoi(device.c_str()));

		if(cuda::checkIfPreFermiGPU(atoi(device.c_str())))
			return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;
		capture.open(atoi(device.c_str()));
	}
	else
	{
		capture.open(device);
		capture.triggerPause();
	}
	
    KinFuApp app (capture, options);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
