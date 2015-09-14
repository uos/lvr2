#include <boost/program_options.hpp>
#include "KinFuApp.cpp"

namespace po = boost::program_options;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
	
    string device = "0";
    string mesh_name = "mesh_output.ply";
    string config_file;
    bool no_reconstruct = false;
    bool optimize = false;
    bool no_viz = false;
    try
    {
		po::options_description generic("Allowed options");
		generic.add_options()
		  ("help,h", "produce help message")
		  ("config,c", po::value<string>(&config_file),
						"path to an optional configuration file in which the other parameters can be stored")
		;

		po::options_description config("Configuration parameters");
		config.add_options()
		  ("device,i", po::value(&device), "set RGB-D device or either a path to an oni file")
		  ("output,s", po::value(&mesh_name), "filename to save reconstructed mesh")
		  ("no_reconstruct,r", po::bool_switch(&no_reconstruct)->default_value(false), "set for no reconstruction, just recording")
		  ("optimize,o", po::bool_switch(&optimize)->default_value(false), "set for live mesh optimization")
		  ("no_viz,v", po::bool_switch(&no_viz)->default_value(false), "set for no live vizualisation, because it reduces gpu performance on really large scale reconstructions")
		;

		po::options_description cmdline_options;
		cmdline_options.add(generic).add(config);

		po::options_description config_file_options;
		config_file_options.add(config);

		po::positional_options_description p;
		p.add("device", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
		po::notify(vm);    

		if(vm.count("help")) {
		  cout << cmdline_options << "\n";
		  return 1;
		}
		if(vm.count("config"))
		{
		  ifstream ifs(config_file.c_str());
		  if(!ifs)
		  {
			cout << "Can't open config file." << endl;
			return 0;
		  }
		  else
		  {
			store(parse_config_file(ifs, config_file_options), vm);
			notify(vm);
		  }
		}
    }
    catch(exception& e)
    {
		cout << e.what() << endl;
		return 0;
    }
    
	cout << "#####################" << endl;
	cout << endl;
	if(device.find_first_not_of("0123456789") == std::string::npos)
	{
		cout << "Using device: " + device << endl; 
		cout << endl; 
	}
	else
	{
		cout << "Using ONI file: " + device << endl;
		cout << endl;
	}
	if(optimize)
	{
		
		cout << "Using online optimization " << endl;
		cout << endl;
	}
	if(no_reconstruct)
	{
		cout << "Online reconstruction will not be used " << endl;
		cout << endl;
		no_viz = true;
	}
	else
	{
		cout << "Saving mesh to: " + mesh_name << endl;
		cout << endl;
	}
	if(no_viz)
	{
		cout << "No live mesh vizualisation " << endl;
		cout << endl;
	}
	cout << "#####################" << endl;
    
    OpenNISource capture;
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
	
    KinFuApp app (capture, mesh_name, no_reconstruct, optimize, no_viz);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
