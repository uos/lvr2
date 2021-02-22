#include "Logging.hpp"

#include "Options.hpp"

// #include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaSlam6D.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"

#include "lvr2/io/descriptions/ScanProjectSchemaRaw.hpp"

#include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/DirectoryKernel.hpp"
// #include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"

// #include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
// #include "lvr2/io/hdf5/ScanProjectIO.hpp"

#include "lvr2/util/Hdf5Util.hpp"
#include <boost/filesystem.hpp>

#include "lvr2/util/Synthetic.hpp"

#include <boost/type_index.hpp>

#include <unordered_map>
#include <unordered_set>

#include <boost/iostreams/code_converter.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include "Hdf5ReaderOld.hpp"
#include "ScanTypesCompare.hpp"

#include <random>
#include <chrono>

using namespace lvr2;

int main(int argc, char** argv)
{
    if(argc > 1)
    {
        std::string infilename = argv[1];
        std::cout << "Load file from '" << infilename << "' with old Hdf5 format." << std::endl;
        auto sp = loadOldHDF5(infilename);


        std::cout << "Construct new Hdf5IO." << std::endl;

        std::string outfilename = "scan_project.h5";
        HDF5KernelPtr kernel(new HDF5Kernel(outfilename));
        HDF5SchemaPtr schema(new ScanProjectSchemaHDF5());

        descriptions::HDF5IO hdf5io(kernel, schema);

        std::cout << "Write to '" << outfilename << "' with new Hdf5 format." << std::endl;
        hdf5io.save(sp);
    } else {
        std::cout << "please specify Hdf5 file that can be load with the old feature based Hdf5IO" << std::endl;
    }

    return 0;
}
