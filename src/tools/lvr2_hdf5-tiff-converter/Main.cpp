//
// Created by ndettmer on 31.01.19.
//

#include <lvr2/io/HDF5IO.hpp>
#include <lvr2/io/ScanDataManager.hpp>

#include <boost/range/iterator_range.hpp>
#include <boost/foreach.hpp>

#include <string>
#include <fstream>


using namespace lvr2;
using std::string;
using std::vector;

// DEBUG
using std::ifstream;
using std::cout;
using std::endl;

void printUsage()
{
    std::cout << "Please submit the path to a .h5 file." << std::endl;
    std::cout << "Usage: ./hdf5-tiff-converter <path>.h5" << std::endl;
}

int main(int argc, char**argv)
{
    if(!argv[1])
    {
        printUsage();
        return 0;
    }

    string filename = argv[1];
    HDF5IO hdf5(filename, false);
    std::vector<size_t> dim;

    // TODO: really use raw data?
    string groupname = "raw/spectral/position_00010";
    string datasetname = "spectral";
    boost::shared_array<unsigned int> spectrals = hdf5.getArray<unsigned int>(groupname, datasetname, dim);

    size_t num_channels = dim[0];
    size_t num_rows = dim[1];
    size_t num_cols = dim[2];

    // for each channel create a cv::Mat containing the spectral intensity data for the channel
    for(size_t channel = 0; channel < num_channels; channel++)
    {
        cv::Mat mat(num_rows, num_cols, CV_32SC1);
        for(size_t row = 0; row < num_rows; row++)
        {
            for(size_t col = 0; col < num_cols; col++)
            {
                mat.at<unsigned int>(row, col) = spectrals.get()[channel * num_cols * num_rows + row * num_cols + col];
            }
        }
    }

    return 0;
}
