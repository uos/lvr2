//
// Created by ndettmer on 31.01.19.
//

#include <lvr2/io/HDF5IO.hpp>
#include <lvr2/io/ScanDataManager.hpp>

#include <boost/range/iterator_range.hpp>
#include <boost/foreach.hpp>

#include <tiffio.h>

#include <string>
#include <fstream>

#include "TIFFIO.hpp"
#include "GeoTIFFIO.hpp"


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
    std::cout << "Usage: ./hdf5-tiff-converter <input_path>.h5 [<output_path>.tif]" << std::endl;
}

int processConversion(std::string input_filename, std::string output_filename)
{
    /* =----------------- HDF5 INPUT ----------------------- */
    HDF5IO hdf5(input_filename, false);
    std::vector<size_t> dim;

    // TODO: really use raw data?
    string groupname = "raw/spectral/position_00010";
    string datasetname = "spectral";
    boost::shared_array<uint8_t> spectrals = hdf5.getArray<uint8_t>(groupname, datasetname, dim);

    size_t num_channels = dim[0];
    size_t num_rows = dim[1];
    size_t num_cols = dim[2];

    /* ---------------- TIFF CONFIG ------------------------*/
    // TODO: ganzes Verzeichnis in TIFF Directory schreiben
    TIFFIO tif(output_filename);
    tif.setFields(num_rows, num_cols, 1);


    /* -------------- FILE CONVERSION ------------------- */
    // for each channel create a cv::Mat containing the spectral intensity data for the channel
    for(size_t channel = 0; channel < num_channels; channel++)
    {
        cv::Mat *mat = new cv::Mat(num_rows, num_cols, CV_8UC1);
        for(size_t row = 0; row < num_rows; row++)
        {
            for(size_t col = 0; col < num_cols; col++)
            {
                mat->at<uint8_t>(row, col) = spectrals.get()[channel * num_cols * num_rows + row * num_cols + col];
            }
        }
        if (channel == 15)
        {
            tif.writePage(mat, channel + 1);
        }
    }

}

int main(int argc, char**argv)
{
    /* ---------------- COMMAND LINE INPUT ----------------- */
    if(!argv[1] || argv[1] == "--help")
    {
        printUsage();
        return 0;
    }
    string input_filename = argv[1];

    string output_filename = "../../../../lvr_output/out.tif";
    if(argv[2])
    {
        output_filename = argv[2];
    }

    GeoTIFFIO gtif(output_filename);
    return 0;

/*    return processConversion(input_filename, output_filename);*/
}
