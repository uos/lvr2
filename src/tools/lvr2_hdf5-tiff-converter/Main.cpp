/**
 * @brief main file for HDF5 to GeoTIFF conversion
 * @author ndettmer <ndettmer@uos.de>
 */

#include <lvr2/io/HDF5IO.hpp>
#include <lvr2/io/ScanDataManager.hpp>

#include <boost/range/iterator_range.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <string>
#include <fstream>

#include <sys/stat.h>

#include "GeoTIFFIO.hpp"


using namespace lvr2;

void printUsage()
{
    std::cout << "Please submit the path to a .h5 file." << std::endl;
    std::cout << "Usage: ./hdf5-tiff-converter <input_path>.h5 <position code> [<output_path>.tif]" << std::endl;
}

/**
 * @brief Extraction of radiometric data from a given HDF5 file into a new GeoTIFF file in given output path
 * @param input_filename Path to the input HDF5 file formatted due to lvr_2 convention
 * @param position_code 5 character code of the scan position (e.g. 00000)
 * @param output_filename Path to the output GeoTIFF file
 * @return standard C++ return value
 */
int processConversion(std::string input_filename, std::string position_code, std::string output_filename)
{
    /* =----------------- HDF5 INPUT ----------------------- */
    HDF5IO hdf5(input_filename, false);
    std::vector<size_t> dim;

    // extract radiometric data
    // TODO: really use raw data?
    std::string groupname = "raw/spectral/position_" + position_code;
    std::string datasetname = "spectral";
    boost::shared_array<uint16_t> spectrals = hdf5.getArray<uint16_t>(groupname, datasetname, dim);

    // extract array dimension information
    size_t num_channels = dim[0];
    size_t num_rows = dim[1];
    size_t num_cols = dim[2];

    GeoTIFFIO gtifio(output_filename, num_cols, num_rows, num_channels);

    /* -------------- FILE CONVERSION ------------------- */
    // for each channel create a cv::Mat containing the spectral intensity data for the channel ...
    for(size_t channel = 0; channel < num_channels; channel++)
    {
        cv::Mat *mat = new cv::Mat(num_rows, num_cols, CV_16UC1);
        for(size_t row = 0; row < num_rows; row++)
        {
            for(size_t col = 0; col < num_cols; col++)
            {
                mat->at<uint16_t>(row, col) = spectrals.get()[channel * num_cols * num_rows + row * num_cols + col];
            }
        }
        // ... and write it to the output GeoTIFF file
        int ret = gtifio.writeBand(mat, channel + 1);
        if (ret != 0)
        {
            return ret;
        }
    }

}

int main(int argc, char**argv)
{
    /* ---------------- COMMAND LINE INPUT ----------------- */
    if(!argv[1] || argv[1] == "--help" || !argv[2])
    {
        printUsage();
        return 0;
    }
    boost::filesystem::path input_filename(argv[1]);
    string position_code = argv[2];

    string output_filename_str = "../../../../lvr_output/out.tif";
    if(argv[3])
    {
        output_filename_str = argv[3];
    }
    boost::filesystem::path output_filename(output_filename_str);

    boost::filesystem::path output_dir = output_filename.parent_path();
    if (!boost::filesystem::exists(output_dir))
    {
        boost::filesystem::create_directory(output_dir);
    }

    if (processConversion(input_filename.string(), position_code, output_filename.string()) < 0)
    {
        std::cout << "An Error occurred during conversion." << std::endl;
    }

    return 0;
}
