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

int main()
{
    string res_path = "../../../../res/";
    string file = "botanischer_garten.h5";
    string filename = res_path + file;

    vector<ScanData> scandatas;

    HDF5IO hdf5(filename, false);

    std::vector<size_t> dim;

    // TODO: really use raw data?
    string groupname = "raw/spectral/position_00010";
    string datasetname = "spectral";
    boost::shared_array<unsigned int> spectrals = hdf5.getArray<unsigned int>(groupname, datasetname, dim);

    size_t num_channels = dim[0];
    size_t num_points_y = dim[1];
    size_t num_points_x = dim[2];

    // debug
    num_channels = 1;
    num_points_y = 5; // TODO: causes segfault if real value is assigned
/*    num_points_x = 5;*/

    for(int channel = 0; channel < num_channels; channel++)
    {
        unsigned int data[num_points_y * num_points_x];
        for(int i = 0; i < num_points_y * num_points_x; i++)
        {
            // pass auf deinen Ram auf!!
            data[i] = spectrals.get()[(channel + 1) * i];
        }
        cv::Mat mat(num_points_y, num_points_x, CV_32SC1, data);

        cout << mat << endl;

    }

    return 0;
}
