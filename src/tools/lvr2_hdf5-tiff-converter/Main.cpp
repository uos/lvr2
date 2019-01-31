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
/*    vector<cv::Mat2d> channels;*/

    HDF5IO hdf5(filename, false);

    std::vector<size_t> dim;

    string groupname = "annotation/position_00010";
    string datasetname = "spectral";
    boost::shared_array<unsigned int> spectrals = hdf5.getArray<unsigned int>(groupname, datasetname, dim);

    for(size_t size: dim)
    {
        cout << size << endl;
    }

    return 0;
}
