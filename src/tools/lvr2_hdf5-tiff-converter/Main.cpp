//
// Created by ndettmer on 11.12.18.
//

#include <lvr2/io/HDF5IO.hpp>

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
    vector<cv::Mat2d> channels;

    HDF5IO hdf5(filename);

    scandatas = hdf5.getScanData("/raw/scans", true);

    // for each scan
    for (ScanData scand: scandatas)
    {
        // build a CVMat for each Channel
        // TODO: find out how to iterate through m_points
/*        unsigned w_color;
        PointBufferPtr points = scand.m_points;
        for (auto point: points)
        {

        }*/
    }

    return 0;
}

