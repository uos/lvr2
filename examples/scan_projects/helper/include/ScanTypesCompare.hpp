#ifndef LVR2_SCAN_TYPES_COMPARE_HPP

#include "lvr2/types/ScanTypes.hpp"

namespace lvr2 {

bool equal(const float& a, const float& b);
bool equal(const double& a, const double& b);

bool equal(CameraImagePtr si1, CameraImagePtr si2);

bool equal(CameraImageGroupPtr si1, CameraImageGroupPtr si2);

bool equal(CameraImageOrGroup sg1, CameraImageOrGroup sg2);

bool equal(CameraPtr sc1, CameraPtr sc2);

bool equal(PointBufferPtr p1, PointBufferPtr p2);

bool equal(ScanPtr s1, ScanPtr s2);

bool equal(LIDARPtr l1, LIDARPtr l2);

bool equal(ScanPositionPtr sp1, ScanPositionPtr sp2);

bool equal(ScanProjectPtr sp1, ScanProjectPtr sp2);

template<typename T>
bool equal(const Channel<T>& c1, const Channel<T>& c2)
{
    
    if(c1.numElements() != c2.numElements())
    {
        return false;
    }

    if(c1.width() != c2.width())
    {
        return false;
    }

    for(size_t i = 0; i < c1.numElements(); i++)
    {
        for(size_t j = 0; j < c1.width(); j++)
        {
            if(c1[i][j] != c2[i][j])
            {
                std::cout << "Entry differs: " << c1[i][j] << " != " << c2[i][j] << std::endl;
                return false;
            }
        }
    }

    return true;
}

} // namespace lvr2

#endif // LVR2_SCAN_TYPES_COMPARE_HPP