#ifndef LVR2_UTIL_HPP
#define LVR2_UTIL_HPP

#include <vector>
#include <boost/shared_array.hpp>

#include <lvr2/io/PointBuffer2.hpp>

#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>

namespace lvr2
{

using VecUChar = BaseVector<unsigned char>;

class Util
{
public:

    template<typename T>
    static boost::shared_array<T> convert_vector_to_shared_array(std::vector<T> source)
    {
        boost::shared_array<T> ret = boost::shared_array<T>( new T[source.size()] );
        std::copy(source.begin(), source.end(), ret.get());

        return ret;
    }

    static int getSpectralChannel(int wavelength, PointBuffer2Ptr pcloud, int fallback = -1);

    struct ColorVecCompare
    {
        bool operator() (const Vector<VecUChar>& lhs, const Vector<VecUChar>& rhs) const
        {
            return (lhs.x < rhs.x) || 
                   (lhs.x == rhs.x && lhs.y < rhs.y) || 
                   (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z); 
        }
    };
};


} // namespace lvr2

#endif
