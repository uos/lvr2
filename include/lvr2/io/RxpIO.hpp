#pragma once

#include <string>

#include <lvr2/io/BaseIO.hpp>

#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Matrix4.hpp>

namespace lvr2 {

using Vec = BaseVector<float>;

class RxpIO : public BaseIO {

    public:
        ModelPtr read(std::string filename);
        ModelPtr read(std::string filename, int reduction_factor, Matrix4<Vec> transform);
        void save(std::string filename);
    
    private:
        bool check_error(int error_code);
};

} // namespace lvr2
