#pragma once

#include "types/Image.hpp"

namespace hdf5_features {

template<typename Derived>
class ImageIO
{
public:
    void save(const Image& image)
    {
        // uses derived.basicMethod()
        std::cout << "[ImageIO]: save() uses "<< m_file_access->filename() << std::endl;
    }

    Image load() {
        Image im;
        std::cout << "[ImageIO]: load() uses "<< m_file_access->filename() << std::endl;
        return im;
    }
private:
    Derived* m_file_access = static_cast<Derived*>(this);
};

} // namespace hdf5_features