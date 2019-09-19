#pragma once

#include "types/Hyper.hpp"
#include "ImageIO.hpp"

namespace hdf5_features {

template<typename Derived>
class HyperIO
{
public:
    void save(const Hyper& hyper)
    {
        auto& derived = static_cast<Derived&>(*this);
        // uses derived.basicMethod()
        std::cout << "[HyperIO]: save() uses " << m_file_access->filename() << std::endl;
    
        Image im = m_image_io->load();
        std::cout << "[HyperIO]: can use ImageIO as well" << std::endl;
    }

    Hyper load() {
        ::Hyper hy;
        std::cout << "[HyperIO]: load() uses " << m_file_access->filename() << std::endl;
        return hy;
    }
private:
    Derived* m_file_access = static_cast<Derived*>(this);

    // feature dependencies
    ImageIO<Derived>* m_image_io = static_cast<ImageIO<Derived>*>(m_file_access);
};

} // namespace hdf5_features