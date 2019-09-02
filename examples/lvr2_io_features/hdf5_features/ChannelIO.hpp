#pragma once

#include "types/Channel.hpp"

namespace hdf5_features {

template<typename Derived>
class ChannelIO
{
public:
    void save(const Channel& channel)
    {
        // uses derived.basicMethod()
        
        std::cout << "[ChannelIO]: save() uses "<< m_file_access->filename() << std::endl;
    }

    Channel load()
    {
        Channel ch;
        std::cout << "[ChannelIO]: load() uses " << m_file_access->filename() << std::endl;
        return ch;
    }

private:
    // build required members
    Derived* m_file_access = static_cast<Derived*>(this);


};

} // namespace hdf5_features