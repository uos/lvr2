#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>
#include <type_traits>
#include <utility>

#include <boost/optional.hpp>
#include <boost/preprocessor/stringize.hpp>

// lvr2 includes
#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/types/MatrixTypes.hpp"

#include "hdf5_features/ChannelIO.hpp"
#include "hdf5_features/HyperIO.hpp"
#include "hdf5_features/ImageIO.hpp"

#include "types/Channel.hpp"
#include "types/Image.hpp"
#include "types/Hyper.hpp"

template<template<typename> typename ...Features>
class HDF5IO : public Features<HDF5IO<Features...> >...
{
public:
    static constexpr std::size_t N = sizeof...(Features);
    using feature_tuple = std::tuple<Features<HDF5IO<Features...> >...>;
    using index_sequence = std::make_index_sequence<N>;

    #if __cplusplus <= 201500L
        // feature of c++17
        #pragma message("using Tp::save... needs c++17 at least or a newer compiler")
    #endif

    using Features<HDF5IO<Features...> >::save...;

    void open(std::string filename)
    {
        std::cout << index_sequence::size() << std::endl;
        m_filename = filename;
    }

    std::string filename() {
        return m_filename;
    }

protected:
    std::string m_filename;
};


/////////////////////
/// USEFUL EXTRA METHODS
//////////////////////

// https://www.fluentcpp.com/2018/08/28/removing-duplicates-crtp-base-classes/
// template<typename Derived, template<typename> typename ... Features>
// struct ExtraFeatures : Features<Derived>...
// { };

// template<typename Derived>
// using ExtraFeaturesA = ExtraFeatures<Derived,ExtraFeature1,ExtraFeature2>;
 
// template<typename Derived>
// using ExtraFeaturesB = ExtraFeatures<Derived,ExtraFeature2,ExtraFeature3>;
 
// template<typename Derived>
// using ExtraFeaturesC = ExtraFeatures<Derived,ExtraFeature1,ExtraFeature3>;


int main(int argc, char** argv)
{
    Channel ch;
    Image im;
    Hyper hy;

    using MyHDF5IO = HDF5IO<hdf5_features::ChannelIO, hdf5_features::ImageIO, hdf5_features::HyperIO>;
    MyHDF5IO io;

    io.open("hello.h5");

    io.save(ch);
    io.save(im);
    io.save(hy);

    ch = io.ChannelIO::load();
    im = io.ImageIO::load();
    hy = io.HyperIO::load();


}