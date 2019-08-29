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

#include "lvr2/io/Hdf5IO.hpp"
#include "lvr2/io/hdf5/ArrayIO.hpp"
#include "lvr2/io/hdf5/ChannelIO.hpp"
#include "lvr2/io/hdf5/PointCloudIO.hpp"
#include "lvr2/io/hdf5/VariantChannelIO.hpp"
#include "lvr2/io/hdf5/MatrixIO.hpp"


#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/types/MatrixTypes.hpp"


/////////////////////
/// USEFUL EXTRA METHODS
//////////////////////

// https://www.fluentcpp.com/2018/08/28/removing-duplicates-crtp-base-classes/
// template<typename Derived, template<typename> typename ... Features>
// struct ExtraFeatures : Features<Derived>...
// { };


namespace lvr2 {


template<typename Derived,
         template<typename> typename ToCheck,
         template<typename> typename Current,
         template<typename> typename ... Features>
constexpr bool HasFeature()
{
  if constexpr( std::is_same<ToCheck<Derived>,Current<Derived>>::value )
    return true;
  else if constexpr( sizeof...(Features) == 0 )
    return false;
  else
    return HasFeature<Derived,ToCheck,Features...>();
}


template<typename ...>
struct JoinTwoHdf5IO;

template<template<typename> typename Feature,
         template<typename> typename ... Features1,
         template<typename> typename ... Features2>
struct JoinTwoHdf5IO<
    Hdf5IO<Features1...>,
    Hdf5IO<Feature,Features2...>
  >
{
  using type= typename
    std::conditional<
      HasFeature<Hdf5IO,Feature,Features1...>(),
      typename JoinTwoHdf5IO<
        Hdf5IO<Features1...>,
        Hdf5IO<Features2...>
      >::type,
      typename JoinTwoHdf5IO<
        Hdf5IO<Features1...,Feature>,
        Hdf5IO<Features2...>
      >::type
    >::type;
};

template<template<typename> typename ... Features1>
struct JoinTwoHdf5IO<
    Hdf5IO<Features1...>
  >
{
  using type= Hdf5IO<Features1...>;
};


} // namespace lvr2

void hdf5io_gen_example()
{
    // Build IO Type with IO features
    using BaseHDF5IO = lvr2::Hdf5IO<
        lvr2::hdf5features::ArrayIO,
        lvr2::hdf5features::ChannelIO
        >;

    // Extend IO with other features
    using MyHDF5IO = BaseHDF5IO::add_features<
            lvr2::hdf5features::VariantChannelIO,
            lvr2::hdf5features::PointCloudIO,
            lvr2::hdf5features::PointCloudIO // duplicate test
        >::type;


    // Check if a feature exists in IO Type
    if(MyHDF5IO::has_feature<lvr2::hdf5features::PointCloudIO>::value)
    {
        std::cout << "MyHDF5IO has the feature lvr2::hdf5features::PointCloudIO" << std::endl;
    } else {
        std::cout << "MyHDF5IO doenst have feature lvr2::hdf5features::PointCloudIO" << std::endl;
    }  

    // test duplicate feature
    using Duplicate = MyHDF5IO::add_feature<lvr2::hdf5features::PointCloudIO>::type;
    lvr2::Channel<double> channel(1000, 3);
    lvr2::PointBufferPtr pointcloud(new lvr2::PointBuffer);
    pointcloud->add("points", channel);
    Duplicate io;
    io.open("gen_test.h5");
    io.save("apointcloud",pointcloud);

}

void hdf5io_usage_example()
{
    ///////////////////////
    // Create Some Data  //
    ///////////////////////
    size_t num_data = 10000;
    boost::shared_array<float> data(new float[num_data]);
    lvr2::Channel<double> channel(1000, 3);

    ///////////////////////
    //   Create an IO    //
    ///////////////////////

    using MyHDF5IO = lvr2::Hdf5IO<
        lvr2::hdf5features::ArrayIO,
        lvr2::hdf5features::ChannelIO,
        lvr2::hdf5features::VariantChannelIO,
        lvr2::hdf5features::PointCloudIO,
        lvr2::hdf5features::MatrixIO
    >;

    MyHDF5IO my_io;
    my_io.open("test.h5");
    
    /////////////////////////
    // 1) ArrayIO Example  //
    /////////////////////////
    my_io.save("agroup", "anarray", num_data, data);
    size_t N;
    boost::shared_array<float> data_loaded = my_io.ArrayIO::load<float>("agroup", "anarray", N);
    
    if(N == num_data)
    {
        std::cout << "ArrayIO successful" << std::endl;
    }

    //////////////////////////
    // 2) ChannelIO Example //
    //////////////////////////
    my_io.save("agroup", "achannel", channel);
    lvr2::ChannelOptional<float> channel_loaded = my_io.ChannelIO::load<float>("agroup", "achannel");
    
    // alternative:
    // channel_loaded = my_io.loadChannel<float>("agroup","achannel");

    if(channel_loaded)
    {
        if(channel_loaded->numElements() == channel.numElements() && channel_loaded->width() == channel.width())
        {
            std::cout << "ChannelIO successful" << std::endl;
        }
    } else {
        std::cout << "channel not found" << std::endl;
    }

    /////////////////////////////
    // 3) PointCloudIO Example //
    /////////////////////////////
    lvr2::PointBufferPtr pointcloud(new lvr2::PointBuffer);
    pointcloud->add("points", channel);
    my_io.save("apointcloud", pointcloud);
    lvr2::PointBufferPtr pointcloud_loaded = my_io.loadPointCloud("apointcloud");
    if(pointcloud_loaded)
    {
        std::cout << "PointCloudIO read success" << std::endl;
        std::cout << *pointcloud_loaded << std::endl;
    } else {
        std::cout << "PointCloudIO read failed" << std::endl;
    }
    
    /////////////////////////////////
    // 4) VariantChannelIO Example //
    /////////////////////////////////

    using VChannel = lvr2::PointBuffer::val_type;

    auto ovchannel 
        = my_io.loadVariantChannel<VChannel>("apointcloud","points");
    
    if(ovchannel) {
        VChannel vchannel = *ovchannel;
        std::cout << "succesfully read VariantChannel: " << vchannel << std::endl;
    } else {
        std::cout << "could not load point from group apointcloud" << std::endl;
    }

    my_io.save("apointcloud", "points2", *ovchannel);

    /////////////////////////////////
    // 5) Eigen IO
    /////////////////////////////////

    lvr2::Transformd T = lvr2::Transformd::Identity();

    std::cout << "saving matrix:" << std::endl;
    std::cout << T << std::endl;

    my_io.save("matrices", "amatrix", T);

    auto T2 = my_io.MatrixIO::template load<lvr2::Transformd>("matrices", "amatrix");
    if(T2)
    {
        std::cout << "succesfully loaded Matrix:" << std::endl;
        std::cout << *T2 << std::endl;
    } else {
        std::cout << "could not load matrix!" << std::endl;
    }
}

int main(int argc, char** argv)
{
    hdf5io_gen_example();
    hdf5io_usage_example();

    return 0;
}