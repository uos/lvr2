#pragma once
#ifndef LVR2_IO_HDF5_VARIANTCHANNELIO_HPP
#define LVR2_IO_HDF5_VARIANTCHANNELIO_HPP

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

// Object to store
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/types/VariantChannel.hpp"

// Dependencies
#include "ChannelIO.hpp"

namespace lvr2 {

namespace hdf5features {

/**
 * @class VariantChannelIO 
 * @brief Hdf5IO Feature for handling VariantChannel related IO
 * 
 * This Feature of the Hdf5IO handles the IO of a VariantChannel object.
 * 
 * Example:
 * @code
 * 
 * MyHdf5IO io;
 * 
 * // example data
 * using MultiChannel = VariantChannel<float, char, int>;
 * MultiChannel vchannel, vchannel_in;
 * Channel<float> samples(100,100);
 * vchannel = samples;
 * 
 * // writing
 * io.open("test.h5");
 * io.save("avariantchannel", vchannel);
 * 
 * // reading
 * vchannel_in = *io.loadVariantChannel<MultiChannel>("avariantchannel");
 * 
 * // if the type is known you can also load via ChannelIO
 * vchannel_in = *io.loadChannel<float>("avariantchannel");
 * 
 * @endcode
 * 
 * Dependencies:
 * - ChannelIO
 * 
 */

struct ConstructType {

};

template<typename Derived = ConstructType>
class VariantChannelIO {
public:

    template<typename ...Tp>
    void save(std::string groupName, std::string datasetName, const VariantChannel<Tp...>& vchannel);
    
    template<typename ...Tp>
    void save(HighFive::Group& group, std::string datasetName, const VariantChannel<Tp...>& vchannel);
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> load(std::string groupName, std::string datasetName);
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> load(HighFive::Group& group, std::string datasetName);
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> loadVariantChannel(std::string groupName, std::string datasetName);

protected:

    template<typename VariantChannelT>
    boost::optional<VariantChannelT> loadDynamic(HighFive::DataType dtype,
        HighFive::Group& group,
        std::string name);

    template<typename ...Tp>
    void saveDynamic(HighFive::Group& group,
        std::string datasetName,
        const VariantChannel<Tp...>& vchannel
    );

    Derived* m_file_access = static_cast<Derived*>(this);
    ChannelIO<Derived>* m_channel_io = static_cast<ChannelIO<Derived>*>(m_file_access);
};

} // hdf5features

/**
 * Define you dependencies here:
 */
template<typename BaseIO>
struct Hdf5Construct<hdf5features::VariantChannelIO, BaseIO> {
    
    // DEPS
    using deps = typename Hdf5Construct<hdf5features::ChannelIO, BaseIO>::type;

    // add actual feature
    using type = typename deps::template add_features<hdf5features::VariantChannelIO>::type;
};




// generator

} // namespace lvr2 

#include "VariantChannelIO.tcc"

#endif // LVR2_IO_HDF5_VARIANTCHANNELIO_HPP
