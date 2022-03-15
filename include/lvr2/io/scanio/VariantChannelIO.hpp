#pragma once
#ifndef VARIANTCHANNELIO
#define VARIANTCHANNELIO

#include "lvr2/io/scanio/FeatureBase.hpp"
#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/io/scanio/yaml/VariantChannel.hpp"

// Dependencies
#include "lvr2/io/scanio/ChannelIO.hpp"


namespace lvr2 
{

namespace scanio
{

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

template<typename FeatureBase>
class VariantChannelIO {
public:

    template<typename ...Tp>
    void save(  std::string groupName,
                std::string datasetName,
                const VariantChannel<Tp...>& vchannel) const;
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> load(
                std::string groupName,
                std::string datasetName) const;
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> loadVariantChannel(
                std::string groupName, 
                std::string datasetName) const;

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    ChannelIO<FeatureBase>* m_channel_io = static_cast<ChannelIO<FeatureBase>*>(m_featureBase);
};

} // namespace scanio

/**
 * Define you dependencies here:
 */
template<typename FeatureBase>
struct FeatureConstruct<lvr2::scanio::VariantChannelIO, FeatureBase> {
    
    // DEPS
    using deps = typename FeatureConstruct<lvr2::scanio::ChannelIO, FeatureBase>::type;

    // add actual feature
    using type = typename deps::template add_features<lvr2::scanio::VariantChannelIO>::type;
};


} // namespace lvr2 

#include "VariantChannelIO.tcc"

#endif // VARIANTCHANNELIO
