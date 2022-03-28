#ifndef VARIANTCHANNELIO
#define VARIANTCHANNELIO

#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/baseio/ChannelIO.hpp"
#include "lvr2/io/YAML.hpp"

using lvr2::baseio::FeatureConstruct;

namespace lvr2 
{
namespace baseio
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

template<typename BaseIO>
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
    BaseIO* m_baseIO = static_cast<BaseIO*>(this);
    ChannelIO<BaseIO>* m_channel_io = static_cast<ChannelIO<BaseIO>*>(m_baseIO);
};

} // namespace scanio

/**
 * Define you dependencies here:
 */
template<typename T>
struct FeatureConstruct<lvr2::baseio::VariantChannelIO, T> {
    
    // DEPS
    using deps = typename FeatureConstruct<lvr2::baseio::ChannelIO, T>::type;

    // add actual feature
    using type = typename deps::template add_features<lvr2::baseio::VariantChannelIO>::type;
};


} // namespace baseio 

#include "VariantChannelIO.tcc"

#endif // VARIANTCHANNELIO
