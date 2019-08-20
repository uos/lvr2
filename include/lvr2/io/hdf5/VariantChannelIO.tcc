
namespace lvr2 {

namespace hdf5features {

template<typename Derived>
template<typename ...Tp>
void VariantChannelIO<Derived>::save(
    std::string groupName,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel)
{
    // TODO
}

template<typename Derived>
template<typename ...Tp>
void VariantChannelIO<Derived>::save(
    HighFive::Group& group,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel)
{

}

template<typename Derived>
template<typename ...Tp>
VariantChannelOptional<Tp...> VariantChannelIO<Derived>::load(
    std::string groupName,
    std::string datasetName)
{
    VariantChannelOptional<Tp...> ret;

    return ret;
}

template<typename Derived>
template<typename ...Tp>
VariantChannelOptional<Tp...> VariantChannelIO<Derived>::load(
    HighFive::Group& group,
    std::string datasetName)
{
    VariantChannelOptional<Tp...> ret;

    return ret;
}


} // hdf5features

} // namespace lvr2 