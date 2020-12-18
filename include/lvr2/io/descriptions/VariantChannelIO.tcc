
namespace lvr2 
{

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
template<typename VariantChannelT>
boost::optional<VariantChannelT> VariantChannelIO<Derived>::load(
    std::string groupName,
    std::string datasetName)
{
    boost::optional<VariantChannelT> ret;

    // check type of dataset
    int type_id = 0;

    // get datatype of group and name from kernel

    if(type_id == 0)
    {
        ret = m_channel_io->template load<float>(groupName, datasetName);
    } else {
        std::cout << "NOT" << std::endl;
    }
    // Construct Variant Channel

    return ret;
}

template<typename Derived>
template<typename VariantChannelT>
boost::optional<VariantChannelT> VariantChannelIO<Derived>::loadVariantChannel(
    std::string groupName,
    std::string datasetName)
{
    return load<VariantChannelT>(groupName, datasetName);
}

} // namespace lvr2 