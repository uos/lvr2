
namespace lvr2 
{

// template<typename T, size_t I>
// void bla();

// Anker
template<typename Derived, typename VChannelT, size_t I,
    typename std::enable_if<I == 0, void>::type* = nullptr>
void store(
    const std::string& group,
    const std::string& name,
    const VChannelT& channel,
    const ChannelIO<Derived>* io
    )
{
    using StoreType = typename VChannelT::template type_of_index<I>;
    io->template save<StoreType>(group, name, 
            channel.template extract<StoreType>());
}

// Recursion
template<typename Derived, typename VChannelT, size_t I,
    typename std::enable_if<I != 0, void>::type* = nullptr>
void store(
    const std::string& group,
    const std::string& name,
    const VChannelT& channel,
    const ChannelIO<Derived>* io
)
{
    if(I == channel.type())
    {
        using StoreType = typename VChannelT::template type_of_index<I>;
        io->template save<StoreType>(group, name, 
            channel.template extract<StoreType>());

    } else {
        store<Derived, VChannelT, I-1>(group, name, channel, io);
    }
}

template<typename Derived, typename VChannelT>
void store(
    const std::string& group,
    const std::string& name,
    const VChannelT& channel,
    const ChannelIO<Derived>* io
)
{
    store<Derived, VChannelT, VChannelT::num_types - 1>(group, name, channel, io);
}

template<typename Derived>
template<typename ...Tp>
void VariantChannelIO<Derived>::save(
    std::string groupName,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel)
{
    // TODO

    using VChannelT = VariantChannel<Tp...>;

    std::cout << datasetName << " " << vchannel.type() << std::endl;

    YAML::Node node;
    node = vchannel;

    m_featureBase->m_kernel->saveMetaYAML(groupName, datasetName, node);



    store<Derived, VChannelT>(groupName, datasetName, vchannel, m_channel_io);

    // Each type?

    // Better with proper template specializations. But it works
    // for(size_t i=0; i<VChannelT::num_types; i++)
    // {
    //     using ZeroType = typename VChannelT::template type_of_index<0>;

    //     ZeroType bla;

    //     Channel<float> c;
    // }



    if(vchannel.template is_type<float>())
    {
        m_channel_io->template save<float>(groupName, datasetName, 
            vchannel.template extract<float>());
    }
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