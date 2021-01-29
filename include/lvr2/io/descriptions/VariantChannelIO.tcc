
namespace lvr2 
{

// Anker
template<typename Derived, typename VChannelT, size_t I,
    typename std::enable_if<I == 0, void>::type* = nullptr>
void saveDynamic(
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
void saveDynamic(
    const std::string& group,
    const std::string& name,
    const VChannelT& channel,
    const ChannelIO<Derived>* io)
{
    if(I == channel.type())
    {
        using StoreType = typename VChannelT::template type_of_index<I>;
        io->template save<StoreType>(group, name, 
            channel.template extract<StoreType>());
    } else {
        saveDynamic<Derived, VChannelT, I-1>(group, name, channel, io);
    }
}

template<typename Derived, typename VChannelT>
void saveDynamic(
    const std::string& group,
    const std::string& name,
    const VChannelT& channel,
    const ChannelIO<Derived>* io)
{
    saveDynamic<Derived, VChannelT, VChannelT::num_types - 1>(group, name, channel, io);
}

template<typename Derived>
template<typename ...Tp>
void VariantChannelIO<Derived>::save(
    std::string groupName,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel)
{
    // std::cout << "[VariantChannelIO - save] " << groupName << ", " << datasetName << ", " << vchannel.typeName() << std::endl;
    using VChannelT = VariantChannel<Tp...>;

    // keep this order! We need Hdf5 to build the dataset first, then writing meta information
    saveDynamic<Derived, VChannelT>(groupName, datasetName, vchannel, m_channel_io);
    
    // creating meta node of variantchannel containing type and size
    YAML::Node node;
    try {
        node = vchannel;
    } catch(YAML::TypedBadConversion<int> ex) {
        std::cout << ex.what() << std::endl;
    }
    m_featureBase->m_kernel->saveMetaYAML(groupName, datasetName, node);
}

// anker
template<typename Derived, typename VariantChannelT, size_t I,
    typename std::enable_if<I == 0, void>::type* = nullptr>
bool _dynamicLoad(
    std::string group, std::string name,
    std::string dyn_type, 
    VariantChannelT& vchannel,
    const ChannelIO<Derived>* io)
{
    using DataT = typename VariantChannelT::template type_of_index<I>;

    if(dyn_type == Channel<DataT>::typeName() )
    {
        using DataT = typename VariantChannelT::template type_of_index<I>;

        ChannelOptional<DataT> copt = io->template load<DataT>(group, name);
        if(copt)
        {
            vchannel = *copt;
        } else {
            std::cout << "[VariantChannelIO] WARNING: Could not reveive Channel from ChannelIO!" << std::endl;
            return false;
        }

        return true;
    }

    std::cout << "[VariantChannelIO] WARNING: data type '" << dyn_type << "' not implemented in PointBuffer." << std::endl;
    return false;
}

template<typename Derived, typename VariantChannelT, size_t I, 
    typename std::enable_if<I != 0, void>::type* = nullptr>
bool _dynamicLoad(
    std::string group, std::string name,
    std::string dyn_type, 
    VariantChannelT& vchannel,
    const ChannelIO<Derived>* io)
{
    using DataT = typename VariantChannelT::template type_of_index<I>;

    if(dyn_type == Channel<DataT>::typeName() )
    {
        ChannelOptional<DataT> copt = io->template load<DataT>(group, name);
        if(copt)
        {
            vchannel = *copt;
        } else {
            std::cout << "[VariantChannelIO] WARNING: Could not receive Channel from ChannelIO!" << std::endl;
            return false;
        }
        
        return true;
    } else {
        return _dynamicLoad<Derived, VariantChannelT, I-1>(group, name, dyn_type, vchannel, io);
    }
}

template<typename Derived, typename VariantChannelT>
bool dynamicLoad(
    std::string group, std::string name,
    std::string dyn_type, 
    VariantChannelT& vchannel,
    const ChannelIO<Derived>* io)
{
    return _dynamicLoad<Derived, VariantChannelT, VariantChannelT::num_types-1>(group, name, dyn_type, vchannel, io);
}

template<typename Derived>
template<typename VariantChannelT>
boost::optional<VariantChannelT> VariantChannelIO<Derived>::load(
    std::string groupName,
    std::string datasetName)
{

    // std::cout << "[VariantChannelIO - load] " << groupName << ", " << datasetName << std::endl;

    boost::optional<VariantChannelT> ret;

    YAML::Node node;
    m_featureBase->m_kernel->loadMetaYAML(groupName, datasetName, node);

    lvr2::MultiChannel mc;
    if(!YAML::convert<lvr2::MultiChannel>::decode(node, mc))
    {
        // fail
        std::cout << timestamp << "[VariantChannelIO - load] Tried to load Meta information that does not suit to VariantChannel types" << std::endl;
        return ret;
    }

    std::string data_type = node["data_type"].as<std::string>();
    
    // load channel with correct datatype
    VariantChannelT vchannel;
    if(dynamicLoad<Derived, VariantChannelT>(
        groupName, datasetName,
        data_type, vchannel, m_channel_io))
    {
        ret = vchannel;
    } else {
        std::cout << "[VariantChannelIO] Error occured while loading group '" << groupName << "', dataset '" << datasetName <<  "'" << std::endl;
    }

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