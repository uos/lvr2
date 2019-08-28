
namespace lvr2 {

namespace hdf5features {

// R == 0
template<typename Derived, typename VariantT, int R, typename std::enable_if<R == 0, void>::type* = nullptr>
void saveVChannel(
    const VariantT& vchannel,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(R == vchannel.type())
    {
        channel_io->save(group, name, vchannel.template extract<typename VariantT::template type_of_index<R> >() );
    } else {
        std::cout << "[VariantChannelIO] WARNING: Nothing was saved" << std::endl;
    }
}

// R != 0
template<typename Derived, typename VariantT, int R, typename std::enable_if<R != 0, void>::type* = nullptr>
void saveVChannel(
    const VariantT& vchannel,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(R == vchannel.type())
    {
        channel_io->save(group, name, vchannel.template extract<typename VariantT::template type_of_index<R> >() );
    } else {
        saveVChannel<Derived, VariantT, R-1>(vchannel, channel_io, group, name);
    }
}


template<typename Derived>
template<typename ...Tp>
void VariantChannelIO<Derived>::save(
    std::string groupName,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel)
{
    // TODO
    HighFive::Group g = hdf5util::getGroup(
        m_file_access->m_hdf5_file,
        groupName,
        true
    );

    this->save<Tp...>(g, datasetName, vchannel);
}

template<typename Derived>
template<typename ...Tp>
void VariantChannelIO<Derived>::save(
    HighFive::Group& group,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel)
{
    saveVChannel<Derived, VariantChannel<Tp...>, VariantChannel<Tp...>::num_types-1>(vchannel, m_channel_io, group, datasetName);
}

// R == 0
template<typename Derived, typename VariantT, int R, typename std::enable_if<R == 0, void>::type* = nullptr>
boost::optional<VariantT> loadVChannel(
    HighFive::DataType dtype,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    boost::optional<VariantT> ret;
    if(dtype == HighFive::AtomicType<typename VariantT::template type_of_index<R> >())
    {
        auto channel = channel_io->template load<typename VariantT::template type_of_index<R> >(group, name);
        if(channel) {
            ret = *channel;
        }
        return ret;
    } else {
        return ret;
    }
}

// R != 0
template<typename Derived, typename VariantT, int R, typename std::enable_if<R != 0, void>::type* = nullptr>
boost::optional<VariantT> loadVChannel(
    HighFive::DataType dtype,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(dtype == HighFive::AtomicType<typename VariantT::template type_of_index<R> >())
    {
        boost::optional<VariantT> ret;
        auto loaded_channel = channel_io->template load<typename VariantT::template type_of_index<R> >(group, name);
        if(loaded_channel)
        {
            ret = *loaded_channel;
        }
        return ret;
    } else {
        return loadVChannel<Derived, VariantT, R-1>(dtype, channel_io, group, name);
    }
}

template<typename Derived>
template<typename ...Tp>
VariantChannelOptional<Tp...> VariantChannelIO<Derived>::loadDynamic(
    HighFive::DataType dtype,
    HighFive::Group& group,
    std::string name)
{
    return loadVChannel<Derived, VariantChannel<Tp...>, VariantChannel<Tp...>::num_types-1>(
        dtype, m_channel_io, group, name);
}

template<typename Derived>
template<typename ...Tp>
VariantChannelOptional<Tp...> VariantChannelIO<Derived>::load(
    std::string groupName,
    std::string datasetName)
{
    VariantChannelOptional<Tp...> ret;

    if(hdf5util::exist(m_file_access->m_hdf5_file, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName, false);
        ret = this->load<Tp...>(g, datasetName);
    } else {
        std::cout << "[VariantChannelIO] WARNING: Group " << groupName << " not found." << std::endl;
    }

    return ret;
}

template<typename Derived>
template<typename ...Tp>
VariantChannelOptional<Tp...> VariantChannelIO<Derived>::load(
    HighFive::Group& group,
    std::string datasetName)
{
    VariantChannelOptional<Tp...> ret;

    std::unique_ptr<HighFive::DataSet> dataset;

    try {
        dataset = std::make_unique<HighFive::DataSet>(
            group.getDataSet(datasetName)
        );
    } catch(HighFive::DataSetException& ex) {
        std::cout << "[VariantChannelIO] WARNING: Dataset " << datasetName << " not found." << std::endl;
    }

    if(dataset)
    {
        // name is dataset
        ret = loadDynamic<Tp...>(dataset->getDataType(), group, datasetName);
    }

    return ret;
}


} // hdf5features

} // namespace lvr2 