
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
template<typename Derived, typename VariantChannelT, int R, typename std::enable_if<R == 0, void>::type* = nullptr>
boost::optional<VariantChannelT> loadVChannel(
    HighFive::DataType dtype,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    boost::optional<VariantChannelT> ret;
    if(dtype == HighFive::AtomicType<typename VariantChannelT::template type_of_index<R> >())
    {
        auto channel = channel_io->template load<typename VariantChannelT::template type_of_index<R> >(group, name);
        if(channel) {
            ret = *channel;
        }
        return ret;
    } else {
        return ret;
    }
}

// R != 0
template<typename Derived, typename VariantChannelT, int R, typename std::enable_if<R != 0, void>::type* = nullptr>
boost::optional<VariantChannelT> loadVChannel(
    HighFive::DataType dtype,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(dtype == HighFive::AtomicType<typename VariantChannelT::template type_of_index<R> >())
    {
        boost::optional<VariantChannelT> ret;
        auto loaded_channel = channel_io->template load<typename VariantChannelT::template type_of_index<R> >(group, name);
        if(loaded_channel)
        {
            ret = *loaded_channel;
        }
        return ret;
    } else {
        return loadVChannel<Derived, VariantChannelT, R-1>(dtype, channel_io, group, name);
    }
}

template<typename Derived>
template<typename VariantChannelT>
boost::optional<VariantChannelT> VariantChannelIO<Derived>::loadDynamic(
    HighFive::DataType dtype,
    HighFive::Group& group,
    std::string name)
{
    return loadVChannel<Derived, VariantChannelT, VariantChannelT::num_types-1>(
        dtype, m_channel_io, group, name);
}

template<typename Derived>
template<typename VariantChannelT>
boost::optional<VariantChannelT> VariantChannelIO<Derived>::load(
    std::string groupName,
    std::string datasetName)
{
    boost::optional<VariantChannelT> ret;

    if(hdf5util::exist(m_file_access->m_hdf5_file, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName, false);
        ret = this->load<VariantChannelT>(g, datasetName);
    } else {
        std::cout << "[VariantChannelIO] WARNING: Group " << groupName << " not found." << std::endl;
    }

    return ret;
}

template<typename Derived>
template<typename VariantChannelT>
boost::optional<VariantChannelT> VariantChannelIO<Derived>::load(
    HighFive::Group& group,
    std::string datasetName)
{
    boost::optional<VariantChannelT> ret;

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
        ret = loadDynamic<VariantChannelT>(dataset->getDataType(), group, datasetName);
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


} // hdf5features

} // namespace lvr2 