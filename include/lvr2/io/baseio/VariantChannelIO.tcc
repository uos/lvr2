
namespace lvr2
{

namespace baseio
{

    // Root
    template <typename Derived, typename VChannelT, size_t I,
                typename std::enable_if<I == 0, void>::type * = nullptr>
    void saveDynamic(
        const std::string &group,
        const std::string &name,
        const VChannelT &channel,
        const ChannelIO<Derived> *io)
    {
        if (I == channel.type())
        {
            using StoreType = typename VChannelT::template type_of_index<I>;
            io->template save<StoreType>(group, name,
                                            channel.template extract<StoreType>());
        }
        else
        {
            std::stringstream ss;
            ss << "[VariantChannelIO - saveDynamic] Error: " << channel.typeName() << " not supported.";

            lvr2::logout::get() << lvr2::error << ss.str() << lvr2::endl;
            throw std::runtime_error(ss.str());
        }
    }

    // Recursion
    template <typename Derived, typename VChannelT, size_t I,
                typename std::enable_if<I != 0, void>::type * = nullptr>
    void saveDynamic(
        const std::string &group,
        const std::string &name,
        const VChannelT &channel,
        const ChannelIO<Derived> *io)
    {
        if (I == channel.type())
        {
            using StoreType = typename VChannelT::template type_of_index<I>;
            io->template save<StoreType>(group, name,
                                            channel.template extract<StoreType>());
        }
        else
        {
            saveDynamic<Derived, VChannelT, I - 1>(group, name, channel, io);
        }
    }

    template <typename Derived, typename VChannelT>
    void saveDynamic(
        const std::string &group,
        const std::string &name,
        const VChannelT &channel,
        const ChannelIO<Derived> *io)
    {
        saveDynamic<Derived, VChannelT, VChannelT::num_types - 1>(group, name, channel, io);
    }

    template <typename Derived>
    template <typename... Tp>
    void VariantChannelIO<Derived>::save(
        std::string groupName,
        std::string channelName,
        const VariantChannel<Tp...> &vchannel) const
    {
        // lvr2::logout::get() << "[VariantChannelIO - save] " << groupName << "; " << channelName << "; " << vchannel.typeName() << lvr2::endl;

        using VChannelT = VariantChannel<Tp...>;

        // keep this order! We need Hdf5 to build the dataset first, then writing meta information
        saveDynamic<Derived, VChannelT>(groupName, channelName, vchannel, m_channel_io);

        // creating meta node of variantchannel containing type and size

        // YAML::Node node;
        // try {
        //     node = vchannel;
        // } catch(YAML::TypedBadConversion<int> ex) {
        //     lvr2::logout::get() << ex.what() << lvr2::endl;
        // }
        // m_baseIO->m_kernel->saveMetaYAML(groupName, channelName, node);
    }

    // anker
    template <typename Derived, typename VariantChannelT, size_t I,
                typename std::enable_if<I == 0, void>::type * = nullptr>
    bool _dynamicLoad(
        std::string group, std::string name,
        std::string dyn_type,
        VariantChannelT &vchannel,
        const ChannelIO<Derived> *io)
    {
        using DataT = typename VariantChannelT::template type_of_index<I>;

        if (dyn_type == Channel<DataT>::typeName())
        {
            using DataT = typename VariantChannelT::template type_of_index<I>;

            ChannelOptional<DataT> copt = io->template load<DataT>(group, name);
            if (copt)
            {
                vchannel = *copt;
            }
            else
            {
                lvr2::logout::get() << lvr2::warning << "[VariantChannelIO] Could not receive Channel from ChannelIO!" << lvr2::endl;
                return false;
            }

            return true;
        }

        // some fallbacks for incorrect namings
        if (dyn_type == "uint16")
        {
            using DataT = uint16_t;

            ChannelOptional<DataT> copt = io->template load<DataT>(group, name);
            if (copt)
            {
                vchannel = *copt;
            }
            else
            {
                lvr2::logout::get() << lvr2::warning << "[VariantChannelIO] Could not receive Channel from ChannelIO!" << lvr2::endl;
                return false;
            }

            return true;
        }

        if (dyn_type == "uint8")
        {
            // lvr2::logout::get() << "[VariantChannelIO] WARNING: Depricated type name 'uint8' found at point field: " << name << lvr2::endl;
            using DataT = uint8_t;

            ChannelOptional<DataT> copt = io->template load<DataT>(group, name);
            if (copt)
            {
                vchannel = *copt;
            }
            else
            {
                lvr2::logout::get() << lvr2::warning << "[VariantChannelIO] Could not receive Channel from ChannelIO!" << lvr2::endl;
                return false;
            }

            return true;
        }

        lvr2::logout::get() << lvr2::warning << "[VariantChannelIO] data type '" << dyn_type << "' not implemented in PointBuffer." << lvr2::endl;
        return false;
    }

    template <typename Derived, typename VariantChannelT, size_t I,
                typename std::enable_if<I != 0, void>::type * = nullptr>
    bool _dynamicLoad(
        std::string group, std::string name,
        std::string dyn_type,
        VariantChannelT &vchannel,
        const ChannelIO<Derived> *io)
    {
        using DataT = typename VariantChannelT::template type_of_index<I>;

        if (dyn_type == Channel<DataT>::typeName())
        {
            ChannelOptional<DataT> copt = io->template load<DataT>(group, name);
            if (copt)
            {
                vchannel = *copt;
            }
            else
            {
                lvr2::logout::get() << lvr2::warning << "[VariantChannelIO] Could not receive Channel from ChannelIO!" << lvr2::endl;
                return false;
            }

            return true;
        }
        else
        {
            return _dynamicLoad<Derived, VariantChannelT, I - 1>(group, name, dyn_type, vchannel, io);
        }
    }

    template <typename Derived, typename VariantChannelT>
    bool dynamicLoad(
        std::string group, std::string name,
        std::string dyn_type,
        VariantChannelT &vchannel,
        const ChannelIO<Derived> *io)
    {
        return _dynamicLoad<Derived, VariantChannelT, VariantChannelT::num_types - 1>(group, name, dyn_type, vchannel, io);
    }

    template <typename Derived>
    template <typename VariantChannelT>
    boost::optional<VariantChannelT> VariantChannelIO<Derived>::load(
        std::string groupName,
        std::string datasetName) const
    {
        // lvr2::logout::get() << "[VariantChannelIO - load] " << groupName << ", " << datasetName << lvr2::endl;

        boost::optional<VariantChannelT> ret;

        YAML::Node node;
        m_baseIO->m_kernel->loadMetaYAML(groupName, datasetName, node);

        lvr2::MultiChannel mc;
        if (!YAML::convert<lvr2::MultiChannel>::decode(node, mc))
        {
            // fail
            lvr2::logout::get() << lvr2::warning << "[VariantChannelIO - load] Tried to load Meta information that does not suit to VariantChannel types" << lvr2::endl;
            return ret;
        }

        // std::string data_type = mc.typeName();
        std::string data_type = node["data_type"].as<std::string>();

        // load channel with correct datatype
        VariantChannelT vchannel;
        if (dynamicLoad<Derived, VariantChannelT>(
                groupName, datasetName,
                data_type, vchannel, m_channel_io))
        {
            ret = vchannel;
        }
        else
        {
            lvr2::logout::get() << lvr2::error << "[VariantChannelIO] Error occured while loading group '" << groupName << "', dataset '" << datasetName << "'" << lvr2::endl;
        }

        return ret;
    }

    template <typename Derived>
    template <typename VariantChannelT>
    boost::optional<VariantChannelT> VariantChannelIO<Derived>::loadVariantChannel(
        std::string groupName,
        std::string datasetName) const
    {
        return load<VariantChannelT>(groupName, datasetName);
    }

} // namespace scanio

} // namespace lvr2