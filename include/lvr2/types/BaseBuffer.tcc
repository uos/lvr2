namespace lvr2 {

template<typename T>
size_t BaseBuffer::channelWidth(const std::string& name)
{
    auto it = this->find(name);
    if(it != this->end() && it->second.is_type<T>())
    {
        return it->second.width();
    }
    return 0;
}

template<typename T>
bool BaseBuffer::hasChannel(const std::string& name)
{
    auto it = this->find(name);
    if(it != this->end() && it->second.is_type<T>())
    {
        return true;
    }
    return false;
}

template<typename T>
void BaseBuffer::addChannel(typename Channel<T>::Ptr data, const std::string& name)
{
    this->insert({name, *data});
}

template<typename T>
void BaseBuffer::addChannel(
    boost::shared_array<T> array,
    std::string name,
    size_t n,
    size_t width)
{
    this->insert({name, Channel<T>(n, width, array)});
}

template<typename T>
void BaseBuffer::addEmptyChannel(
    const std::string& name,
    size_t n,
    size_t width)
{
    Channel<T> channel(n, width);
    // init zeros
    std::fill(channel.dataPtr().get(), channel.dataPtr().get() + n * width, 0);
    this->insert({name, channel});
}

template<typename T>
bool BaseBuffer::removeChannel(const std::string& name)
{
    auto it = this->find(name);
    if(it != this->end() && it->second.is_type<T>())
    {
        erase(it);
        return true;
    }
    return false;
}

template<typename T>
typename Channel<T>::Optional BaseBuffer::getChannel(const std::string& name)
{
    typename Channel<T>::Optional ret;

    auto it = this->find(name);
    if(it != this->end() && it->second.is_type<T>())
    {
        ret = boost::get<Channel<T> >(it->second);
    }

    return ret;
}

template<typename T>
ElementProxy<T> BaseBuffer::getHandle(unsigned int idx, const std::string& name)
{
    // std::cout << "WARNING: runtime critical access [BaseBuffer::getHandle]" << std::endl;
    auto it = this->find(name);
    if(it != this->end() && it->second.is_type<T>())
    {
        return boost::get<Channel<T> >(it->second)[idx];
    }
    return ElementProxy<T>();
}

template<typename T>
boost::shared_array<T> BaseBuffer::getArray(
    const std::string& name, size_t& n, size_t& w)
{
    auto it = this->find(name);
    if(it != this->end() && it->second.is_type<T>())
    {
        Channel<T> channel = boost::get<Channel<T> >(it->second);
        n = channel.numElements();
        w = channel.width();
        return channel.dataPtr();
    } 
    n = 0;
    w = 0;
    return boost::shared_array<T>();
}


template<typename T>
void BaseBuffer::addAtomic(T data, const std::string& name)
{
    Channel<T> channel(1, 1);
    channel[0][0] = data;
    this->insert({name, channel});
}

template<typename T>
boost::optional<T> BaseBuffer::getAtomic(const std::string& name)
{
    boost::optional<T> ret;
    typename Channel<T>::Optional channel = getChannel<T>(name);
    if(channel)
    {
        ret = (*channel)[0][0];
    }
    return ret;
}



} // namespace lvr2