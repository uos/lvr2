
namespace lvr2 {

template<typename... T>
template<typename U>
void VariantChannelMap<T...>::add(const std::string& name, Channel<U> channel)
{
    this->insert({name, channel});
}

template<typename... T>
template<typename U>
Channel<U>& VariantChannelMap<T...>::get(const std::string& name)
{
    return boost::get<Channel<U> >(this->at(name));
}

template<typename... T>
template<typename U>
const Channel<U>& VariantChannelMap<T...>::get(const std::string& name) const
{
    return boost::get<Channel<U> >(this->at(name));
}

template<typename... T>
int VariantChannelMap<T...>::type(const std::string& name) const
{
    return this->at(name).which();
}

template<typename... T>
template<typename U>
bool VariantChannelMap<T...>::is_type(const std::string& name) const
{
    return this->type(name) == index_of_type<U>::value;
}

template<typename... T>
template<typename U>
std::vector<std::string> VariantChannelMap<T...>::keys()
{
    std::vector<std::string> ret;

    for(auto it = this->begin(); it != this->end(); ++it)
    {
        if(it->second.is_type<U>())
        {
            ret.push_back(it.first);
        }
    }

    return ret;
}

template<typename... T>
template<typename U>
size_t VariantChannelMap<T...>::numChannels()
{
    size_t ret = 0;

    for(auto it = this->begin(); it != this->end(); ++it)
    {
        if(it->second.is_type<U>())
        {
            ret ++;
        }
    }

    return ret;
}


} // namespace lvr2