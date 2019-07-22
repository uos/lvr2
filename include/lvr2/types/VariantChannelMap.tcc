
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
    // if(this->at(name).which() == index_of_type<U>::value)
    // {

    // }
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

// template<typename... U>
// std::ostream& operator<<(std::ostream& os, const VariantChannelMap<U...>& cm)
// {
//     os << "bla";
//     return os;
// }

} // namespace lvr2