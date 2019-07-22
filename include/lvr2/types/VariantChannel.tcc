
namespace lvr2 {

template<typename... T>
size_t VariantChannel<T...>::numElements() const
{
    return boost::apply_visitor(NumElementsVisitor(), *this);
}

template<typename... T>
size_t VariantChannel<T...>::width() const
{
    return boost::apply_visitor(WidthVisitor(), *this);
}

template<typename... T>
template<typename U>
boost::shared_array<U> VariantChannel<T...>::dataPtr() const
{
    return boost::apply_visitor(DataPtrVisitor<U>(), *this);
}

template<typename... T>
int VariantChannel<T...>::type() const
{
    return this->which();
}

template<typename... T>
template<typename U>
bool VariantChannel<T...>::is_type() const {
    return this->which() == index_of_type<U>::value;
}

}