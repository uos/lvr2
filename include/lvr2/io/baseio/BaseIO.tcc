namespace lvr2 
{

namespace baseio
{

template<typename SchemaPtrT, template<typename> typename ...Features>
template<template<typename> typename F>
bool BaseIO<SchemaPtrT, Features...>::has() {
    return BaseIO<SchemaPtrT, Features...>::template has_feature<F>::value;
}

template<typename SchemaPtrT, template<typename> typename ...Features>
template<template<typename> typename F>
F<BaseIO<SchemaPtrT, Features...> >* BaseIO<SchemaPtrT, Features...>::scast() {
    return static_cast< F<BaseIO<SchemaPtrT, Features...> >* >(this);
}

template<typename SchemaPtrT, template<typename> typename ...Features>
template<template<typename> typename F>
F<BaseIO<SchemaPtrT, Features...> >* BaseIO<SchemaPtrT, Features...>::dcast() {
    return dynamic_cast< F<BaseIO<SchemaPtrT, Features...> >* >(this);
}

} // namespace baseio

} // namespace lvr2