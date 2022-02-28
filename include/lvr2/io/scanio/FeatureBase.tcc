namespace lvr2 {

template<typename SchemaPtrT, template<typename> typename ...Features>
template<template<typename> typename F>
bool FeatureBase<SchemaPtrT, Features...>::has() {
    return FeatureBase<SchemaPtrT, Features...>::template has_feature<F>::value;
}

template<typename SchemaPtrT, template<typename> typename ...Features>
template<template<typename> typename F>
F<FeatureBase<SchemaPtrT, Features...> >* FeatureBase<SchemaPtrT, Features...>::scast() {
    return static_cast< F<FeatureBase<SchemaPtrT, Features...> >* >(this);
}

template<typename SchemaPtrT, template<typename> typename ...Features>
template<template<typename> typename F>
F<FeatureBase<SchemaPtrT, Features...> >* FeatureBase<SchemaPtrT, Features...>::dcast() {
    return dynamic_cast< F<FeatureBase<SchemaPtrT, Features...> >* >(this);
}

} // namespace lvr2