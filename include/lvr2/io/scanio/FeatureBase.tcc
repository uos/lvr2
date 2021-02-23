namespace lvr2 {

template<template<typename> typename ...Features>
template<template<typename> typename F>
bool FeatureBase<Features...>::has() {
    return FeatureBase<Features...>::template has_feature<F>::value;
}

template<template<typename> typename ...Features>
template<template<typename> typename F>
F<FeatureBase<Features...> >* FeatureBase<Features...>::scast() {
    return static_cast< F<FeatureBase<Features...> >* >(this);
}

template<template<typename> typename ...Features>
template<template<typename> typename F>
F<FeatureBase<Features...> >* FeatureBase<Features...>::dcast() {
    return dynamic_cast< F<FeatureBase<Features...> >* >(this);
}

} // namespace lvr2