namespace lvr2 {

template<template<typename> typename ...Features>
void Hdf5IO<Features...>::open(std::string filename) {
    
    m_filename = filename;
    this->m_hdf5_file = hdf5util::open(filename);

    if (!m_hdf5_file->isValid())
    {
        throw std::runtime_error("[Hdf5IO] Hdf5 file not valid!");
        return;
    }
}

template<template<typename> typename ...Features>
template<template<typename> typename F>
bool Hdf5IO<Features...>::has() {
    return Hdf5IO<Features...>::template has_feature<F>::value;
}

template<template<typename> typename ...Features>
template<template<typename> typename F>
F<Hdf5IO<Features...> >* Hdf5IO<Features...>::scast() {
    return static_cast< F<Hdf5IO<Features...> >* >(this);
}

template<template<typename> typename ...Features>
template<template<typename> typename F>
F<Hdf5IO<Features...> >* Hdf5IO<Features...>::dcast() {
    return dynamic_cast< F<Hdf5IO<Features...> >* >(this);
}

} // namespace lvr2