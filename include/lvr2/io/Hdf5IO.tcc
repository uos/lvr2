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

} // namespace lvr2