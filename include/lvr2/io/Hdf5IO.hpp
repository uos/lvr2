#pragma once

#include <memory>

#include "hdf5/Hdf5Util.hpp"

#include <H5Tpublic.h>
#include <hdf5_hl.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

namespace lvr2 {

template<template<typename> typename ...Features>
class Hdf5IO : public Features<Hdf5IO<Features...> >...
{
public:
    static constexpr std::size_t N = sizeof...(Features);

    #if __cplusplus <= 201500L
        // feature of c++17
        #pragma message("using Tp::save... needs c++17 at least or a newer compiler")
    #endif

    using Features<Hdf5IO<Features...> >::save...;


    Hdf5IO()
    :m_compress(true),
    m_chunkSize(1e7),
    m_usePreviews(true)
    {

    }

    void open(std::string filename);

    bool                    m_compress;
    size_t                  m_chunkSize;
    bool                    m_usePreviews;
    unsigned int            m_previewReductionFactor;
    std::string m_filename;
    std::shared_ptr<HighFive::File>         m_hdf5_file;

};

} // namespace lvr2

#include "Hdf5IO.tcc"