#pragma once

#include <memory>
#include <tuple>
#include <type_traits>

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
protected:
    template <typename T, typename Tuple>
    struct has_type;

    template <typename T>
    struct has_type<T, std::tuple<>> : std::false_type {};

    template <typename T, typename U, typename... Ts>
    struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};

    template <typename T, typename... Ts>
    struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};

public:
    static constexpr std::size_t N = sizeof...(Features);
    using features = std::tuple<Features<Hdf5IO<Features...> >...>;

    template<template<typename> typename F> 
    struct has_feature {
        static constexpr bool value = has_type<F<Hdf5IO>, features>::type::value;
    };

    

    template<
        template<typename> typename F,
        template<typename> typename ...Fs
    >
    struct add_features;

    template<
        template<typename> typename F,
        template<typename> typename ...Fs
    >
    struct add_features {
        using type = typename add_features<F>::type::template add_features<Fs...>::type;
    };

    template<
        template<typename> typename F
    >
    struct add_features<F> {
        using type = typename std::conditional<
            Hdf5IO<Features...>::has_feature<F>::value,
            Hdf5IO<Features...>,
            Hdf5IO<Features...,F>
            >::type;
    };

    template<template<typename> typename F>
    struct add_feature {
        using type = typename add_features<F>::type;
    };

    #if __cplusplus <= 201500L
        // feature of c++17
        #pragma message("using Tp::save... needs c++17 at least or a newer compiler")
    #endif

    using Features<Hdf5IO<Features...> >::save...;
    // using Features<Hdf5IO<Features...> >::read...;


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