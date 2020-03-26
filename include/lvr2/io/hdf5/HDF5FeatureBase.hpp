#pragma once
#ifndef LVR2_IO_GHDF5IO_HPP
#define LVR2_IO_GHDF5IO_HPP

#include <memory>
#include <tuple>
#include <type_traits>

#include "lvr2/io/hdf5/Hdf5Util.hpp"

#include <H5Tpublic.h>
#include <hdf5_hl.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>



namespace lvr2 {

/**
 * @class Hdf5Construct
 * @brief Helper class how to construct a IO feature with its dependencies
 * 
 */
template<template<typename> typename Feature, typename Derived>
struct Hdf5Construct;


/**
 * @class Hdf5IO
 * @brief Manager Class for all Hdf5IO components located in hdf5 directory
 * 
 * 
 */

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

    template<
        template<typename> typename F,
        template<typename> typename ...Fs
    >
    struct add_features_with_deps;

    template<
        template<typename> typename F,
        template<typename> typename ...Fs
    >
    struct add_features_with_deps {
        using type = typename add_features_with_deps<F>::type::template add_features_with_deps<Fs...>::type;
    };

    template<template<typename> typename F>
    struct add_features_with_deps<F> {
        using type = typename Hdf5Construct<F, Hdf5IO<Features...> >::type;
    };

    /////////////////////////////////////////////
    /// USE ONLY THESE METHODS IN APPLICATION ///
    /////////////////////////////////////////////


    template<template<typename> typename F>
    static constexpr bool HasFeature = has_feature<F>::value;
    
    template<template<typename> typename ...F>
    using AddFeatures = typename add_features_with_deps<F...>::type;

    template<typename OTHER>
    using Merge = typename OTHER::template add_features<Features...>::type;


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

    virtual ~Hdf5IO() {}

    void open(std::string filename);

    template<template<typename> typename F>
    bool has();

    template<template<typename> typename F>
    F<Hdf5IO>* scast();

    template<template<typename> typename F>
    F<Hdf5IO>* dcast();

    bool                    m_compress;
    size_t                  m_chunkSize;
    bool                    m_usePreviews;
    unsigned int            m_previewReductionFactor;
    std::string m_filename;
    std::shared_ptr<HighFive::File>         m_hdf5_file;

};

template<template<typename> typename Feature, typename Derived = Hdf5IO<> >
struct Hdf5Construct {
    using type = typename Derived::template add_features<Feature>::type;
};

template<template<typename> typename Feature>
using Hdf5Build = typename Hdf5Construct<Feature>::type;


} // namespace lvr2

#include "HDF5FeatureBase.tcc"

#endif // LVR2_IO_GHDF5IO_HPP