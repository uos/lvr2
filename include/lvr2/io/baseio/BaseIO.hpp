#pragma once
#ifndef LVR2_IO_DESC_BaseIO_HPP
#define LVR2_IO_DESC_BaseIO_HPP

#include <memory>
#include <tuple>
#include <type_traits>

#include "lvr2/io/kernels/FileKernel.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2 
{

namespace baseio
{

/**
 * @class FeatureConstruct
 * @brief Helper class how to construct a IO feature with its dependencies
 * 
 */
template<template<typename> typename Feature, typename Derived>
struct FeatureConstruct;


/**
 * @class BaseIO
 * @brief Manager Class for all BaseIO components located in hdf5 directory
 * 
 * 
 */

template<typename SchemaPtrT, template<typename> typename ...Features>
class BaseIO :  
    public std::enable_shared_from_this<BaseIO<SchemaPtrT, Features...>>, 
    public Features<BaseIO<SchemaPtrT, Features...> >...
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
    using features = std::tuple<Features<BaseIO<SchemaPtrT, Features...> >...>;

    template<template<typename> typename F> 
    struct has_feature {
        static constexpr bool value = has_type<F<BaseIO>, features>::type::value;
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
            BaseIO<SchemaPtrT, Features...>::has_feature<F>::value,
            BaseIO<SchemaPtrT, Features...>,
            BaseIO<SchemaPtrT, Features...,F>
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
        using type = typename FeatureConstruct<F, BaseIO<SchemaPtrT, Features...> >::type;
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


    // #if __cplusplus <= 201500L
    //     // feature of c++17
    //     #pragma message("using Tp::save... needs c++17 at least or a newer compiler")
    // #endif

    //using Features<BaseIO<SchemaPtrT, Features...> >::save...;

    BaseIO(
        const FileKernelPtr inKernel, 
        const SchemaPtrT inDesc,
        const bool load_data = false) 
    : m_kernel(inKernel)
    , m_description(inDesc)
    , m_load_data(load_data)
    {

    }

    virtual ~BaseIO() {}

    template<template<typename> typename F>
    bool has();

    template<template<typename> typename F>
    F<BaseIO>* scast();

    template<template<typename> typename F>
    F<BaseIO>* dcast();

    const FileKernelPtr             m_kernel;
    const SchemaPtrT                m_description;

    const bool                      m_load_data;

};

template<template<typename> typename Feature, typename Derived>
struct FeatureConstruct {
    using type = typename Derived::template add_features<Feature>::type;
};

// SchemaPtrT defaults to ScanProjectSchemaPtr for compatibility
// If your schema is not a subclass of ScanProjectSchema you will need to specify the schema ptr type
// MeshIO example: FeatureBuild<MeshIO, MeshSchemaPtr>
// ScanProjectIO example: FeatureBuild<ScanProjectIO>
template<template<typename> typename Feature, typename SchemaPtrT = ScanProjectSchemaPtr>
using FeatureBuild = typename FeatureConstruct<Feature, BaseIO<SchemaPtrT>>::type;

} // namespace baseio



} // namespace lvr2

#include "BaseIO.tcc"

#endif // LVR2_IO_GBaseIO_HPP
