#pragma once
#ifndef LVR2_IO_GFeatureBase_HPP
#define LVR2_IO_GFeatureBase_HPP

#include <memory>
#include <tuple>
#include <type_traits>

#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2 {

/**
 * @class FeatureConstruct
 * @brief Helper class how to construct a IO feature with its dependencies
 * 
 */
template<template<typename> typename Feature, typename Derived>
struct FeatureConstruct;


/**
 * @class FeatureBase
 * @brief Manager Class for all FeatureBase components located in hdf5 directory
 * 
 * 
 */

template<template<typename> typename ...Features>
class FeatureBase : public Features<FeatureBase<Features...> >...
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
    using features = std::tuple<Features<FeatureBase<Features...> >...>;

    template<template<typename> typename F> 
    struct has_feature {
        static constexpr bool value = has_type<F<FeatureBase>, features>::type::value;
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
            FeatureBase<Features...>::has_feature<F>::value,
            FeatureBase<Features...>,
            FeatureBase<Features...,F>
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
        using type = typename FeatureConstruct<F, FeatureBase<Features...> >::type;
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

    //using Features<FeatureBase<Features...> >::save...;

    FeatureBase(
        const FileKernelPtr inKernel, 
        const ScanProjectSchemaPtr inDesc) : m_kernel(inKernel), m_description(inDesc)
    {

    }

    virtual ~FeatureBase() {}

    template<template<typename> typename F>
    bool has();

    template<template<typename> typename F>
    F<FeatureBase>* scast();

    template<template<typename> typename F>
    F<FeatureBase>* dcast();

    const FileKernelPtr             m_kernel;
    const ScanProjectSchemaPtr      m_description;

};

template<template<typename> typename Feature, typename Derived = FeatureBase<> >
struct FeatureConstruct {
    using type = typename Derived::template add_features<Feature>::type;
};

template<template<typename> typename Feature>
using FeatureBuild = typename FeatureConstruct<Feature>::type;


} // namespace lvr2

#include "FeatureBase.tcc"

#endif // LVR2_IO_GFeatureBase_HPP
