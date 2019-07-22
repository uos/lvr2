#pragma once

#ifndef LVR2_TYPES_CHANNELMANAGER
#define LVR2_TYPES_CHANNELMANAGER

#include <algorithm>
#include "MultiChannelMap.hpp"
#include "lvr2/io/DataStruct.hpp"

namespace lvr2 {

using intOptional = boost::optional<int>;
using floatOptional = boost::optional<float>;
using ucharOptional = boost::optional<unsigned char>;

using FloatProxy = ElementProxy<float>;
using UCharProxy = ElementProxy<unsigned char>;
using IndexProxy = ElementProxy<unsigned int>;

/**
 * @brief ChannelManager class
 *      Store and access AttributeChannels. It expands the MultiChannelMap with
 *      downwoards compitibility functions of the old ChannelManager.
 * 
 */
class ChannelManager : public MultiChannelMap {
using base = MultiChannelMap;
public:
    using base::base;

    //////////////////////////////////
    //// Width Channel functions ////
    //////////////////////////////////

    /**
     * @brief Gets a channels width.
     * @param[in] name Key of the channel.
     * @tparam T type of channel to search for.
     * @return 0 if not found, otherwise the channels width.
     */
    template<typename T>
    size_t channelWidth(const std::string& name);

    /**
     * @brief Gets an uchar channels width.
     * @param[in] name Key of the channel.
     * @return 0 if not found, otherwise the channels width.
     */
    inline size_t ucharChannelWidth(const std::string& name)
    {
        return channelWidth<unsigned char>(name);
    }

    /**
     * @brief Gets a float channels width.
     * @param[in] name Key of the channel.
     * @return 0 if not found, otherwise the channels width.
     */
    inline size_t floatChannelWidth(const std::string& name)
    {
        return channelWidth<float>(name);
    }

    /**
     * @brief Gets an index channels width.
     * @param[in] name Key of the channel.
     * @return 0 if not found, otherwise the channels width.
     */
    inline size_t indexChannelWidth(const std::string& name)
    {
        return channelWidth<unsigned int>(name);
    }

    //////////////////////////////////
    //// Has Channel functions ////
    //////////////////////////////////

    /**
     * @brief Checks if a channel is available.
     * @param[in] name Key of the channel.
     * @tparam T Type of the channel.
     */
    template<typename T>
    bool hasChannel(const std::string& name);

    /**
     * @brief Checks if an uchar channel is available.
     * @param[in] name Key of the channel.
     */
    inline bool hasUCharChannel(const std::string& name)
    {
        return hasChannel<unsigned char>(name);
    }

    /**
     * @brief Checks if a float channel is available.
     * @param[in] name Key of the channel.
     */
    inline bool hasFloatChannel(const std::string& name)
    {
        return hasChannel<float>(name);
    }

    /**
     * @brief Checks if an index channel is available.
     * @param[in] name Key of the channel.
     */
    inline bool hasIndexChannel(const std::string& name)
    {
        return hasChannel<unsigned int>(name);
    }

    //////////////////////////////////
    //// Add Channel functions ////
    //////////////////////////////////

    /**
     * @brief Adds a channel pointer to the map.
     * @param[in] data The channel pointer to add. 
     * @param[in] name The key of the channel.
     * @tparam T Type of the channel.
     */
    template<typename T>
    void addChannel(typename Channel<T>::Ptr data, const std::string& name);

    /**
     * @brief Adds a float channel pointer to the map.
     * @param[in] data The channel pointer to add. 
     * @param[in] name Key of the channel.
     */
    inline void addFloatChannel(FloatChannelPtr data, const std::string& name)
    {
        addChannel<float>(data, name);
    }

    /**
     * @brief Adds an uchar channel pointer to the map.
     * @param[in] data The channel pointer to add. 
     * @param[in] name Key of the channel.
     */
    inline void addUCharChannel(UCharChannelPtr data, const std::string& name)
    {
        addChannel<unsigned char>(data, name);
    }

    /**
     * @brief Adds an index channel pointer to the map.
     * @param[in] data The channel pointer to add. 
     * @param[in] name Key of the channel.
     */
    inline void addIndexChannel(IndexChannelPtr data, const std::string& name)
    {
        addChannel<unsigned int>(data, name);
    }

    /**
     * @brief Constructs a channel from an boost::shared_array and saves it to the map.
     * @param[in] array The shared array of the data. 
     * @param[in] name Key of the channel.
     * @tparam T Type of the channel.
     */
    template<typename T>
    void addChannel(boost::shared_array<T> array, std::string name, size_t n, size_t width);

    /**
     * @brief Constructs an index channel from an boost::shared_array and saves it to the map.
     * @param[in] array The shared array of the data. 
     * @param[in] name Key of the channel.
     */
    inline void addIndexChannel( indexArray array, std::string name, size_t n, size_t width)
    {
        addChannel<unsigned int>(array, name, n, width);
    }

    /**
     * @brief Constructs a float channel from an boost::shared_array and saves it to the map.
     * @param[in] array The shared array of the data. 
     * @param[in] name Key of the channel.
     */
    inline void addFloatChannel( floatArr array, std::string name, size_t n, size_t width)
    {
        addChannel<float>(array, name, n, width);
    }

    /**
     * @brief Constructs an uchar channel from an boost::shared_array
     *          and saves it to the map.
     * 
     * @param[in] array The shared array of the data. 
     * @param[in] name Key of the channel.
     */
    inline void addUCharChannel(ucharArr array, std::string name, size_t n, size_t width)
    {
        addChannel<unsigned char>(array, name, n, width);
    }

    /**
     * @brief Adds an empty channel to the map.
     * 
     * @param[in] name Key of the channel.
     * @param[in] n Number of elements.
     * @param[in] width Width of one element.
     * @tparam T Type of the channel.
     */
    template<typename T>
    void addEmptyChannel( const std::string& name, size_t n, size_t width);

    /**
     * @brief Adds an empty float channel to the map.
     * @param[in] name Key of the channel.
     * @param[in] n Number of elements.
     * @param[in] width Width of one element.
     */
    inline void addEmptyFloatChannel( const std::string& name, size_t n, size_t width)
    {
        addEmptyChannel<float>(name, n, width);
    }

    /**
     * @brief Adds an empty uchar channel to the map.
     * @param[in] name Key of the channel.
     * @param[in] n Number of elements.
     * @param[in] width Width of one element.
     */
    inline void addEmptyUCharChannel( const std::string& name, size_t n, size_t width)
    {
        addEmptyChannel<unsigned char>(name, n, width);
    }

    /**
     * @brief Adds an empty index channel to the map.
     * @param[in] name Key of the channel.
     * @param[in] n Number of elements.
     * @param[in] width Width of one element.
     */
    inline void addEmptyIndexChannel( const std::string& name, size_t n, size_t width)
    {
        addEmptyChannel<unsigned int>(name, n, width);
    }

    //////////////////////////////////
    //// Remove Channel functions ////
    //////////////////////////////////
    
    /**
     * @brief Removes a channel with a specific type.
     * @detail If the type is not required use: erase.
     * 
     * @tparam T Type of the channel.
     * @param[in] name Key of the channel.
     * @return true If the channel was removed.
     * @return false If no channel was removed.
     */
    template<typename T>
    bool removeChannel(const std::string& name);

    /**
     * @brief Removes an index channel.
     * @detail If the type is not required use: erase.
     * 
     * @param[in] name Key of the channel.
     * @return true If the channel was removed.
     * @return false If no channel was removed.
     */
    bool removeIndexChannel(const std::string& name)
    {
        return removeChannel<unsigned int>(name);
    }

    /**
     * @brief Removes a float channel.
     * @detail If the type is not required use: erase.
     * 
     * @param[in] name Key of the channel.
     * @return true If the channel was removed.
     * @return false If no channel was removed.
     */
    bool removeFloatChannel(const std::string& name)
    {
        return removeChannel<float>(name);
    }

    /**
     * @brief Removes an uchar channel.
     * @detail If the type is not required use: erase.
     * 
     * @param[in] name Key of the channel.
     * @return true If the channel was removed.
     * @return false If no channel was removed.
     */
    bool removeUCharChannel(const std::string& name)
    {
        return removeChannel<unsigned char>(name);
    }

    ///////////////////////////////
    //// Get Channel functions ////
    ///////////////////////////////

    /**
     * @brief Gets a channel and returns it as optional. 
     * 
     * @param[in] name Key of the channel.
     * @tparam T Type of the channel.
     * @return An OptionalChannel which is filled if the channel was found.
     */
    template<typename T>
    typename Channel<T>::Optional getChannel(const std::string& name);

    /**
     * @brief Gets a float channel and returns it as optional. 
     * 
     * @param[in] name Key of the channel.
     * @return An OptionalChannel which is filled if the channel was found.
     */
    inline Channel<float>::Optional getFloatChannel(const std::string& name)
    {
        return getChannel<float>(name);
    }

    /**
     * @brief Gets an uchar channel and returns it as optional. 
     * 
     * @param[in] name Key of the channel.
     * @return An OptionalChannel which is filled if the channel was found.
     */
    inline Channel<unsigned char>::Optional getUCharChannel(const std::string& name)
    {
        return getChannel<unsigned char>(name);
    }

    /**
     * @brief Gets an index channel and returns it as optional. 
     * 
     * @param[in] name Key of the channel.
     * @return An OptionalChannel which is filled if the channel was found.
     */
    inline Channel<unsigned int>::Optional getIndexChannel(const std::string& name)
    {
        return getChannel<unsigned int>(name);
    }

    /**
     * @brief Gets a float channel and returns it as optional.
     * 
     * @param[in] name Key of the channel.
     * @param[out] channelOptional The float channel optional.
     */
    inline void getChannel(const std::string& name, FloatChannelOptional& channelOptional )
    { 
        channelOptional = getFloatChannel(name);
    }

    /**
     * @brief Gets an index channel and returns it as optional.
     * 
     * @param[in] name Key of the channel.
     * @param[out] channelOptional The index channel optional.
     */
    inline void getChannel(const std::string& name, IndexChannelOptional& channelOptional)
    {
        channelOptional = getIndexChannel(name);
    }

    /**
     * @brief Gets an uchar channel and returns it as optional.
     * 
     * @param[in] name Key of the channel.
     * @param[out] channelOptional The uchar channel optional.
     */
    inline void getChannel(const std::string& name, UCharChannelOptional& channelOptional)
    {
        channelOptional = getUCharChannel(name);
    }

    ///////////////////////////////
    //// Get Handle functions ////
    ///////////////////////////////

    /**
     * @brief Get a Handle object (ElementProxy) of a specific typed channel.
     * 
     * @tparam T The type of the channel.
     * @param[in] idx The index of the element to access.
     * @param[in] name Key of the channel.
     * @return ElementProxy<T> The handle.
     */
    template<typename T>
    ElementProxy<T> getHandle(int idx, const std::string& name);
    
    /**
     * @brief Get a Handle object (ElementProxy) of a float channel.
     * 
     * @param[in] idx The index of the element to access.
     * @param[in] name Key of the channel.
     * @return FloatProxy The handle.
     */
    inline FloatProxy getFloatHandle(int idx, const std::string& name)
    {
        return getHandle<float>(idx, name);
    }

    /**
     * @brief Get a Handle object (ElementProxy) of an uchar channel.
     * 
     * @param[in] idx The index of the element to access.
     * @param[in] name Key of the channel.
     * @return UCharProxy The handle.
     */
    inline UCharProxy getUCharHandle(int idx, const std::string& name)
    {
        return getHandle<unsigned char>(idx, name);
    }
    
    /**
     * @brief Get a Handle object (ElementProxy) of an index channel.
     * 
     * @param[in] idx The index of the element to access.
     * @param[in] name Key of the channel.
     * @return IndexProxy The handle.
     */
    inline IndexProxy getIndexHandle(int idx, const std::string& name)
    {
        return getHandle<unsigned int>(idx, name);
    }

    ///////////////////////////////
    //// Get Array functions ////
    ///////////////////////////////

    /**
     * @brief Gets a channel as array.
     * 
     * @tparam T Type of the channel.
     * @param[out] n Number of elements stored in the channel.
     * @param[out] w Width of an element.
     * @param[in] name Key of the channel.
     * @return boost::shared_array<T> The data pointer. Empty if the channel was not found.
     */
    template<typename T>
    boost::shared_array<T> getArray(size_t& n, unsigned& w, const std::string& name);

    /**
     * @brief Gets a float channel as array.
     * 
     * @param[out] n Number of elements stored in the channel.
     * @param[out] w Width of an element.
     * @param[in] name Key of the channel.
     * @return floatArr The data pointer. Empty if the channel was not found.
     */
    inline floatArr getFloatArray(size_t& n, unsigned& w, const std::string& name)
    {
        return getArray<float>(n, w, name);
    }

    /**
     * @brief Gets an uchar channel as array.
     * 
     * @param[out] n Number of elements stored in the channel.
     * @param[out] w Width of an element.
     * @param[in] name Key of the channel.
     * @return ucharArr The data pointer. Empty if the channel was not found.
     */
    inline ucharArr getUCharArray(size_t& n, unsigned& w, const std::string& name)
    {
        return getArray<unsigned char>(n, w, name);
    }

    /**
     * @brief Gets an index channel as array.
     * 
     * @param[out] n Number of elements stored in the channel.
     * @param[out] w Width of an element.
     * @param[in] name Key of the channel.
     * @return indexArray The data pointer. Empty if the channel was not found.
     */
    inline indexArray getIndexArray(size_t& n, unsigned& w, const std::string& name)
    {
        return getArray<unsigned int>(n, w, name);
    }

    ///////////////////////////////
    //// Add Atomic functions ////
    ///////////////////////////////

    /**
     * @brief Adds an atomic value. Exists only for compatibility reasons.
     *          Dont use atomics, they are implemented as Channels.
     *          -> memory overhead.
     * 
     * @tparam T Type of the atomic value.
     * @param[in] data The atomic data to add to the channel manager.
     * @param[in] name The key of the atomic value (don't use keys that are already used for channels).
     */
    template<typename T>
    void addAtomic(T data, const std::string& name);

    /**
     * @brief Adds an atomic float value. Exists only for compatibility reasons.
     *          Dont use atomics, they are implemented as Channels.
     *          -> memory overhead.
     * @param[in] data The atomic data to add to the channel manager.
     * @param[in] name The key of the atomic value (don't use keys that are already used for channels).
     */
    inline void addFloatAtomic(float data, const std::string& name)
    {
        addAtomic(data, name);
    }

    /**
     * @brief Adds an atomic uchar value. Exists only for compatibility reasons.
     *          Dont use atomics, they are implemented as Channels.
     *          -> memory overhead.
     *          Kept because of api stability.
     * @param[in] data The atomic data to add to the channel manager.
     * @param[in] name The key of the atomic value (don't use keys that are already used for channels).
     */
    inline void addUCharAtomic(unsigned char data, const std::string& name)
    {
        addAtomic(data, name);
    }

    /**
     * @brief Adds an atomic int value. Exists only for compatibility reasons.
     *          Dont use atomics, they are implemented as Channels.
     *          -> memory overhead.
     *          Kept because of api stability.
     * @param[in] data The atomic data to add to the channel manager.
     * @param[in] name The key of the atomic value (don't use keys that are already used for channels).
     */
    inline void addIntAtomic(int data, const std::string& name)
    {
        addAtomic(data, name);
    }
    ///////////////////////////////
    //// Get Atomic functions ////
    ///////////////////////////////

    /**
     * @brief Gets an atomic value.
     * 
     * @tparam T The atomic values type.
     * @param[in] name Key of the atomic value. 
     * 
     * @return The atomic value as optional. The optional is set if the atomic value was found.
     * 
     */
    template<typename T>
    boost::optional<T> getAtomic(const std::string& name);

    /**
     * @brief Gets an atomic float value.
     * 
     * @param[in] name Key of the atomic value. 
     * 
     * @return The atomic value as optional. The optional is set if the atomic value was found.
     * 
     */
    inline floatOptional getFloatAtomic(const std::string& name)
    {
        return getAtomic<float>(name);
    }

    /**
     * @brief Gets an atomic uchar value.
     * 
     * @param[in] name Key of the atomic value. 
     * 
     * @return The atomic value as optional. The optional is set if the atomic value was found.
     * 
     */
    inline ucharOptional getUCharAtomic(const std::string& name)
    {
        return getAtomic<unsigned char>(name);
    }

    /**
     * @brief Gets an atomic int value.
     * 
     * @param[in] name Key of the atomic value. 
     * 
     * @return The atomic value as optional. The optional is set if the atomic value was found.
     * 
     */
    inline intOptional getIntAtomic(const std::string& name)
    {
        return getAtomic<int>(name);
    }


};

} // namespace lvr2

#include "ChannelManager.tcc"

#endif // LVR2_TYPES_CHANNELMANAGER