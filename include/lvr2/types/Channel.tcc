namespace lvr2 {

template<typename T>
Channel<T>::Channel()
: m_elementWidth(0)
, m_numElements(0)
{}

template<typename T>
Channel<T>::Channel(size_t n, size_t width)
: m_elementWidth(width), m_numElements(n)
, m_data(new T[n * width])
{}

template<typename T>
Channel<T>::Channel(size_t n, size_t width, DataPtr ptr)
: m_numElements(n)
, m_elementWidth(width)
, m_data(ptr)
{}

template<typename T>
ElementProxy<T> Channel<T>::operator[](const unsigned& idx)
{
    T* ptr = m_data.get();
    return ElementProxy<T>(&(ptr[idx * m_elementWidth]), m_elementWidth);
}

template<typename T>
size_t Channel<T>::width() const 
{
    return m_elementWidth;
}

template<typename T>
size_t Channel<T>::numElements() const 
{
    return m_numElements;
}

template<typename T>
typename Channel<T>::DataPtr Channel<T>::dataPtr() const {
    return m_data;
}


} // namespace lvr2