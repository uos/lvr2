namespace lssr
{

template<typename T>
bool VertexTraits<T>::HAS_COLOR = false;

template<typename CoordType>
bool VertexTraits<ColorVertex<CoordType> >::HAS_COLOR = true;

}
