namespace lssr
{

template<typename T>
bool VertexTraits<T>::HAS_COLOR = false;

template<typename CoordType, typename ColorT>
bool VertexTraits<ColorVertex<CoordType, ColorT> >::HAS_COLOR = true;

}
