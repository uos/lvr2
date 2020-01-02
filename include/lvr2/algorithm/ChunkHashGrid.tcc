namespace lvr2
{

template <typename T>
T ChunkHashGrid::findChunk(std::string layer, size_t hashValue, int x, int y, int z)
{
    return boost::get<T>(findVariantChunk(layer, hashValue, x, y, z));
}

} // namespace lvr2
