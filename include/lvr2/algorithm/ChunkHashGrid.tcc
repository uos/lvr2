namespace lvr2
{

template <typename T>
void ChunkHashGrid::setChunk(std::string layer, int x, int y, int z, T data)
{
    // store chunk persistently
    m_io.saveChunk<T>(data, layer, x, y, z);

    // add chunk to cache
    loadChunk(layer, x, y, z, data);
}

template <typename T>
boost::optional<T> ChunkHashGrid::getChunk(std::string layer, int x, int y, int z)
{
    std::size_t chunkHash = hashValue(x, y, z);

    if (isChunkLoaded(layer, chunkHash))
    {
        // move chunk to the front of the cache queue
        m_items.remove({layer, chunkHash});
        m_items.push_front({layer, chunkHash});

        return m_hashGrid[layer][chunkHash];
    }

    if (loadChunk<T>(layer, x, y, z))
    {
        return m_hashGrid[layer][chunkHash];
    }

    return boost::optional<T>{};
}

template <typename T>
bool ChunkHashGrid::loadChunk(std::string layer, int x, int y, int z)
{
    if (isChunkLoaded(layer, x, y, z))
    {
        return true;
    }

    T data = m_io.loadChunk<T>(layer, x, y, z);
    if (data == nullptr)
    {
        return false;
    }

    loadChunk(layer, x, y, z, data);

    return true;
}

} // namespace lvr2
