namespace lvr2
{

template <typename T>
void ChunkHashGrid::setGeometryChunk(std::string layer, int x, int y, int z, T data)
{
    // store chunk persistently
    m_io.saveChunk<T>(data, layer, x, y, z);

    // update bounding box based on channel geometry 
    expandBoundingBox(data);

    // add chunk to cache
    loadChunk(layer, x, y, z, data);
}

template <typename T>
void ChunkHashGrid::setChunk(std::string layer, int x, int y, int z, T data)
{
    // store chunk persistently
    m_io.saveChunk<T>(data, layer, x, y, z);

    // update bounding box based on chunk index 
    if(x > getChunkMaxChunkIndex().x || y > getChunkMaxChunkIndex().y || z > getChunkMaxChunkIndex().z ||
        x < getChunkMinChunkIndex().x || y < getChunkMinChunkIndex().y || z < getChunkMinChunkIndex().z)
    {
        BoundingBox<BaseVector<float>> boundingBox = getBoundingBox();
        boundingBox.expand(BaseVector<float>(x * m_chunkSize, y * m_chunkSize, z * m_chunkSize));
        setBoundingBox(boundingBox);
    }

    // add chunk to cache
    loadChunk(layer, x, y, z, data);
}

template <typename T>
boost::optional<T> ChunkHashGrid::getChunk(std::string layer, int x, int y, int z)
{
    // skip if the Coordinates are too large or too negative
    if(x > getChunkMaxChunkIndex().x || y > getChunkMaxChunkIndex().y || z > getChunkMaxChunkIndex().z ||
        x < getChunkMinChunkIndex().x || y < getChunkMinChunkIndex().y || z < getChunkMinChunkIndex().z)
    {
        return boost::optional<T>{};
    }
    std::size_t chunkHash = hashValue(x, y, z);

    if (isChunkLoaded(layer, chunkHash))
    {
        // move chunk to the front of the cache queue
        m_items.remove({layer, chunkHash});
        m_items.push_front({layer, chunkHash});

        return boost::get<T>(m_hashGrid[layer][chunkHash]);
    }

    if (loadChunk<T>(layer, x, y, z))
    {
        return boost::get<T>(m_hashGrid[layer][chunkHash]);
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
