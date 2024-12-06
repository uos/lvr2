namespace lvr2
{

template <typename BaseIO>
void ChunkIO<BaseIO>::saveAmount(BaseVector<std::size_t> amount)
{
    boost::shared_array<size_t> amountArr(new size_t[3]{amount.x, amount.y, amount.z});
    m_array_io->save(m_chunkName, m_amountName, 3, amountArr);
}

template <typename BaseIO>
void ChunkIO<BaseIO>::saveChunkSize(float chunkSize)
{
    boost::shared_array<float> chunkSizeArr(new float[1]{chunkSize});
    m_array_io->save(m_chunkName, m_chunkSizeName, 1, chunkSizeArr);
}

template <typename BaseIO>
void ChunkIO<BaseIO>::saveBoundingBox(BoundingBox<BaseVector<float>> boundingBox)
{
    boost::shared_array<float> boundingBoxArr(new float[6]{boundingBox.getMin()[0],
                                                           boundingBox.getMin()[1],
                                                           boundingBox.getMin()[2],
                                                           boundingBox.getMax()[0],
                                                           boundingBox.getMax()[1],
                                                           boundingBox.getMax()[2]});
    std::vector<size_t> boundingBoxDim({2, 3});
    m_array_io->save(m_chunkName, m_boundingBoxName, boundingBoxDim, boundingBoxArr);
}

template <typename BaseIO>
void ChunkIO<BaseIO>::save(BaseVector<std::size_t> amount,
                            float chunkSize,
                            BoundingBox<BaseVector<float>> boundingBox)
{
    saveAmount(amount);
    saveChunkSize(chunkSize);
    saveBoundingBox(boundingBox);
}

template <typename BaseIO>
template <typename T>
void ChunkIO<BaseIO>::saveChunk(T data, std::string layer, int x, int y, int z)
{
    std::string chunkName = std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);

    /// TODO: Put this into a seperate decription object
    /// and add it to this class!
    boost::filesystem::path chunkPath(m_chunkName);
    boost::filesystem::path layerPath(layer);

    std::string groupName = (chunkPath / layerPath).string();
    static_cast<typename IOType<BaseIO, T>::io_type*>(m_baseIO)->save(groupName, chunkName, data);
}

template <typename BaseIO>
BaseVector<size_t> ChunkIO<BaseIO>::loadAmount()
{
    BaseVector<size_t> amount;
    size_t dimensionAmount;
    boost::shared_array<size_t> amountArr
        = m_array_io->template load<size_t>(m_chunkName, m_amountName, dimensionAmount);
    if (dimensionAmount != 3)
    {
        lvr2::logout::get() << lvr2::error << "Error loading chunk data: amount has not the correct "
                     "dimension. Found: "
                  << dimensionAmount << "; Expected: 3" << lvr2::endl;
    }
    else
    {
        amount = BaseVector<size_t>(amountArr[0], amountArr[1], amountArr[2]);
    }
    return amount;
}

template <typename BaseIO>
float ChunkIO<BaseIO>::loadChunkSize()
{
    float chunkSize;
    size_t dimensionChunkSize;
    boost::shared_array<float> chunkSizeArr
        = m_array_io->template load<float>(m_chunkName, m_chunkSizeName, dimensionChunkSize);
    if (dimensionChunkSize != 1)
    {
        lvr2::logout::get() << lvr2::error << "Error loading chunk data: chunkSize has not the correct "
                     "dimension. Found: "
                  << dimensionChunkSize << "; Expected: 1" << lvr2::endl;
        chunkSize = 0;
    }
    else
    {
        chunkSize = chunkSizeArr[0];
    }
    return chunkSize;
}

template <typename BaseIO>
BoundingBox<BaseVector<float>> ChunkIO<BaseIO>::loadBoundingBox()
{
    BoundingBox<BaseVector<float>> boundingBox;
    std::vector<size_t> dimensionBox;
    boost::shared_array<float> boundingBoxArr
        = m_array_io->template load<float>(m_chunkName, m_boundingBoxName, dimensionBox);
    if (dimensionBox.size() < 2)
    {
        throw std::out_of_range(
            "Error loading chunk data: bounding_box dimensions do not match. Found: "
            + std::to_string(dimensionBox.size()) + "; Expected: 2");
    }
    if (dimensionBox.at(0) != 2 && dimensionBox.at(1) != 3)
    {
        throw std::out_of_range(
            "Error loading chunk data: bounding_box dimensions do not match. Found: "
            + std::to_string(dimensionBox.at(0)) + ", " + std::to_string(dimensionBox.at(1))
            + "; Expected: {2, 3}");
    }
    else
    {
        boundingBox = BoundingBox<BaseVector<float>>(
            BaseVector<float>(boundingBoxArr[0], boundingBoxArr[1], boundingBoxArr[2]),
            BaseVector<float>(boundingBoxArr[3], boundingBoxArr[4], boundingBoxArr[5]));
    }
    return boundingBox;
}

template <typename BaseIO>
template <typename T>
T ChunkIO<BaseIO>::loadChunk(std::string layer, int x, int y, int z)
{
    std::string chunkName = std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);

    return static_cast<typename IOType<BaseIO, T>::io_type*>(m_baseIO)
        ->load(m_chunkName + "/" + layer + "/" + chunkName);
}

} // namespace lvr2
