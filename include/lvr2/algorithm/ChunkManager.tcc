namespace lvr2
{

template <typename T>
ChannelPtr<T> ChunkManager::extractChannelOfArea(
    std::unordered_map<std::size_t, MeshBufferPtr>& chunks,
    std::string channelName,
    std::size_t staticVertexIndexOffset,
    std::size_t numVertices,
    std::size_t numFaces,
    std::vector<std::unordered_map<std::size_t, std::size_t>>& areaVertexIndices)
{
    ChannelPtr<T> channel = nullptr;

    auto areaVertexIndicesIt   = areaVertexIndices.begin();
    std::size_t dynIndexOffset = 0;

    for (auto chunkIt = chunks.begin(); chunkIt != chunks.end(); ++chunkIt)
    {
        MeshBufferPtr chunk                           = chunkIt->second;
        typename Channel<T>::Optional chunkChannelOpt = chunk->getChannel<T>(channelName);

        if (chunkChannelOpt)
        {
            Channel<T> chunkChannel = *chunkChannelOpt;

            if (!channel)
            {
                size_t numElements = chunkChannel.numElements();
                if (chunkChannel.numElements() == chunk->numVertices())
                {
                    std::cout << "adding vertex attribute '" << channelName << "'" << std::endl;
                    numElements = numVertices;
                }
                else if (chunkChannel.numElements() == chunk->numFaces())
                {
                    std::cout << "adding face attribute '" << channelName << "'" << std::endl;
                    numElements = numFaces;
                }
                else
                {
                    std::cout << "adding other attribute '" << channelName << "'" << std::endl;
                }

                channel = std::make_shared<Channel<T>>(
                    numElements,
                    chunkChannel.width(),
                    boost::shared_array<T>(new T[numElements * chunkChannel.width()]));
            }

            if (chunkChannel.numElements() == chunk->numVertices())
            {
                // add data to vertex attribute

                std::size_t numDuplicates = *chunk->getAtomic<unsigned int>("num_duplicates");
                std::size_t indexOffset = staticVertexIndexOffset - numDuplicates + dynIndexOffset;
                dynIndexOffset += chunkChannel.numElements() - numDuplicates;

                for (unsigned int i = 0; i < chunkChannel.numElements(); i++)
                {
                    size_t index = 0;

                    auto it = (*areaVertexIndicesIt).find(i);
                    if (it != (*areaVertexIndicesIt).end())
                    {
                        index = it->second;
                    }
                    else
                    {
                        index = i + indexOffset;
                    }

                    for (unsigned int j = 0; j < channel->width(); j++)
                    {
                        channel->dataPtr()[index * channel->width() + j]
                            = chunkChannel.dataPtr()[i * channel->width() + j];
                    }
                }
            }
            else if (chunkChannel.numElements() == chunk->numFaces())
            {
                // add data to face attribute

                for (unsigned int i = 0; i < chunkChannel.numElements(); i++)
                {
                    size_t index = dynIndexOffset + i;

                    for (unsigned int j = 0; j < channel->width(); j++)
                    {
                        channel->dataPtr()[index * channel->width() + j]
                            = chunkChannel.dataPtr()[i * channel->width() + j];
                    }
                }

                dynIndexOffset += chunkChannel.numElements();
            }
            else
            {
                // add data to other attribute

                for (unsigned int i = 0; i < chunkChannel.numElements(); i++)
                {
                    for (unsigned int j = 0; j < channel->width(); j++)
                    {
                        channel->dataPtr()[i * channel->width() + j]
                            = chunkChannel.dataPtr()[i * channel->width() + j];
                    }
                }
            }
        }

        ++areaVertexIndicesIt;
    }

    return channel;
}

} // namespace lvr2
