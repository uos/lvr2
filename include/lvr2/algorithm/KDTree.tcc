/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * KDTree.tcc
 *
 *  @date Apr 28, 2019
 *  @author Malte Hillmann
 */

namespace lvr2
{

template<typename PointT, unsigned int N>
typename KDTree<PointT, N>::Ptr KDTree<PointT, N>::create(std::unique_ptr<PointT[]>&& points, size_t numPoints, size_t maxLeafSize)
{
    Ptr ret(new KDTree<PointT, N>(std::move(points), numPoints));
    ret->init(maxLeafSize);

    return ret;
}

template<typename PointT, unsigned int N>
void KDTree<PointT, N>::init(size_t maxLeafSize)
{
    QueryPoint bbMin = QueryPoint::Constant(std::numeric_limits<double>::max());
    QueryPoint bbMax = QueryPoint::Constant(-std::numeric_limits<double>::max());

    #pragma omp parallel
    {
        QueryPoint min = bbMin;
        QueryPoint max = bbMax;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < m_numPoints; i++)
        {
            for (unsigned int j = 0; j < N; j++)
            {
                min[j] = std::min(min[j], (double)m_points[i][j]);
                max[j] = std::max(max[j], (double)m_points[i][j]);
            }
        }
        #pragma omp critical
        {
            bbMin = bbMin.cwiseMin(min);
            bbMax = bbMax.cwiseMax(max);
        }
    }

    #pragma omp parallel // allows "pragma omp task"
    #pragma omp single // only execute every task once
    m_tree = createRecursive(m_points.get(), m_points.get() + m_numPoints, bbMin, bbMax, maxLeafSize);
}


template<typename PointT, unsigned int N>
template<typename InPointT, typename FloatT>
bool KDTree<PointT, N>::nnSearch(const InPointT& inPoint,
                                 PointT*& neighbor,
                                 FloatT& distance,
                                 double maxDistance) const
{
    QueryPoint point = toQueryPoint(inPoint);

    DistPoint candidate;
    candidate.point = nullptr;
    candidate.distanceSq = maxDistance * maxDistance;

    m_tree->nnInternal(point, candidate);

    neighbor = candidate.point; // might be nullptr
    distance = std::sqrt(candidate.distanceSq); // might be infinity
    return neighbor != nullptr;
}

template<typename PointT, unsigned int N>
template<typename InPointT, typename FloatT>
size_t KDTree<PointT, N>::knnSearch(const InPointT& inPoint,
                                    size_t k,
                                    std::vector<PointT*>& neighbors,
                                    std::vector<FloatT>& distances,
                                    double maxDistance) const
{
    QueryPoint point = toQueryPoint(inPoint);
    double worstDistSq = maxDistance * maxDistance;
    Queue queue;
    m_tree->knnInternal(point, k, queue, worstDistSq);

    neighbors.resize(queue.size());
    distances.resize(queue.size());
    // Fill the return vector from the back
    size_t index = neighbors.size() - 1;
    while (!queue.empty())
    {
        auto& p = queue.top();
        neighbors[index] = p.point;
        distances[index] = std::sqrt(p.distanceSq);
        queue.pop();
        index--;
    }

    return neighbors.size();
}

template<typename PointT, unsigned int N>
template<typename InPointT>
size_t KDTree<PointT, N>::knnSearch(const InPointT& inPoint,
                                    size_t k,
                                    std::vector<PointT*>& neighbors,
                                    double maxDistance) const
{
    QueryPoint point = toQueryPoint(inPoint);
    double worstDistSq = maxDistance * maxDistance;
    Queue queue;
    m_tree->knnInternal(point, k, queue, worstDistSq);

    neighbors.resize(queue.size());
    // Fill the return vector from the back
    size_t index = neighbors.size() - 1;
    while (!queue.empty())
    {
        auto& p = queue.top();
        neighbors[index] = p.point;
        queue.pop();
        index--;
    }

    return neighbors.size();
}

template<typename PointT, unsigned int N>
class KDTree<PointT, N>::KDNode : public KDTree<PointT, N>::KDTreeInternal
{
public:
    /**
     * @brief Construct a new KDNode object
     *
     * @param lesser The Subtree where point[axis] <= split
     * @param greater The Subtree where point[axis] >= split
     * @param axis The Axis to split along
     * @param split The Split value of the Axis
     */
    KDNode(KDPtr&& lesser, KDPtr&& greater, unsigned int axis, double split)
        : axis(axis), split(split), lesser(std::move(lesser)), greater(std::move(greater))
    {}
    virtual ~KDNode() = default;

    std::pair<KDTreeInternal*, KDTreeInternal*> childOrder(const QueryPoint& point) const
    {
        return point[axis] < split ? std::make_pair(lesser.get(), greater.get()) : std::make_pair(greater.get(), lesser.get());
    }

    void nnInternal(const QueryPoint& point, DistPoint& neighbor) const override
    {
        double cmpDistSq = std::pow(point[axis] - split, 2);
        auto [ first, second ] = childOrder(point);
        first->nnInternal(point, neighbor);
        if (cmpDistSq < neighbor.distanceSq)
        {
            second->nnInternal(point, neighbor);
        }
    }
    void knnInternal(const QueryPoint& point, size_t k, Queue& neighbors, double& worstDistSq) const override
    {
        double cmpDistSq = std::pow(point[axis] - split, 2);
        auto [ first, second ] = childOrder(point);
        first->knnInternal(point, k, neighbors, worstDistSq);
        if (cmpDistSq < worstDistSq)
        {
            second->knnInternal(point, k, neighbors, worstDistSq);
        }
    }

    unsigned int axis;
    double split;
    KDPtr lesser, greater;
};

template<typename PointT, unsigned int N>
class KDTree<PointT, N>::KDLeaf : public KDTree<PointT, N>::KDTreeInternal
{
public:
    KDLeaf(PointT* start, PointT* end)
        : start(start), end(end)
    {}
    virtual ~KDLeaf() = default;

    void nnInternal(const QueryPoint& point, DistPoint& neighbor) const override
    {
        for (PointT* iter = start; iter != end; ++iter)
        {
            double distanceSq = 0;
            for (unsigned int i = 0; i < N; i++)
            {
                distanceSq += std::pow(point[i] - (*iter)[i], 2);
            }
            if (distanceSq < neighbor.distanceSq)
            {
                neighbor.point = iter;
                neighbor.distanceSq = distanceSq;
            }
        }
    }
    void knnInternal(const QueryPoint& point, size_t k, Queue& neighbors, double& worstDistSq) const override
    {
        for (PointT* iter = start; iter != end; ++iter)
        {
            DistPoint p;
            p.distanceSq = 0;
            for (unsigned int i = 0; i < N; i++)
            {
                p.distanceSq += std::pow(point[i] - (*iter)[i], 2);
            }
            if (p.distanceSq < worstDistSq)
            {
                p.point = iter;
                neighbors.push(p);
                if (neighbors.size() > k)
                {
                    neighbors.pop();
                    worstDistSq = neighbors.top().distanceSq;
                }
            }
        }
    }

    PointT* start;
    PointT* end;
};

template<typename PointT, unsigned int N>
typename KDTree<PointT, N>::KDPtr KDTree<PointT, N>::createRecursive(
    PointT* start, PointT* end,
    const QueryPoint& min, const QueryPoint& max,
    size_t maxLeafSize)
{
    size_t n = end - start;
    if (n <= maxLeafSize)
    {
        return KDPtr(new KDLeaf(start, end));
    }

    unsigned int splitAxis = 0;
    double axisLength = max[0] - min[0];
    for (unsigned int i = 1; i < N; i++)
    {
        if (max[i] - min[i] > axisLength)
        {
            splitAxis = i;
            axisLength = max[i] - min[i];
        }
    }

    // find the middle on the split axis
    PointT* mid = start + n / 2;
    std::nth_element(start, mid, end, [splitAxis](const PointT & a, const PointT & b)
    {
        return a[splitAxis] < b[splitAxis];
    });
    double splitValue = (*mid)[splitAxis];

    // recursively create subtrees

    KDPtr lesser, greater;
    QueryPoint lesserMin = min, lesserMax = max;
    QueryPoint greaterMin = min, greaterMax = max;
    lesserMax[splitAxis] = splitValue;
    greaterMin[splitAxis] = splitValue;

    if (n > 8 * maxLeafSize) // stop the omp task subdivision early to avoid spamming tasks
    {
        #pragma omp task shared(lesser)
        lesser = createRecursive(start, mid, lesserMin, lesserMax, maxLeafSize);

        #pragma omp task shared(greater)
        greater = createRecursive(mid, end, greaterMin, greaterMax, maxLeafSize);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        #pragma omp barrier
#else
        #pragma omp taskwait
#endif
    }
    else
    {
        lesser = createRecursive(start, mid, lesserMin, lesserMax, maxLeafSize);
        greater = createRecursive(mid, end, greaterMin, greaterMax, maxLeafSize);
    }

    return KDPtr(new KDNode(std::move(lesser), std::move(greater), splitAxis, splitValue));
}

template<typename BaseVecT>
SearchKDTree<BaseVecT>::SearchKDTree(PointBufferPtr buffer)
    : KDTree<IndexedPoint<BaseVecT>>(new PointT[buffer->numPoints()], buffer->numPoints())
{
    auto points = *buffer->getFloatChannel("points");
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->m_numPoints; i++)
    {
        this->m_points[i].index = i;
        this->m_points[i].point = points[i];
    }

    this->init();
}

template<typename BaseVecT>
int SearchKDTree<BaseVecT>::kSearch(
    const BaseVecT& qp,
    int k,
    std::vector<size_t>& indices,
    std::vector<CoordT>& distances
) const
{
    std::vector<PointT*> neighbors;
    size_t n = this->knnSearch(qp, k, neighbors, distances);

    indices.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        indices[i] = neighbors[i]->index;
    }
    return n;
}

template<typename BaseVecT>
int SearchKDTree<BaseVecT>::kSearch(
    const BaseVecT& qp,
    int k,
    std::vector<size_t>& indices
) const
{
    std::vector<PointT*> neighbors;
    size_t n = this->knnSearch(qp, k, neighbors);

    indices.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        indices[i] = neighbors[i]->index;
    }
    return n;
}

template<typename BaseVecT>
int SearchKDTree<BaseVecT>::radiusSearch(
    const BaseVecT& qp,
    int k,
    float r,
    std::vector<size_t>& indices,
    std::vector<CoordT>& distances
) const
{
    std::vector<PointT*> neighbors;
    size_t n = this->knnSearch(qp, k, neighbors, distances, r);

    indices.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        indices[i] = neighbors[i]->index;
    }
    return n;
}

} // namespace lvr2
