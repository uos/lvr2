/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/*
 * Meap.tcc
 */

#include <lvr2/util/Panic.hpp>

using std::move;
using std::swap;

namespace lvr2
{

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
Meap<KeyT, ValueT, MapT>::Meap(size_t capacity)
{
    m_heap.reserve(capacity);
    // TODO: maybe add `reserve()` to the attribute map interface and add this
    // m_indices.reserve();
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
void Meap<KeyT, ValueT, MapT>::insert(const KeyT& key, const ValueT& value)
{
    // Insert to the back of the vector
    auto idx = m_heap.size();
    m_heap.push_back({ key, value });
    m_indices.insert(key, idx);

    // Correct heap by bubbling up
    bubbleUp(idx);
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
const MeapPair<KeyT, ValueT>& Meap<KeyT, ValueT, MapT>::peekMin() const
{
    if (m_heap.empty())
    {
        panic("attempt to peek at min in an empty heap");
    }

    return m_heap[0];
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
MeapPair<KeyT, ValueT> Meap<KeyT, ValueT, MapT>::popMin()
{
    if (m_heap.empty())
    {
        panic("attempt to peek at min in an empty heap");
    }

    // We only need to repair if there is more than one element left, because
    // after removing the one element, this heap doesn't contain any elements.
    // If you have zero elements, they can't be unordered.
    if (m_heap.size() != 1)
    {
        // Swap the minimal element with the last element in the vector
        swap(m_heap[0], m_heap.back());
        swap(m_indices[m_heap[0].key], m_indices[m_heap.back().key]);
        size_t currIndex = 0;

        // Checks if there exists a child of `father` which has a smaller
        // discriminator than the discriminator at `father`.
        const auto hasSmallerChildren = [this](size_t father)
        {
            // We use the size - 1, because we need to ignore the old min which
            // sits at the end of the vector now!
            const auto len = m_heap.size() - 1;
            const auto left = leftChild(father);
            const auto right = rightChild(father);
            const auto& disc = m_heap[father].value;
            return (left < len && m_heap[left].value < disc)
                || (right < len && m_heap[right].value < disc);
        };

        // Returns the index of the child of `father` with the smaller
        // discriminator. This function assumes that there exists at least one
        // (the left) child.
        const auto smallerChildOf = [this](size_t father)
        {
            const auto left = leftChild(father);
            const auto right = rightChild(father);

            // In case there exists only one child (again: size - 1 to ignore
            // the previous min at the end of the vector).
            if (right >= m_heap.size() - 1)
            {
                return left;
            }
            return m_heap[left].value > m_heap[right].value ? right : left;
        };

        // Repair the heap by sifting down the element
        while (hasSmallerChildren(currIndex))
        {
            const auto smallerChild = smallerChildOf(currIndex);
            swap(m_heap[smallerChild], m_heap[currIndex]);
            swap(m_indices[m_heap[smallerChild].key], m_indices[m_heap[currIndex].key]);
            currIndex = smallerChild;
        }
    }

    // Move (minimum) element out of the vector and return it
    const auto out = move(m_heap.back());
    m_heap.pop_back();
    m_indices.erase(out.key);
    return out;
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
void Meap<KeyT, ValueT, MapT>::decreaseValue(const KeyT& key, const ValueT& newValue)
{
    auto idx = m_indices[key];
    if (newValue > m_heap[idx].value)
    {
        // TODO: we might just allow a greater value. We can simply bubble
        // down, if we want to.
        panic("call to decreaseValue() with a greater new value");
    }
    else if (newValue < m_heap[idx].value)
    {
        m_heap[idx].value = newValue;
        bubbleUp(idx);
    }
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
bool Meap<KeyT, ValueT, MapT>::isEmpty() const
{
    return m_heap.empty();
}


template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
size_t Meap<KeyT, ValueT, MapT>::father(size_t child) const
{
    return (child - 1) / 2;
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
size_t Meap<KeyT, ValueT, MapT>::leftChild(size_t father) const
{
    return 2 * father + 1;
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
size_t Meap<KeyT, ValueT, MapT>::rightChild(size_t father) const
{
    return 2 * father + 2;
}

template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
void Meap<KeyT, ValueT, MapT>::bubbleUp(size_t idx)
{
    // Bubble new element up until the order is correct
    while (idx != 0 && m_heap[idx].value < m_heap[father(idx)].value)
    {
        swap(m_heap[idx], m_heap[father(idx)]);
        swap(m_indices[m_heap[idx].key], m_indices[m_heap[father(idx)].key]);
        idx = father(idx);
    }
}


} // namespace lvr2
