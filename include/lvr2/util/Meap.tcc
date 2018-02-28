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

template<typename KeyT, typename ValueT>
Meap<KeyT, ValueT>::Meap(size_t capacity)
{
    m_heap.reserve(capacity);
    // TODO: maybe add `reserve()` to the attribute map interface and add this
    m_indices.reserve(capacity);
}

template<typename KeyT, typename ValueT>
bool Meap<KeyT, ValueT>::containsKey(KeyT key) const
{
    return m_indices.find(key) != m_indices.end();
}

template<typename KeyT, typename ValueT>
optional<ValueT> Meap<KeyT, ValueT>::insert(KeyT key, const ValueT& value)
{
    auto previous = m_indices.find(key);
    if (previous != m_indices.end())
    {
        auto prevValue = m_heap[previous->second].value;
        updateValue(key, value);
        return prevValue;
    }
    else
    {
        // Insert to the back of the vector
        auto idx = m_heap.size();
        m_heap.push_back({ key, value });
        m_indices.insert({key, idx});

        // Correct heap by bubbling up
        bubbleUp(idx);
        return boost::none;
    }
}

template<typename KeyT, typename ValueT>
void Meap<KeyT, ValueT>::clear()
{
    m_heap.clear();
    m_indices.clear();
}

template<typename KeyT, typename ValueT>
size_t Meap<KeyT, ValueT>::numValues() const
{
    return m_indices.size();
}

template<typename KeyT, typename ValueT>
optional<const ValueT&> Meap<KeyT, ValueT>::get(KeyT key) const
{
    auto maybeIndex = m_indices.get(key);
    if (maybeIndex)
    {
        return m_heap[*maybeIndex].value;
    }
    else
    {
        return boost::none;
    }
}

template<typename KeyT, typename ValueT>
const MeapPair<KeyT, ValueT>& Meap<KeyT, ValueT>::peekMin() const
{
    if (m_heap.empty())
    {
        panic("attempt to peek at min in an empty heap");
    }

    return m_heap[0];
}

template<typename KeyT, typename ValueT>
MeapPair<KeyT, ValueT> Meap<KeyT, ValueT>::popMin()
{
    if (m_heap.empty())
    {
        panic("attempt to peek at min in an empty heap");
    }

    // Swap the minimal element with the last element in the vector
    swap(m_heap[0], m_heap.back());
    swap(m_indices[m_heap[0].key], m_indices[m_heap.back().key]);

    // Move (minimum) element out of the vector
    const auto out = move(m_heap.back());
    m_heap.pop_back();
    m_indices.erase(out.key);

    // We only need to repair if there is more than one element left, because
    // after removing the one element, this heap doesn't contain any elements.
    // If you have zero elements, they can't be unordered.
    if (!m_heap.empty())
    {
        // At the root of the heap, there might be an element which is too big,
        // thus we need to bubble it down.
        bubbleDown(0);
    }

    return out;
}

template<typename KeyT, typename ValueT>
void Meap<KeyT, ValueT>::updateValue(const KeyT& key, const ValueT& newValue)
{
    auto idx = m_indices[key];
    if (newValue > m_heap[idx].value)
    {
        m_heap[idx].value = newValue;
        bubbleDown(idx);
    }
    else if (newValue < m_heap[idx].value)
    {
        m_heap[idx].value = newValue;
        bubbleUp(idx);
    }
}

template<typename KeyT, typename ValueT>
optional<ValueT> Meap<KeyT, ValueT>::erase(KeyT key)
{
    const auto maybeIndex = m_indices.find(key);
    if (maybeIndex == m_indices.end())
    {
        return boost::none;
    }

    auto index = maybeIndex->second;

    // Swap the element to remove with the last element in the vector
    auto swapKey = m_heap.back().key;
    swap(m_heap[index], m_heap.back());
    swap(m_indices[key], m_indices[swapKey]);

    // Move element out of the vector
    const auto out = move(m_heap.back()).value;
    m_heap.pop_back();
    m_indices.erase(key);

    // If the removed element was the last one in the meap, we don't have to
    // do any cleanup. Otherwise we have to put the previous last element into
    // the correct position.
    if (!m_heap.empty())
    {
        // If the element was deleted from the root (=> there is no father) or
        // if the father is already smaller than the current value, we attempt
        // to bubble the value down (which will do nothing if the position
        // is already correct). Otherwise it has to bubble up.
        if (index == 0 || m_heap[father(index)].value < m_heap[index].value)
        {
            bubbleDown(index);
        }
        else
        {
            bubbleUp(index);
        }
    }

    return out;
}


template<typename KeyT, typename ValueT>
bool Meap<KeyT, ValueT>::isEmpty() const
{
    return m_heap.empty();
}


template<typename KeyT, typename ValueT>
size_t Meap<KeyT, ValueT>::father(size_t child) const
{
    return (child - 1) / 2;
}

template<typename KeyT, typename ValueT>
size_t Meap<KeyT, ValueT>::leftChild(size_t father) const
{
    return 2 * father + 1;
}

template<typename KeyT, typename ValueT>
size_t Meap<KeyT, ValueT>::rightChild(size_t father) const
{
    return 2 * father + 2;
}

template<typename KeyT, typename ValueT>
void Meap<KeyT, ValueT>::bubbleUp(size_t idx)
{
    // Bubble new element up until the order is correct
    while (idx != 0 && m_heap[idx].value < m_heap[father(idx)].value)
    {
        swap(m_heap[idx], m_heap[father(idx)]);
        swap(m_indices[m_heap[idx].key], m_indices[m_heap[father(idx)].key]);
        idx = father(idx);
    }
}

template<typename KeyT, typename ValueT>
void Meap<KeyT, ValueT>::bubbleDown(size_t idx)
{
    // Checks if there exists a child of `father` which has a smaller value
    // than the value at `father`.
    const auto hasSmallerChildren = [this](size_t father)
    {
        const auto len = m_heap.size();
        const auto left = leftChild(father);
        const auto right = rightChild(father);
        const auto& disc = m_heap[father].value;
        return (left < len && m_heap[left].value < disc)
            || (right < len && m_heap[right].value < disc);
    };

    // Returns the index of the child of `father` with the smaller value. This
    // function assumes that there exists at least one (the left) child.
    const auto smallerChildOf = [this](size_t father)
    {
        const auto left = leftChild(father);
        const auto right = rightChild(father);

        // In case there exists only one child
        if (right >= m_heap.size())
        {
            return left;
        }
        return m_heap[left].value > m_heap[right].value ? right : left;
    };

    // Repair the heap by sifting down the element
    while (hasSmallerChildren(idx))
    {
        const auto smallerChild = smallerChildOf(idx);
        swap(m_heap[smallerChild], m_heap[idx]);
        swap(m_indices[m_heap[smallerChild].key], m_indices[m_heap[idx].key]);
        idx = smallerChild;
    }
}

template<typename KeyT, typename ValueT>
void Meap<KeyT, ValueT>::debugOutput() const
{
    size_t levelWidth = 1;
    size_t levelCount = 0;
    size_t totalCount = 0;
    unordered_set<KeyT> keys;

    cout << "HEAP:" << endl;
    for (auto& e: m_heap)
    {
        cout << "(" << e.key << " -> " << e.value << ")[" << totalCount << "], ";
        keys.insert(e.key);

        levelCount += 1;
        totalCount += 1;
        if (levelCount == levelWidth)
        {
            levelWidth *= 2;
            levelCount = 0;
            cout << endl;
        }
    }

    cout << endl << "MAP:" << endl;
    for (auto k: keys)
    {
        cout << k << " -> ";
        auto idx = m_indices.get(k);
        if (idx)
        {
            cout << *idx << endl;
        }
        else
        {
            cout << "!! NONE !!" << endl;
        }
    }
}

} // namespace lvr2
