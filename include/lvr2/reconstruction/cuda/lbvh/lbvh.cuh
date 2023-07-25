/**
 * Copyright (c) 2023, University Osnabrück
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

#pragma once
#ifndef LVR2_LBVH_CUH
#define LVR2_LBVH_CUH


#include <cuda/std/limits>

#include "aabb.cuh"
#include "morton_code.cuh"

namespace lvr2
{

namespace lbvh 
{
    struct __align__(16) BVHNode // size: 48 bytes
    {
        AABB bounds; // 24 bytes
        unsigned int parent; // 4
        unsigned int child_left; // 4 bytes
        unsigned int child_right; // 4 bytes
        int atomic; // 4
        unsigned int range_left; // 4
        unsigned int range_right; // 4
    };

    __device__
    inline HashType morton_code(
        const float3& point, 
        const lbvh::AABB &extent, 
        float resolution = 1024.0
    ) noexcept 
    {
        float3 p = point;

        // scale to [0, 1]
        p.x -= extent.min.x;
        p.y -= extent.min.y;
        p.z -= extent.min.z;

        p.x /= (extent.max.x - extent.min.x);
        p.y /= (extent.max.y - extent.min.y);
        p.z /= (extent.max.z - extent.min.z);
        return morton_code(p, resolution);
    }

    __device__
    inline HashType morton_code(
        const lbvh::AABB &box, 
        const lbvh::AABB &extent, 
        float resolution = 1024.0
    ) noexcept 
    {
        auto p = centroid(box);
        return morton_code(p, extent, resolution);
    }


    __device__ inline bool is_leaf(const BVHNode* node) 
    {
        return node->child_left == UINT_MAX && node->child_right == UINT_MAX;
    }
    
    // Sets the bounding box and traverses to root
    __device__ inline void process_parent(unsigned int node_idx,
                                    BVHNode* nodes,
                                    const unsigned long long int *morton_codes,
                                    unsigned int* root_index,
                                    unsigned int N)
    {
        unsigned int current_idx = node_idx;
        BVHNode* current_node = &nodes[current_idx];

        while(true) {
            // Allow only one thread to process a node
            if (atomicAdd(&(current_node->atomic), 1) != 1)
                return; // terminate the first thread encountering this

            unsigned int left = current_node->range_left;
            unsigned int right = current_node->range_right;

            // Set bounding box if the node is no leaf
            if (!is_leaf(current_node)) {
                // Fuse bounding box from children AABBs
                current_node->bounds = merge(nodes[current_node->child_left].bounds,
                                                nodes[current_node->child_right].bounds);
            }


            if (left == 0 && right == N - 1) {
                root_index[0] = current_idx; // return the root
                return; // at the root, abort
            }


            unsigned int parent_idx;
            BVHNode *parent;

            if (left == 0 || (right != N - 1 && highest_bit(morton_codes[right], morton_codes[right + 1]) <
                                                highest_bit(morton_codes[left - 1], morton_codes[left]))) {
                // parent = right, set parent left child and range to node
                parent_idx = N + right;

                parent = &nodes[parent_idx];
                parent->child_left = current_idx;
                parent->range_left = left;
            } else {
                // parent = left -1, set parent right child and range to node
                parent_idx = N + left - 1;

                parent = &nodes[parent_idx];
                parent->child_right = current_idx;
                parent->range_right = right;
            }

            current_node->parent = parent_idx; // store the parent in the current node

            // up to the parent next
            current_node = parent;
            current_idx = parent_idx;
        }
    }

    /**
     * Merge an internal node into a leaf node using the leftmost leaf node of the subtree
     * @tparam T
     * @param node
     * @param leaf
     */
    __forceinline__ __device__ void make_leaf(
        unsigned int node_idx,
        unsigned int leaf_idx,
        BVHNode* nodes, unsigned int N
    ) 
    {
        BVHNode* node = &nodes[node_idx];

        unsigned int parent_idx = node->parent;

        BVHNode* leaf = &nodes[leaf_idx];

        leaf->parent = parent_idx;
        leaf->bounds = node->bounds;

        // copy the range into the leaf
        leaf->range_left = node->range_left;
        leaf->range_right = node->range_right;

        // adjust the structure at the node's parent
        if(parent_idx != UINT_MAX) {
            BVHNode* parent = &nodes[parent_idx];

            if(parent->child_left == node_idx) {
                parent->child_left = leaf_idx;
            } else {
                parent->child_right = leaf_idx;
            }
        }
    }
} // namespace lbvh

} // namespace lvr2

#endif // LVR2_LBVH_CUH