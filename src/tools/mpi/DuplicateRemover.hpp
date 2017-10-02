#ifndef LAS_VEGAS_DUPLICATEREMOVER_HPP
#define LAS_VEGAS_DUPLICATEREMOVER_HPP

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <lvr/geometry/HalfEdgeMesh.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/MeshBuffer.hpp>
#include <lvr/io/PLYIO.hpp>
#include <lvr/io/DataStruct.hpp>
#include <lvr/io/Progress.hpp>
#include "sortPoint.hpp"
#include <utility>


namespace lvr
{

class DuplicateRemover
{
public:
    MeshBufferPtr removeDuplicates(MeshBufferPtr mptr);
};


}
#endif //LAS_VEGAS_DUPLICATEREMOVER_HPP