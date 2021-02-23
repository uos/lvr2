#ifndef LVR2_IO_YAML_UTIL_HPP
#define LVR2_IO_YAML_UTIL_HPP

#include <string>

namespace YAML {

inline const Node & cnode(const Node &n) {
    return n;
}

Node MergeNodes(Node a, Node b)
{
    if (!b.IsMap()) {
        // If b is not a map, merge result is b, unless b is null
        return b.IsNull() ? a : b;
    }
    if (!a.IsMap()) {
        // If a is not a map, merge result is b
        return b;
    }
    if (!b.size()) {
        // If a is a map, and b is an empty map, return a
        return a;
    }
    // Create a new map 'c' with the same mappings as a, merged with b
    auto c = Node(NodeType::Map);
    for (auto n : a) {
        if (n.first.IsScalar()) {
        const std::string & key = n.first.Scalar();
        auto t = Node(cnode(b)[key]);
        if (t) {
            c[n.first] = MergeNodes(n.second, t);
            continue;
        }
        }
        c[n.first] = n.second;
    }
    // Add the mappings from 'b' not already in 'c'
    for (auto n : b) {
        if (!n.first.IsScalar() || !cnode(c)[n.first.Scalar()]) {
        c[n.first] = n.second;
        }
    }
    return c;
}

} // namespace YAML

#endif // LVR2_IO_YAML_UTIL_HPP