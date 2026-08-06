#include <gtest/gtest.h>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/types/MatrixTypes.hpp>

TEST(BaseVector, EigenConversions) {
    const lvr2::BaseVector<float> bv(1.5f, -2.0f, 3.14f);
    const lvr2::Vector3f vec = static_cast<lvr2::Vector3f>(bv);

    ASSERT_FLOAT_EQ(vec.x(), 1.5f);
    ASSERT_FLOAT_EQ(vec.y(), -2.0f);
    ASSERT_FLOAT_EQ(vec.z(), 3.14f);

    // Convert back
    const lvr2::BaseVector<float> new_bv = vec;
    ASSERT_FLOAT_EQ(new_bv.x, 1.5f);
    ASSERT_FLOAT_EQ(new_bv.y, -2.0f);
    ASSERT_FLOAT_EQ(new_bv.z, 3.14f);
}

