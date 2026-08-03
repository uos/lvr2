#include <optional>
#include <gtest/gtest.h>
#include <lvr2/algorithm/ClosestSurfacePoint.hpp>
#include <lvr2/algorithm/raycasting/BVHRaycaster.hpp>
#include <lvr2/algorithm/raycasting/Intersection.hpp>
#include <lvr2/types/MatrixTypes.hpp>
#include <lvr2/types/MeshBuffer.hpp>
#include <lvr2/util/Synthetic.hpp>

TEST(BVHClosestPoint, EmptyMesh) {
    // An BVHRaycaster build from an empty mesh should always
    // return std::nullopt as the closest point
    
    const lvr2::MeshBufferPtr mesh = std::make_shared<lvr2::MeshBuffer>();
    const auto raycaster = std::make_unique<lvr2::BVHRaycaster<lvr2::AllInt>>(mesh);

    const lvr2::Vector3f query = lvr2::Vector3f::Ones();
    ASSERT_EQ(raycaster->getClosestPoint(query), std::nullopt);
}

TEST(BVHClosestPoint, Sphere) {
    // A BVHRaycaster build from a valid mesh should return a valid point.
    // On a unit sphere we also know the closest point pretty well.
    // The sphere has some approximation error due to mesh discretization,
    // therefore we only test for mm accuracy below.
    
    // NOTE: The default sphere discretization is way to low!
    const lvr2::MeshBufferPtr mesh = lvr2::synthetic::genSphere(360, 360);
    const auto raycaster = std::make_unique<lvr2::BVHRaycaster<lvr2::AllInt>>(mesh);

    // X-Axis
    const lvr2::Vector3f queryX = lvr2::Vector3f(2.0, 0.0, 0.0);
    const std::optional<lvr2::ClosestSurfacePointQueryResult> resultX = raycaster->getClosestPoint(queryX);
    ASSERT_TRUE(resultX.has_value());
    ASSERT_LT((resultX.value().point - lvr2::Vector3f(1.0, 0.0, 0.0)).norm(), 0.001);

    // Y-Axis
    const lvr2::Vector3f queryY = lvr2::Vector3f(0.0, 2.0, 0.0);
    const std::optional<lvr2::ClosestSurfacePointQueryResult> resultY = raycaster->getClosestPoint(queryY);
    ASSERT_TRUE(resultY.has_value());
    ASSERT_LT((resultY.value().point - lvr2::Vector3f(0.0, 1.0, 0.0)).norm(), 0.001);

    // Z-Axis
    const lvr2::Vector3f queryZ = lvr2::Vector3f(0.0, 0.0, 2.0);
    const std::optional<lvr2::ClosestSurfacePointQueryResult> resultZ = raycaster->getClosestPoint(queryZ);
    ASSERT_TRUE(resultZ.has_value());
    ASSERT_LT((resultZ.value().point - lvr2::Vector3f(0.0, 0.0, 1.0)).norm(), 0.001);
}
