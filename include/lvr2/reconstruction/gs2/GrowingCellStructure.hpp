//
// Created by patrick on 10.02.19.
//

#ifndef LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP
#define LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP

#include "lvr2/attrmaps/HashMap.hpp"
#include "lvr2/config/BaseOption.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/gs2/DynamicKDTree.hpp"
#include "lvr2/reconstruction/gs2/TumbleTree.hpp"

namespace lvr2
{

template <typename BaseVecT, typename NormalT>
class GrowingCellStructure
{
  public:
    /**
     * Construct a GCS instance
     * @param surface pointsetsurface to get pointcloud information from
     */
    GrowingCellStructure(PointsetSurfacePtr<BaseVecT>& surface);

    /**
     * Public method of the Reconstruction Class, calling all the other methods, generating the mesh
     * approximating the pointcloud's surface
     * @param mesh pointer to the mesh
     */
    void getMesh(HalfEdgeMesh<BaseVecT>& mesh);

    int getRuntime() const { return m_runtime; }

    int getBasicSteps() const { return m_basicSteps; }

    int getNumSplits() const { return m_numSplits; }

    float getBoxFactor() const { return m_boxFactor; }

    bool isWithCollapse() const { return m_withCollapse; }

    float getLearningRate() const { return m_learningRate; }

    float getNeighborLearningRate() const { return m_neighborLearningRate; }

    float getDecreaseFactor() const { return m_decreaseFactor; }

    int getAllowMiss() const { return m_allowMiss; }

    float getCollapseThreshold() const { return m_collapseThreshold; }

    bool isFilterChain() const { return m_filterChain; }

    int getDeleteLongEdgesFactor() const { return m_deleteLongEdgesFactor; }

    bool isInterior() const { return m_interior; }

    void setRuntime(int m_runtime) { GrowingCellStructure::m_runtime = m_runtime; }

    void setBasicSteps(int m_basicSteps) { GrowingCellStructure::m_basicSteps = m_basicSteps; }

    void setNumSplits(int m_numSplits) { GrowingCellStructure::m_numSplits = m_numSplits; }

    void setBoxFactor(float m_boxFactor) { GrowingCellStructure::m_boxFactor = m_boxFactor; }

    void setWithCollapse(bool m_withCollapse)
    {
        GrowingCellStructure::m_withCollapse = m_withCollapse;
    }

    void setLearningRate(float m_learningRate)
    {
        GrowingCellStructure::m_learningRate = m_learningRate;
    }

    void setNeighborLearningRate(float m_neighborLearningRate)
    {
        GrowingCellStructure::m_neighborLearningRate = m_neighborLearningRate;
    }

    void setDecreaseFactor(float m_decreaseFactor)
    {
        GrowingCellStructure::m_decreaseFactor = m_decreaseFactor;
    }

    void setAllowMiss(int m_allowMiss) { GrowingCellStructure::m_allowMiss = m_allowMiss; }

    void setCollapseThreshold(float m_collapseThreshold)
    {
        GrowingCellStructure::m_collapseThreshold = m_collapseThreshold;
    }

    void setFilterChain(bool m_filterChain) { GrowingCellStructure::m_filterChain = m_filterChain; }

    void setDeleteLongEdgesFactor(int m_deleteLongEdgesFactor)
    {
        GrowingCellStructure::m_deleteLongEdgesFactor = m_deleteLongEdgesFactor;
    }

    void setInterior(bool m_interior) { GrowingCellStructure::m_interior = m_interior; }

    void setNumBalances(int m_balances) { GrowingCellStructure::m_balances = m_balances; }

  private:
    PointsetSurfacePtr<BaseVecT>* m_surface; // helper-surface
    HalfEdgeMesh<BaseVecT>* m_mesh;

    // SHARED members
    int m_runtime;        // how many steps?
    int m_basicSteps;     // how many steps until collapse?
    int m_numSplits;      // how many splits
    float m_boxFactor;    // for initial mesh
    bool m_withCollapse;  // should we collapse?
    float m_learningRate; // learning rate of the algorithm
    float m_neighborLearningRate;
    bool m_filterChain; // should a filter chain be applied?
    bool m_interior;    // should the interior be reconstructed or the exterior?
    int m_balances;
    float m_avgSignalCounter = 0;

    // "GCS" related members
    TumbleTree* tumble_tree;
    DynamicKDTree<BaseVecT>* kd_tree;
    std::vector<Cell*> cellArr; // TODO: OUTSOURCE IT INTO THE TUMBLETREE CLASS, NEW PARAMETER FOR
                                // THE TUMBLE TREE CONSTRUCTOR
                                // CONTAINING THE MAXMIMUM SIZE OF THE MESH
    float m_decreaseFactor; // for sc calc
    int m_allowMiss;
    float m_collapseThreshold; // threshold for the collapse - when does it make sense
    int m_deleteLongEdgesFactor;
    int notFoundCounter = 0;
    int flipCounter = 0;

    // "GSS" related members
    bool m_useGSS = false;
    HashMap<FaceHandle, std::pair<float, float>>
        faceAgeErrorMap; // hashmap for mapping a FaceHandle to <age, error>
    float m_avgFaceSize = 0;
    float m_avgEdgeLength = 0;

    float m_limSkip;
    float m_limSingle;
    float m_maxAge;
    bool m_withRemove;

    // SHARED MEMBER FUNCTIONS

    void executeBasicStep(PacmanProgressBar& progress_bar);

    void executeVertexSplit();

    void executeEdgeCollapse();

    void getInitialMesh();

    BaseVecT getRandomPointFromPointcloud();

    VertexHandle getClosestPointInMesh(BaseVecT point, PacmanProgressBar& progress_bar);

    void initTestMesh(); // test

    // GCS MEMBER FUNCTIONS

    void performLaplacianSmoothing(VertexHandle vertexH, BaseVecT random, float factor = 0.01);

    void aggressiveCutOut(VertexHandle vH);

    double avgDistanceBetweenPointsInPointcloud();

    int numVertexValences(int minValence);

    std::pair<double, double> equilaterality();

    double avgValence();

    // void coalescing();

    // ADDITIONAL FUNCTIONS

    void removeWrongFaces();

    int cellVecSize();

    // TODO: add gss related functions
};
} // namespace lvr2

#include "lvr2/reconstruction/gs2/GrowingCellStructure.tcc"

#endif // LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP
