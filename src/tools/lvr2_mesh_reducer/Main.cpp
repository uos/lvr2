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


#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

#include <boost/optional.hpp>
#include "Options.hpp"



using std::unique_ptr;
using std::make_unique;

#include "lvr2/types/Model.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/geometry/PMPMesh.hpp"
#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/algorithm/ReductionAlgorithms.hpp"
#include "lvr2/util/Progress.hpp"
#include "lvr2/util/Logging.hpp"

using Vec = lvr2::BaseVector<float>;

template<typename VecT>
std::shared_ptr<lvr2::BaseMesh<VecT> > make_hem(
  lvr2::MeshBufferPtr mesh_buffer,
  std::string hem_impl)
{
    if(hem_impl == "lvr")
    {
        return std::make_shared<lvr2::HalfEdgeMesh<VecT> >(mesh_buffer);
    }
    if(hem_impl == "pmp")
    {
        return std::make_shared<lvr2::PMPMesh<VecT> >(mesh_buffer);
    }
    
    throw std::runtime_error("Could not find half edge mesh implementation.");
}

int main(int argc, char** argv)
{
    // =======================================================================
    // Parse and print command line parameters
    // =======================================================================
    // Parse command line arguments
    meshreduce::Options options(argc, argv);

    options.printLogo();


    std::cout << "LVR2 Mesh Reduction" << std::endl;
  
    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage())
    {
        return EXIT_SUCCESS;
    }
    std::cout << options << std::endl;
    std::cout << "Loading mesh ..." << endl;
    lvr2::ModelPtr model = lvr2::ModelFactory::readModel(options.getInputFileName());
    
    if(!model->m_mesh)
    {
      std::cout << "lvr could not load mesh / input file is not a mesh." << std::endl;
      return 0;
    }

    lvr2::MeshBufferPtr meshBuffer = model->m_mesh;
    cout << *meshBuffer << endl;
  
    auto mesh = make_hem<Vec>(meshBuffer, options.getHemImplementation());

    std::cout << lvr2::timestamp << "Computing face normals..." << std::endl;

    // Calculate initial face normals
    auto faceNormals = calcFaceNormals(*mesh);

    // Reduce mesh complexity
    const auto reductionRatio = options.getEdgeCollapseReductionRatio();
    std::cout << lvr2::timestamp << "Collapsing faces..." << std::endl;

    if (reductionRatio > 0.0)
    {
        if (reductionRatio > 1.0)
        {
            throw "The reduction ratio needs to be between 0 and 1!";
        }

        // Each edge collapse removes two faces in the general case.
        // TODO: maybe we should calculate this differently...
        const auto count = static_cast<size_t>((mesh->numFaces() / 2) * reductionRatio);
        auto collapsedCount = simpleMeshReduction(*mesh, count, faceNormals);
    }

    // =======================================================================
    // Finalize mesh
    // =======================================================================
    lvr2::SimpleFinalizer<Vec> finalizer;
    // Run finalize algorithm
    auto buffer = finalizer.apply(*mesh);

    std::cout << "Result: " << std::endl;
    std::cout << *buffer << std::endl;

    // =======================================================================
    // Write all results (including the mesh) to file
    // =======================================================================
    // Create output model and save to file
    auto m = std::make_shared<lvr2::Model>(buffer);

    lvr2::ModelFactory::saveModel(m, "reduced_mesh.ply");
   
    cout << lvr2::timestamp << "Program end." << endl;

    return 0;
}
