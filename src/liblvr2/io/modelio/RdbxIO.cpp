//
// Created by Nikolas on 13.09.22.
//

#include "lvr2/io/modelio/RdbxIO.hpp"


#include <riegl/rdb.hpp>
#include <riegl/rdb/default.hpp>


#include <iomanip>
#include <array>
#include <vector>
#include <cstdint>
#include <iostream>
#include <exception>


namespace lvr2
{
    RdbxIO::RdbxIO()
    {

    }
    RdbxIO::~RdbxIO()
    {

    }

    /**
     * @brief Creates a new rdb database and
     * write its content into @filename, currently only
     * x,y,z and reflectance
     *
     * @param filename
     */
    void RdbxIO::save(string filename)
    {
        /** // checks for validity
        if (!m_model)
        {
            std::cerr << "No model set for export!" << std::endl;
            return;
        }

        if ( !this->m_model->m_pointCloud ) {
            std::cerr << "No point buffer available for output." << std::endl;
            return;
        }

        floatArr m_points;
        floatArr m_pointReflectance;
        size_t m_numPoints = 0;

        // Get buffers
        if ( m_model->m_pointCloud )
        {
            PointBufferPtr pointBuffer = m_model->m_pointCloud;
        }

        //RDB code to read the data
        try
        {
            // New RDB library context
            riegl::rdb::Context context;

            // New database instance
            riegl::rdb::Pointcloud rdb(context);

            // Step 1: Create new point cloud database
            {
                // This object contains all settings that are required
                // to create a new RDB point cloud database.
                riegl::rdb::pointcloud::CreateSettings settings;

                // Define primary point attribute, usually the point coordinates
                // details see class riegl::rdb::pointcloud::PointAttribute
                settings.primaryAttribute.name         = "riegl.xyz";
                settings.primaryAttribute.title        = "XYZ";
                settings.primaryAttribute.description  = "Cartesian point coordinates";
                settings.primaryAttribute.unitSymbol   = "m";
                settings.primaryAttribute.length       =  3;
                settings.primaryAttribute.resolution   =  0.00025;
                settings.primaryAttribute.minimumValue = -535000.0; // minimum,
                settings.primaryAttribute.maximumValue = +535000.0; //   maximum and
                settings.primaryAttribute.defaultValue =       0.0; //     default in m
                settings.primaryAttribute.storageClass = riegl::rdb::pointcloud::PointAttribute::VARIABLE;

                // Define database settings
                settings.chunkMode = riegl::rdb::pointcloud::CreateSettings::POINT_COUNT;
                settings.chunkSize = 50000; // maximum number of points per chunk
                settings.compressionLevel = 10; // 10% compression rate

                // Finally create new database
                rdb.create(filename , settings);
            }

            // Step 2: Define some additional point attributes
            //         Please note that there is also a shortcut for built-in
            //         RIEGL default point attributes which we use to define
            //         the "riegl.class" attribute at the end of this block.
            {
                // Before we can modify the database, we must start a transaction
                riegl::rdb::pointcloud::TransactionScope transaction(rdb,
                       "Initialization",      // transaction title
                       "Save rdbx" // software name
                );

                // Target surface reflectance
                {
                    riegl::rdb::pointcloud::PointAttribute attribute;
                    //
                    attribute.name         = "riegl.reflectance";
                    attribute.title        = "Reflectance";
                    attribute.description  = "Target surface reflectance";
                    attribute.unitSymbol   = "dB";
                    attribute.length       =  1;
                    attribute.resolution   =    0.010;
                    attribute.minimumValue = -100.000; // minimum,
                    attribute.maximumValue = +100.000; //   maximum and
                    attribute.defaultValue =    0.000; //     default in dB
                    attribute.storageClass = riegl::rdb::pointcloud::PointAttribute::CONSTANT;
                    //
                    rdb.pointAttribute().add(attribute);
                }
                /** // Point color
                {
                    riegl::rdb::pointcloud::PointAttribute attribute;
                    //
                    attribute.name         = "riegl.rgba";
                    attribute.title        = "True Color";
                    attribute.description  = "Point color acquired by camera";
                    attribute.unitSymbol   = ""; // has no unit
                    attribute.length       =   4;
                    attribute.resolution   =   1.000;
                    attribute.minimumValue =   0.000;
                    attribute.maximumValue = 255.000;
                    attribute.defaultValue = 255.000;
                    attribute.storageClass = riegl::rdb::pointcloud::PointAttribute::VARIABLE;
                    //
                    rdb.pointAttribute().add(attribute);
                }
                // Point classification - by using a shortcut for built-in RIEGL attributes:
                {
                    rdb.pointAttribute().add("riegl.class");
                }
                // Echo amplitude - by using the constant from "riegl/rdb/default.hpp"
                {
                    rdb.pointAttribute().add(riegl::rdb::pointcloud::RDB_RIEGL_AMPLITUDE);
                }**/

                /** // Finally commit transaction
                transaction.commit();
            }
        }
        catch(const riegl::rdb::Error &error)
        {
            std::cerr << error.what() << " (" << error.details() << ")" << std::endl;
        }
        catch(const std::exception &error)
        {
            std::cerr << error.what() << std::endl;
        }

        try
        {
            // New RDB library context
            riegl::rdb::Context context;
            riegl::rdb::Pointcloud rdb(context);
            riegl::rdb::pointcloud::OpenSettings settings;
            rdb.open(filename, settings);

            // Query some attribute details
            using riegl::rdb::pointcloud::PointAttribute;
            const PointAttribute detailsCoordinates = rdb.pointAttribute().get("riegl.xyz");
            const PointAttribute detailsReflectance = rdb.pointAttribute().get("riegl.reflectance");

            // Before we can modify the database, we must start a transaction
            riegl::rdb::pointcloud::TransactionScope transaction(
                    rdb,                  // point cloud object
                    "Import",             // transaction title
                    "Point Importer v1.0" // software name
            );

            // Get buffers
            if ( m_model->m_pointCloud )
            {
                PointBufferPtr pc( m_model->m_pointCloud );

                m_numPoints = pc->numPoints();

                m_points    = pc->getPointArray();
            }

            const uint32_t BUFFER_SIZE =  10000; // point block/chunk size
            const uint32_t POINT_COUNT = m_numPoints; // total number of points

            std::vector< std::array<double, 3> > bufferCoordinates(BUFFER_SIZE);
            std::vector< float >                 bufferReflectance(BUFFER_SIZE);

            // Start new insert query
            riegl::rdb::pointcloud::QueryInsert query = rdb.insert();

            // Tell insert query where to read the "reflectance" values from
            /**if (0) // you can either specify the name of the point attribute...
            {
                query.bindBuffer(
                        "riegl.reflectance",
                        bufferReflectance
                );
            }
            else // ...or use one of the constants from "riegl/rdb/default.hpp":
            {
                query.bindBuffer(
                        riegl::rdb::pointcloud::RDB_RIEGL_REFLECTANCE,
                        bufferReflectance
                );
            }**/

            /** // Tell insert query where to read the point coordinates from
            if (0) // buffers for vectors can be bound for all vector elements at once...
            {
                query.bindBuffer("riegl.xyz", bufferCoordinates);
            }
            else // ...or for each vector element separately:
            {
                // The 'stride' defines the number of bytes between the buffer
                // locations of the attribute values of two consecutive points.
                const int32_t stride = sizeof(bufferCoordinates[0]);
                query.bindBuffer("riegl.xyz[0]", bufferCoordinates[0][0], stride);
                query.bindBuffer("riegl.xyz[1]", bufferCoordinates[0][1], stride);
                query.bindBuffer("riegl.xyz[2]", bufferCoordinates[0][2], stride);
            }
            // Insert points block-wise
            for (uint32_t total = 0; total < POINT_COUNT;)
            {
                // Fill buffers with some random data
                //TODO: reflectance
                for (uint32_t i = 0; i < BUFFER_SIZE; i++)
                {
                    size_t pos = i * 3;
                    bufferCoordinates[i][0] = m_points[pos];
                    bufferCoordinates[i][1] = m_points[pos + 1];
                    bufferCoordinates[i][2] = m_points[pos + 2];
                    //bufferReflectance[i]    = randomReflectance(rng);
                }

                // Actually insert points
                total += query.next(BUFFER_SIZE);
                static uint32_t block = 1;
                //std::cout << "block: " << block++ << ", "
                //          << "total: " << total << std::endl;
            }

            // Finally commit transaction
            transaction.commit();
            // Success
        }
        catch(const riegl::rdb::Error &error)
        {
            std::cerr << error.what() << " (" << error.details() << ")" << std::endl;
        }
        catch(const std::exception &error)
        {
            std::cerr << error.what() << std::endl;
        }**/
    }


    void RdbxIO::save(ModelPtr model, std::string filename)
    {
        m_model = model;
        save(filename);
    }

    ModelPtr RdbxIO::read(string filename)
    {
        ModelPtr model (new Model);
        PointBufferPtr pointBuffer(new PointBuffer);

        try {
            // New RDB library context
            riegl::rdb::Context context;

            // Access existing database
            riegl::rdb::Pointcloud rdb(context);
            riegl::rdb::pointcloud::OpenSettings settings;
            rdb.open(filename, settings);

            // Prepare point attribute buffers
            static const uint32_t BUFFER_SIZE = 10000;
            std::vector<std::array<double, 3> > bufferCoordinates(BUFFER_SIZE);
            std::vector<float> bufferReflectance(BUFFER_SIZE);

            // Select query with empty filter to get all Points, only used to count all Points
            riegl::rdb::pointcloud::QuerySelect countselect = rdb.select("");

            // Second select query with empty filter to get all Points, used to fill Buffers
            riegl::rdb::pointcloud::QuerySelect select = rdb.select("");

            // Get index graph root node
            riegl::rdb::pointcloud::QueryStat stat = rdb.stat();
            riegl::rdb::pointcloud::GraphNode root = stat.index();

            // Get total number of Points
            uint32_t numPoints = root.pointCountTotal ;


            // Binding Buffers to the select query, so they get filled on select.next()
            using namespace riegl::rdb::pointcloud;
            select.bindBuffer(RDB_RIEGL_XYZ,         bufferCoordinates);
            select.bindBuffer(RDB_RIEGL_REFLECTANCE, bufferReflectance);

            // arrays to store point coordinates and their reflectance
            float *pointArray = new float[3 * numPoints];
            float *intensitiesArray = new float[numPoints];

            // variable to keep track of current point,
            // since i in following for loop is reset after BUFFER_SIZE=10000 steps
            uint32_t currPoint = 0;

            // calling select.next() to fill Buffers, and then store all loaded points in our arrays
            // 10.000 Points are loaded at the same time, we decided on this number because more will often cause problems while execution
            while (const uint32_t count = select.next(BUFFER_SIZE)) {
                for (uint32_t i = 0; i < count; i++) {
                    // filling arrays with data from rdbx file
                    pointArray[3 * currPoint] = bufferCoordinates[i][0];
                    pointArray[3 * currPoint + 1] = bufferCoordinates[i][1];
                    pointArray[3 * currPoint + 2] = bufferCoordinates[i][2];
                    intensitiesArray[currPoint] = bufferReflectance[i];
                    ++currPoint;
                }
            }

            select.close();

            // parsing float Arrays to floatArr
            floatArr parr(pointArray);
            floatArr iarr(intensitiesArray);

            // filling PointBuffer with data
            pointBuffer->setPointArray(parr, numPoints);    //versuchen die Anzahl punkte aus dem Pointbuffer zu lesen

            // passing reflectance as intensities
            pointBuffer->addFloatChannel(iarr, "intensities", numPoints, 1);

            // adding pointBuffer to model
            model->m_pointCloud = pointBuffer;
            return model;
        }

        catch(const riegl::rdb::Error &error)
        {
            std::cerr << error.what() << " (" << error.details() << ")" << std::endl;
            std::exit(-1); // error
        }
        catch(const std::exception &error)
        {

            std::cerr << error.what() << std::endl;
            std::exit(-1); // error
        }

    }

} // lvr2