# Handling Scan Projects with LVR2

This folder contains examples handling different types of scan project structes for a given purpose.

## Scan Projects
Scan projects are collections of sensory data collected by different sensors at certain scan positions. Internally, entities of a scan project are managed hierarchichally as follows:

1. Scan projects
2. Scan positions
3. Sensors
4. Sensor data

## Meta Information

Each of the entities above gets its own meta information.
Given a specialization of an entity ("type") this information is 
extended.

### Scan projects
A scan project is a container for scan positions. The meta information
consists of 

- Coordinate Reference System (crs)
- Transformation
- Pose Estimation

### Sensors
.... TODO

