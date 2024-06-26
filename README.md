This repository contains source files (Python scripts) for solving elasticity problems using FEniCS library.

Scripts are based on the *legacy* FEniCS version. The input meshes are generated using **GMSH** and converted to **XML** format using `dolfin-convert` command.

Some of the example problems solved are shown below.

## Linear and Hyperelasticity
### Linear elasticity
#### Cook's membrane in plane-strain condition
<img src="./LinearAndHyperelastic/Cooksmembrane2D-LE/Cooksmembrane2d-LE-nelem8-dispY.png" alt="Y-displacement" width="250"/>
<img src="./LinearAndHyperelastic/Cooksmembrane2D-LE/Cooksmembrane2d-LE-nelem8-pressure.png" alt="Pressure" width="250"/>


### Hyperelasticity

#### Cook's membrane in plane-strain condition
<img src="./LinearAndHyperelastic/Cooksmembrane2D-NH/Cooksmembrane2d-NH-nelem8-dispY.png" alt="Y-displacement" width="250"/>
<img src="./LinearAndHyperelastic/Cooksmembrane2D-NH/Cooksmembrane2d-NH-nelem8-pressure.png" alt="Pressure" width="250"/>


#### Block 3D
<img src="./LinearAndHyperelastic/block3d/block3d-nelem4-dispZ.png" alt="Y-displacement" width="250"/>
<img src="./LinearAndHyperelastic/block3d/block3d-nelem4-pressure.png" alt="Pressure" width="250"/>


## Morphoelasticity - Growth-driven deformations

#### Dilatation of a Cube
<img src="./Growth/cube/cube-growth-10times-dispX.png" alt="Growth Cube" width="250"/>

#### Deformation of rods
<img src="./Growth/rod/rod3d-circle.png" alt="Rod - Circle shape" width="230"/>
<img src="./Growth/rod/rod3d-spiral.png" alt="Rod - Spiral shape" width="230"/>
<img src="./Growth/rod/rod3d-helical1.png" alt="Rod - Helical shape 1" width="230"/>

## Magnetomechanics
