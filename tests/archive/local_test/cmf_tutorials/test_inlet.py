import cmf
from numpy import transpose
from matplotlib import pyplot as plt
import datetime

p = cmf.project()
# Create cell with 1000m2 and surface water storage
c = p.NewCell(0,0,1,1000,True)
# Set puddle depth to 2mm
c.surfacewater.puddledepth = 0.002
# Add a thick layer, low conductivity. Use Green-Ampt-Infiltration
c.add_layer(0.1, cmf.VanGenuchtenMualem(Ksat=0.1))
c.install_connection(cmf.GreenAmptInfiltration)

# Create a Neumann Boundary condition connected to W1
In = cmf.NeumannBoundary.create(c.surfacewater)
# Create a timeseries with daily alternating values.
In.flux = 5


# Create a solver
solver = cmf.CVodeIntegrator(p, 1e-8)

# Calculate results
Vsoil, Vsurf, = transpose([(c.layers[0].volume, c.surfacewater.volume)
                             for t in solver.run(cmf.Time(1,1,2012),cmf.Time(2,1,2012),cmf.min)])

# Present results
plt.figure()
plt.plot(Vsurf,label='Surface')
plt.plot(Vsoil,label='Soil')
plt.ylabel('Water content in mm')
plt.legend(loc=0)
plt.show()