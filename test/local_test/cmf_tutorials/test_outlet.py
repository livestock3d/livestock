import cmf
from numpy import transpose
from matplotlib import pyplot as plt

p = cmf.project()
# Create cell with 1000m2 and surface water storage
c = p.NewCell(0,0,1,1000,True)
# Set puddle depth to 2mm
c.surfacewater.puddledepth = 0.002
# Add a thick layer, low conductivity. Use Green-Ampt-Infiltration
c.add_layer(0.1, cmf.VanGenuchtenMualem(Ksat=0.1))
c.install_connection(cmf.GreenAmptInfiltration)
# Create outlet, 10 m from cell center, 1m below cell
outlet = p.NewOutlet('outlet',10,0,0)
# Create connection, distance is calculated from position
con = cmf.KinematicSurfaceRunoff(c.surfacewater,outlet,flowwidth=10)
# set rainfall, a good shower to get surface runoff for sure (100mm/day)
c.set_rainfall(100.)
# Create a solver
solver = cmf.CVodeIntegrator(p,1e-8)

# Calculate results
Vsoil, Vsurf, qsurf,qinf = transpose([(c.layers[0].volume, c.surfacewater.volume, outlet(t), c.layers[0](t))
                             for t in solver.run(cmf.Time(1,1,2012),cmf.Time(2,1,2012),cmf.min)])
# Present results
ax1=plt.subplot(211)
plt.plot(Vsurf,label='Surface')
plt.plot(Vsoil,label='Soil')
plt.ylabel('Water content in mm')
plt.legend(loc=0)
plt.subplot(212,sharex=ax1)
plt.plot(qsurf,label='Surface')
plt.plot(qinf,label='Soil')
plt.ylabel('Flux in mm/day')
plt.xlabel('Time in minutes')
plt.legend(loc=0)
plt.show()