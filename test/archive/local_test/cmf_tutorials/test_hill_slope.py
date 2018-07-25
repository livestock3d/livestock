import cmf
import numpy as np
from matplotlib.animation import FuncAnimation
import datetime
# Import the hill_plot
from cmf.draw import hill_plot
# Import some matplotlib stuff
from matplotlib.pylab import figure,show,cm


def z(x):
    return 10/(1+np.exp((x-100)/30))
# create a project
p = cmf.project()

for i in range(20):
    x = i * 10.
    # create a cell with surface storage
    c = p.NewCell(x,0,z(x),100,True)

for c_upper, c_lower in zip(p[:-1], p[1:]):
    c_upper.topology.AddNeighbor(c_lower, 10.)

# Customize cells
for c in p:
    # create layers
    for d in [0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.25,1.5,1.75,2.]:
        rc = cmf.VanGenuchtenMualem(Ksat=50*np.exp(-d),alpha=0.1,n=2.0,phi=0.5)
        rc.w0 = 0.9996
        c.add_layer(d,rc)
    # set initial conditions
    c.saturated_depth=2.
    # use Richards connection
    c.install_connection(cmf.Richards)
    c.install_connection(cmf.GreenAmptInfiltration)
    # Add more connections here... (eg. ET, snowmelt, canopy overflow)

cmf.connect_cells_with_flux(p,cmf.Darcy)
cmf.connect_cells_with_flux(p,cmf.KinematicSurfaceRunoff)

for c in p:
    c.set_rainfall(10.)

outlet = p.NewOutlet('outlet',200,0,0)
for l in p[-1].layers:
    # create a Darcy connection with 10m flow width between each soil layer
    # and the outlet
    cmf.Darcy(l,outlet,FlowWidth=10.)
cmf.KinematicSurfaceRunoff(p[-1].surfacewater,outlet,10.)

solver = cmf.CVodeIntegrator(p,1e-9)
solver.t = datetime.datetime(2012,1,1)



# Create a new matplotlib figure fig
fig = figure(figsize=(16,9))
# Create a subplot with a light grey background
ax = fig.add_subplot(111,axisbg='0.8')
# Create the hillplot for water filled pore space with using a yellow to green to blue colormap
image = hill_plot(p,solver.t)
# Set the scale for the arrows. This value will change from model to model.
image.scale = 100.
# white arrows are nicer to see
image.q_sub.set_facecolor('w')

def run(frame):
    # Run model for one day
    t = solver(cmf.day)
    # Update image
    image(t)


animation = FuncAnimation(fig,run,repeat=False,
                          frames=365)
show()