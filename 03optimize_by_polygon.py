from t2grids import *
from t2data import *
from t2incons import *
from t2listing import *
import time

start = time.time()

print ('3- Optimize grid')
#read large model grid
#this assumes 2 refinements - if only one then should read 2a
geo=mulgrid('02a_grid.dat')
#nodes to optimize:
opt=np.loadtxt('03polygon_optimize.dat',delimiter=',')
print (opt)
#need to find cols in polygon
opt_nodes=geo.nodes_in_polygon(opt)
print ('Nodes:', len(opt_nodes), opt_nodes)
#going from nodes to nodenames
opt_nodes_list=[tempnode.name for tempnode in opt_nodes]
geo.optimize(opt_nodes_list,1.0,0.5,0.0, pest=False)
#write geometry file. Note the filename is just for tracking 
#until I get it right.  Then I rename to 001_grid.dat
geo.write('03_grid.dat')

end = time.time()
print('Grid optimized in', (end-start), 's')