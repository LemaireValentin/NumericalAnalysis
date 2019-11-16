import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import gmsh
import sys

from mysolve import *

DEBUG = True

# This scripts assembles and solves a simple finite element problem
# using exclusively the python api of Gmsh.

cm = 0.01

# Homework model parameters
ref = 1       # mesh refinement factor (1 coarse - 5 fine)
gap = 0.2*cm  # core-plate distance, air gap, entrefer
freq = 50     # working frequency
vel = 100      # plate velocity
mur = 100.    # Relative magnetic permeability of region CORE


# geometrical parameters
L1 = 15*cm # box x-length
L2 =  1*cm # plate thickness
L3 =  8*cm # box y-length
L4  = 2*cm # core thickness

# Physical regions
DIRICHLET0 = 11 # Physical Line tag of a=0 boundary condition
PLATE = 1
AIR = 2
CORE = 3        # Physical Surface tag of magnetic core
COILN = 4       # Physical Surface tag of positive current coil
COILP = 5       # Physical Surface tag of negative current coil

# Model parameters
mu0 = 4.e-7*np.pi
sigma = 5e7     # electric conductivity
J = 1.e7        # Current density (A/m^2)
jomega = complex(0, 2*np.pi*freq)
CoilSection = L4*L4
Integration = 'Gauss2'


def create_geometry(gap, ref):
    model.add("ndt")
    lc1=L1/10/ref;
    lc2=L4/8/ref;

    L0X = 2*cm;
    L0Y = L2 + gap ;

    factory.addPoint(  0      ,  0, 0, lc1,  1)
    factory.addPoint( L1      ,  0, 0, lc1,  2)
    factory.addPoint(  0      , L2, 0, lc1,  3)
    factory.addPoint( L0X     , L2, 0, lc2,  4)
    factory.addPoint( L0X+1*L4, L2, 0, lc2,  5)
    factory.addPoint( L0X+2*L4, L2, 0, lc2,  6)
    factory.addPoint( L0X+3*L4, L2, 0, lc2,  7)
    factory.addPoint( L0X+4*L4, L2, 0, lc2,  8)
    factory.addPoint( L1      , L2, 0, lc1,  9)
    factory.addPoint(  0      , L3, 0, lc1, 10)
    factory.addPoint( L1      , L3, 0, lc1, 11)
    
    factory.addLine(1,2, 1)
    factory.addLine(2,9, 2)
    factory.addLine(9,8, 3)
    factory.addLine(8,7, 4)
    factory.addLine(7,6, 5)
    factory.addLine(6,5, 6)
    factory.addLine(5,4, 7)
    factory.addLine(4,3, 8)
    factory.addLine(3,1, 9)
    factory.addCurveLoop([1,2,3,4,5,6,7,8,9], 1)

    factory.addLine(9,11, 10)
    factory.addLine(11,10, 11)
    factory.addLine(10,3, 12)
    factory.addCurveLoop([10,11,12,-8,-7,-6,-5,-4,-3], 2)


    factory.addPoint(L0X     , L0Y, 0, lc2, 20)
    factory.addPoint(L0X+1*L4, L0Y, 0, lc2, 21)
    factory.addPoint(L0X+2*L4, L0Y, 0, lc2, 22)
    factory.addPoint(L0X+3*L4, L0Y, 0, lc2, 23)
    factory.addPoint(L0X+4*L4, L0Y, 0, lc2, 24)

    factory.addPoint(L0X     , L0Y+L4, 0, lc1, 25)
    factory.addPoint(L0X+1*L4, L0Y+L4, 0, lc1, 26)
    factory.addPoint(L0X+2*L4, L0Y+L4, 0, lc1, 27)
    factory.addPoint(L0X+3*L4, L0Y+L4, 0, lc1, 28)

    factory.addPoint(L0X+1*L4, L0Y+2*L4, 0, lc1, 29)
    factory.addPoint(L0X+4*L4, L0Y+2*L4, 0, lc1, 30)

    # core
    factory.addLine(21,22, 20)
    factory.addLine(22,27, 21)
    factory.addLine(27,28, 22)
    factory.addLine(28,23, 23)
    factory.addLine(23,24, 24)
    factory.addLine(24,30, 25)
    factory.addLine(30,29, 26)
    factory.addLine(29,26, 27)
    factory.addLine(26,21, 28)
    factory.addCurveLoop([20,21,22,23,24,25,26,27,28], 3)

    # left coil region
    factory.addLine(26,25, 29)
    factory.addLine(25,20, 30)
    factory.addLine(20,21, 31)
    factory.addCurveLoop([29,30,31,-28], 4)
    
    # right coil region
    factory.addLine(22,23, 32)
    factory.addCurveLoop([-23,-22,-21,32], 5)

    factory.addPlaneSurface([1], 1)
    factory.addPlaneSurface([2,3,4,5], 2)
    factory.addPlaneSurface([3], 3)
    factory.addPlaneSurface([4], 4)
    factory.addPlaneSurface([5], 5)

    factory.synchronize()

    model.addPhysicalGroup(2, [1], 1)
    model.addPhysicalGroup(2, [2], 2)
    model.addPhysicalGroup(2, [3], 3)
    model.addPhysicalGroup(2, [4], 4)
    model.addPhysicalGroup(2, [5], 5)
    model.addPhysicalGroup(1, [1,2,10,11,12,9], 11)

    model.setPhysicalName(2, 1, 'PLATE')
    model.setPhysicalName(2, 2, 'AIR')
    model.setPhysicalName(2, 3, 'CORE')
    model.setPhysicalName(2, 4, 'COILN')
    model.setPhysicalName(2, 5, 'COILP')
    model.setPhysicalName(1, 11, 'DIR')
    return

import sys
PY3 = sys.version_info.major == 3
def printf(*args):
    if not DEBUG: return
    if PY3:
        exec("print( *args )")
    else:
        for item in args: exec("print item,")
        exec("print")

def errorf(*args):
    if PY3:
        exec("print( *args )")
    else:
        for item in args: exec("print item,")
        exec("print")
    exit(1)
    
def solve(freq, vel, mur):
    jomega = complex(0, 2 * np.pi * freq)
    mshNodes = np.array(model.mesh.getNodes()[0])
    numMeshNodes = len(mshNodes)
    printf('numMeshNodes =', numMeshNodes)
    maxNodeTag = int(np.amax(mshNodes))
    printf('maxNodeTag =', maxNodeTag)
    
    
    # initializations of global assembly arrays iteratively filled-in during assembly
    matrowflat = np.array([], dtype=np.int32)
    matcolflat = np.array([], dtype=np.int32)
    matflat = np.array([], dtype=np.int32)
    rhsrowflat = np.array([], dtype=np.int32)
    rhsflat = np.array([], dtype=np.int32)

    # typNodes[tag] = {0,1,2} 0: does not exist, 1:internal node, 2:boundary node
    # Existing node tags are set to 1 here.
    # Boundary node tag are identified later.
    typNodes = np.zeros(maxNodeTag+1, dtype=np.int32)
    for tagNode in mshNodes:
        typNodes[tagNode] = 1

    # The mesh is iterated over, looping successively (nested loops) over:
    # Physical groups/Geometrical entities/Element types/Elements
    vGroups = model.getPhysicalGroups()
    for iGroup in vGroups:
        dimGroup = iGroup[0] # 1D, 2D or 3D
        tagGroup = iGroup[1] # the word 'tag' is systematically used instead of 'number'
        vEntities = model.getEntitiesForPhysicalGroup(dimGroup, tagGroup)
        for tagEntity in vEntities:
            dimEntity = dimGroup # FIXME dimEntity should be optional when tagEntity given.
            vElementTypes = model.mesh.getElementTypes(dimEntity,tagEntity)
            for elementType in vElementTypes:
                vTags, vNodes = model.mesh.getElementsByType(elementType, tagEntity)
                numElements = len(vTags)
                numGroupNodes = len(vNodes)
                enode = np.array(vNodes).reshape((numElements,-1))
                numElementNodes = enode.shape[1]
                printf('\nIn group', tagGroup, ', numElements = e =', numElements)
                printf('numGroupNodes =', numGroupNodes,', numElementNodes = n =', numElementNodes)
                printf('%enodes (e,n) =', enode.shape)

                # Assembly of stiffness matrix for all 2 dimensional elements
                # (i.e., triangles or quadrangles)
                if dimEntity==2 :

                    uvw,weights = gmsh.model.mesh.getIntegrationPoints(2,"Gauss2")
                    numcomp, sf = model.mesh.getBasisFunctions(elementType, uvw, 'Lagrange')
                    
                    numGaussPoints = weights.shape[0]
                    printf('numGaussPoints = g =', numGaussPoints, ', %weights (g) =', weights.shape)
                    sf = np.array(sf).reshape((numGaussPoints,-1))
                    printf('%sf (g,n) =', sf.shape)
                    if sf.shape[1] != numElementNodes:
                        errorf('Something went wrong')
                    numcomp, dsfdu = model.mesh.getBasisFunctions(elementType, uvw, 'GradLagrange')
                    
                    #remove useless dsfdw
                    dsfdu = np.array(dsfdu).reshape((numGaussPoints,numElementNodes,3))[:,:,:-1]
                    printf('%dsfdu (g,n,u) =', dsfdu.shape)
                    
                    qjac, qdet, qpoint = model.mesh.getJacobians(elementType, uvw, tagEntity)
                    printf('Gauss integr:',len(qjac),len(qdet),len(qpoint),
                           '= (9, 1, 3) x',numGaussPoints,'x',numElements)
                    qdet = np.array(qdet).reshape((numElements,numGaussPoints))
                    printf('%qdet (e,g) =', qdet.shape)
                    #remove components of dxdu useless in dimEntity dimensions (here 2D)
                    dxdu = np.array(qjac).reshape((numElements,numGaussPoints,3,3))[:,:,:-1,:-1]
                    # jacobien stored by row, so dxdu[i][j] = dxdu_ij = dxi/duj 
                    printf('%dxdu (e,g,x,u)=', dxdu.shape)

                    # material characteristic 
                    if tagGroup == CORE:
                        nu = complex(1.,0)/(mur*mu0)
                    else:
                        nu = complex(1.,0)/mu0
                        
                    # dsdfx = dudx * dsfdu
                    dudx = np.linalg.inv(dxdu) # dudx[j][k] = dudx_jk = duj/dxk
                    printf('%dudx (e,g,u,x) =', dudx.shape)
                    dsfdx  = np.einsum("egxu,gnu->egnx",dudx,dsfdu); # sum over u = dot product
                    printf('%dsfdx (e,g,n,x) =', dsfdx.shape)

                    # performs the Gauss integration with einsum
                    localmat = nu * np.einsum("egik,egjk,eg,g->eij", dsfdx, dsfdx, qdet, weights) 
                    printf('%localmat (e,n,n) =', localmat.shape)

                    if tagGroup == PLATE:
                        localmat += sigma*jomega*np.einsum("gi,gj,eg,g->eij", sf, sf, qdet, weights)
                        Liesf = np.einsum("egik,k->egi", dsfdx, np.array([vel,0]))
                        localmat += sigma*np.einsum("gi,egj,eg,g->eij", sf, Liesf, qdet, weights)

                    # The next two lines are rather obscure.
                    # See explanations at the bottom of the file. 
                    matcol = np.repeat(enode[:,:,None],numElementNodes,axis=2)
                    matrow = np.repeat(enode[:,None,:],numElementNodes,axis=1)
                    
                    matcolflat = np.append(matcolflat, matcol.flatten())
                    matrowflat = np.append(matrowflat, matrow.flatten())
                    matflat = np.append(matflat, localmat.flatten())

                    if tagGroup == COILP or tagGroup == COILN:
                        if tagGroup == COILP:
                            load = J
                        elif tagGroup == COILN:
                            load = -J
                        localrhs = load * np.einsum("gn,eg,g->en", sf, qdet, weights)
                        printf('Check rhs:', np.sum(localrhs), "=", load*CoilSection)
                        rhsrowflat = np.append(rhsrowflat, enode.flatten())
                        rhsflat = np.append(rhsflat, localrhs.flatten())

                # identify boundary node
                if tagGroup == DIRICHLET0:
                    for tagNode in vNodes:
                        typNodes[tagNode] = 2

    printf('\nDimension of arrays built by the assembly process')
    printf('%colflat = ', matcolflat.shape)
    printf('%rowflat = ', matrowflat.shape)
    printf('%localmatflat = ', matflat.shape)
    printf('%rhsrowflat = ', rhsrowflat.shape)
    printf('%rhsflat = ', rhsflat.shape)

    # Associate to all mesh nodes a line number in the system matrix
    # reserving top lines for internal nodes and bottom lines for fixed nodes (boundary nodes).
    node2unknown = np.zeros(maxNodeTag+1, dtype=np.int32)
    index = 0
    for tagNode,typ in enumerate(typNodes):
        if  typ == 1: # not fixed
            index += 1
            node2unknown[tagNode] = index
    numUnknowns = index
    printf('numUnknowns =', numUnknowns)
    for tagNode,typ in enumerate(typNodes):
        if  typ == 2: # fixed
            index += 1
            node2unknown[tagNode] = index

    if index != numMeshNodes:
        errorf('Something went wrong')

    unknown2node = np.zeros(numMeshNodes+1, dtype=np.int32)
    for node, unkn in enumerate(node2unknown):
        unknown2node[unkn] = node

    printf('\nDimension of nodes vs unknowns arrays')
    printf('%mshNodes=',mshNodes.shape)
    printf('%typNodes=',typNodes.shape)
    printf('%node2unknown=',node2unknown.shape)
    printf('%unknown2node=',unknown2node.shape)

    # Generate system matrix A=globalmat and right hand side b=globalrhs

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    # 'node2unknown-1' are because python numbers rows and columns from 0
    globalmat = scipy.sparse.coo_matrix(
        (matflat, (node2unknown[matcolflat.astype(int)]-1, node2unknown[matrowflat.astype(int)]-1) ),
        shape=(numMeshNodes, numMeshNodes)).todense()
    
    globalrhs = np.zeros(numMeshNodes)
    for index,node in enumerate(rhsrowflat):
        globalrhs[node2unknown[int(node)]-1] += rhsflat[int(index)]

    printf('%globalmat =', globalmat.shape, ' %globalrhs =', globalrhs.shape)

    A = globalmat[:numUnknowns,:numUnknowns]
    b = globalrhs[:numUnknowns]
    success, sol = mysolve(A, b)
    if not success:
        errorf('Solver not implemented yet')
    sol = np.append(sol,np.zeros(numMeshNodes-numUnknowns))
    printf('%sol =', sol.shape)
    
    # Export solution
    sview = gmsh.view.add("solution")
    gmsh.view.addModelData(sview,0,"","NodeData",unknown2node[1:],sol[:,None])
    # gmsh.view.write(sview,"a.pos")
    printf('Flux (computed) =', np.max(sol)-np.min(sol))

    return A, b, numMeshNodes, sol


def ndtfun(gap, ref, freq, vel, mur):
    model = gmsh.model
    factory = model.geo
    gmsh.initialize(sys.argv)

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", ref)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("View[0].IntervalsType", 3)
    gmsh.option.setNumber("View[0].NbIso", 20)

    create_geometry(gap, ref)
    model.mesh.generate(2)
    A, b, num_nodes, sol = solve(freq, vel, mur)

    return A, b, num_nodes, sol, np.linalg.cond(A)

#
model = gmsh.model
factory = model.geo
gmsh.initialize(sys.argv)
#
#gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", ref)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("View[0].IntervalsType", 3)
gmsh.option.setNumber("View[0].NbIso", 20)
#
create_geometry(gap, ref)
model.mesh.generate(2)
solve(freq, vel, mur)
#
gmsh.write('ndt.msh')
#
gmsh.fltk.run()



# Solve linear system Ax=b

## Explanation for the serialization of elementary matrices.
##
## node = numpy.array([[1,2,3]]) # one element with nodes 1,2 and 3
## col = numpy.repeat(enode[:,:,None],3,axis=2)
## row = numpy.repeat(enode[:,None,:],3,axis=1)
## >>> col
## array([[[1, 1, 1],
##         [2, 2, 2],
##         [3, 3, 3]]])
## >>> row
## array([[[1, 2, 3],
##         [1, 2, 3],
##         [1, 2, 3]]])
