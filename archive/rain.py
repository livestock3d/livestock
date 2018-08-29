__author__ = "Christian Kongsgaard"
__license__ = "MIT"
__version__ = "0.0.1"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import threading
import queue
import pymesh as pm
import numpy as np

# Livestock imports
import livestock_linux.geometry as ls_geo

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Rain and Flow Library


def drain_mesh_paths(files_path):
    """ Estimates the trail of a drainage path on a mesh. """

    # Get files
    mesh_path = files_path + '/drain_mesh.obj'
    cpus = open(files_path + '/cpu.txt', 'r').readline()

    # Load mesh
    mesh = pm.load_mesh(mesh_path)
    mesh.enable_connectivity()

    # Result list
    drain_points = []
    drain_faces = []

    # Initialize mesh data
    mesh.add_attribute('face_centroid')
    mesh.add_attribute('face_index')
    start_pts = mesh.get_attribute('face_centroid')
    center_z = []
    face_index = mesh.get_attribute('face_index')
    faces = mesh.faces
    vertices = mesh.vertices
    face_destination = []
    ray_points = []

    # Construct start point list
    start_points = []
    i = 0
    while i < len(start_pts):
        for j in range(0, len(face_index)):
            start_points.append([face_index[j], np.array([start_pts[i], start_pts[i+1], start_pts[i+2]])])
            center_z.append(start_pts[i + 2])
            i += 3

    # Helper functions
    def face_vertices(face_index_):
        face = faces[int(face_index_)]
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        return v0, v1, v2

    def over_edge(point):
        """Handles when paths goes over the edge."""

        for k in range(0, len(face_index)):
            if center_z[k] >= point[2]:
                pass

            elif center_z[k] <= point[2]:
                # check to see if a similar point has already been processed

                for j_ in range(0, len(ray_points)):
                    if np.allclose(point, ray_points[j_]):
                        return face_destination[j_]

                # if not shoot ray
                v = face_vertices(k)
                intersect = ls_geo.ray_triangle_intersection(point, np.array([0, 0, -1]), v)

                if intersect[0]:
                    ray_points.append(point)
                    face_destination.append(k)
                    return k

                else:
                    pass

            else:
                print('Error in over_edge function!')
                print('centerZ:', center_z[k])
                print('point:', point)
                return None

    # Task function
    def drain_path():

        while 1:
            # Get job from queue
            job = q.get()
            index = job[0]
            pt = job[1]

            particles = []
            particles.append(pt)
            face_indices = []
            face_indices.append(int(index))
            run = True
            # print('index:',index)
            # print('point:',pt)

            while run:
                # Get adjacent faces
                adjacent_faces = mesh.get_face_adjacent_faces(int(index))

                # Check if center points of adjacent faces have a lower Z-value
                z = None

                for ad in adjacent_faces:
                    if not z:
                        z = center_z[ad]
                        i = ad

                    elif z > center_z[ad]:
                        z = center_z[ad]
                        i = ad

                if z > pt[2]:
                    v0, v1, v2 = face_vertices(index)
                    pt = ls_geo.lowest_face_vertex(v0, v1, v2)

                    if len(adjacent_faces) < 3:
                        over = over_edge(pt)

                        if over:
                            particles.append(pt)
                            index = over
                            pt = start_points[index][1]

                        else:
                            run = False

                    else:
                        run = False

                else:
                    index = start_points[i][0]
                    pt = start_points[i][1]

                particles.append(pt)
                face_indices.append(int(index))
            #print('particles:',particles)
            #print(len(particles))

            # End task
            drain_points.append(particles)
            drain_faces.append(face_indices)
            q.task_done()

    # Call task function
    q = queue.Queue()

    for i in range(int(cpus)):
        t = threading.Thread(target=drain_path)
        t.setDaemon(True)
        t.start()

    # Put jobs in queue
    for pts in start_points:
        q.put(pts)

    # Wait until all tasks in the queue have been processed
    q.join()

    # Open file, which the points should be written to
    pt_file = open('drain_points.txt', 'w')
    face_file = open('drain_faces.txt', 'w')

    # Write points to file
    for particles in drain_points:
        for pt in particles:
            pt_file.write(str(pt[0]) + ',' + str(pt[1]) + ',' + str(pt[2]) + '\t')
        pt_file.write('\n')

    # Write face indices to file
    for curves in drain_faces:
        for index in curves:
            face_file.write(str(index) + '\t')
        face_file.write('\n')


    #Close outfiles and save mesh
    pt_file.close()
    face_file.close()
    pm.save_mesh('new_drain_mesh.obj', mesh)

    return True


def drain_pools(path):
    import pymesh as pm
    from numpy import array, allclose
    from numpy import sum as npsum
    from scipy.optimize import newton

    # Paths
    meshFile = path + '/drainMesh.obj'
    endPtsFile = path + '/EndPoints.txt'
    volPtsFile = path + '/VolumePoints.txt'

    # Initialize Mesh
    mesh = pm.load_mesh(meshFile)
    mesh.enable_connectivity()
    mesh.add_attribute('face_centroid')
    mesh.add_attribute('face_index')
    mesh.add_attribute('face_area')
    cenPts = mesh.get_attribute('face_centroid')
    faceIndex = mesh.get_attribute('face_index')
    faceArea = mesh.get_attribute('face_area')
    faceVert = mesh.faces
    vertices = mesh.vertices
    #print(mesh.get_attribute_names())
    boolWarning = None
    poolWarning = None

    # Construct face center list
    faceCen = []
    i = 0
    while i < len(cenPts):
        faceCen.append(array([float(cenPts[i]), float(cenPts[i + 1]), float(cenPts[i + 2])]))
        i += 3

    # Load points
    ptsLine = open(endPtsFile, 'r').readlines()
    endPts = []
    for l in ptsLine:
        l = l[:-1]
        l = l.split(',')
        endPts.append(array([float(l[0]), float(l[1]), float(l[2])]))
    #print(len(endPts))

    # Load volumes
    volLine = open(volPtsFile, 'r').readlines()
    vol = []
    for v in volLine:
        v = v[:-1]
        vol.append(float(v))

    pts = []
    vols = []
    fI = []

    for i,pt in enumerate(endPts):

        # Check if point is in list:
        if i == 0:
            pts.append(pt)
            vols.append(vol[i])

            # Find equivalent face center of points
            for index, cen in enumerate(faceCen):
                if allclose(cen,pt):
                    fI.append(index)
                    break

        else:
            found = False
            j = 0
            while j < len(pts):
                # If it is in list: add volume
                if allclose(pts[j], pt):
                    vols[j] += vol[i]
                    j = len(pts)
                    found = True
                j += 1

            # Else: put point and volume in list
            if not found:
                pts.append(pt)
                vols.append(vol[i])

                # Find equivalent face center of points
                for index, cen in enumerate(faceCen):
                    if allclose(cen,pt):
                        fI.append(index)
                        break

    # Pool function
    def pool(faceIndex,point,volume):
        found = False

        # Compute first z-value
        A = faceArea[faceIndex]
        h = volume/A
        Z = point[2]+h

        # Initialize face index, z-values and areas
        adjFace = [faceIndex, ]
        faceZ = [point[2],]
        faceA = [A,]

        # Find adjacent faces
        for faceIn in adjFace:

            for af in mesh.get_face_adjacent_faces(faceIn):

                # Get Z-value of face-centroid
                fc = faceCen[af][2]

                # Append them to list if their centroid is lower than the computed Z-value and are not already in list
                if fc < Z:
                    if af not in adjFace:

                        # If current face holds a volume add that volume to the current volume
                        if af in fI:
                            #print('found in fI')
                            queueIndex = fI.index(af)

                            if queueIndex in notDoneList:
                                #print('found in notDoneList')
                                volume += vols[queueIndex]
                                notDoneList.remove(queueIndex)
                                doneList.append(queueIndex)

                            elif queueIndex in doneList:
                                #print('found in doneList')
                                vols[queueIndex] += volume
                                notDoneList.append(queueIndex)
                                doneList.remove(queueIndex)
                                return

                            else:
                                pass

                        # Append Z-value, area and face-index
                        faceZ.append(fc)
                        faceA.append(faceArea[af])
                        adjFace.append(int(af))

                        # Convert to numpy array
                        faZ = array(faceZ)
                        faA = array(faceA)

                        # Compute new z-value
                        Z = (npsum(faZ*faA)+volume)/npsum(faA)

        #print('Approx Z:',Z)

        # Create approximate volume mesh
        apxVert = []
        apxFace = []
        iApxVert = 0

        for af in adjFace:
            iApxVert = len(apxVert)
            apxVert.append(vertices[faceVert[af][0]])
            apxVert.append(vertices[faceVert[af][1]])
            apxVert.append(vertices[faceVert[af][2]])
            apxFace.append([iApxVert, iApxVert + 1, iApxVert + 2])

        # Create boundary mesh
        apxVert = array(apxVert)
        apxFace = array(apxFace)
        apxMesh = pm.form_mesh(apxVert, apxFace)

        # Boundary Box
        maxmin = apxMesh.bbox
        x1, y1, z1 = maxmin[0]
        x2, y2, z2 = maxmin[1]*1.1  # Increase Bbox with 10%
        x1 = x1*0.9  # Decrease Bbox with 10%
        y1 = y1*0.9  # Decrease Bbox with 10%

        #print('apxMesh:',maxmin[0],'\n\t',maxmin[1])

        zMax = mesh.bbox[1][2]
        #print('zMax:',zMax)
        #pm.save_mesh('apxmesh.obj', apxMesh)

        # Findheight helper functions
        def createBbox(z):
            bVert = []
            bFace = []
            bVox = []

            # Add vertices
            bVert.append(array([x1, y1, z1]))  # 0
            bVert.append(array([x1, y2, z1]))  # 1
            bVert.append(array([x1, y2, z]))  # 2
            bVert.append(array([x1, y1, z]))  # 3

            bVert.append(array([x2, y2, z]))  # 4
            bVert.append(array([x2, y2, z1]))  # 5
            bVert.append(array([x2, y1, z1]))  # 6
            bVert.append(array([x2, y1, z]))  # 7

            # Add faces
            bFace.append([0, 1, 3])  # side 1
            bFace.append([1, 2, 3])  # side 1
            bFace.append([0, 3, 7])  # side 2
            bFace.append([0, 6, 7])  # side 2
            bFace.append([7, 6, 5])  # side 3
            bFace.append([5, 7, 4])  # side 3
            bFace.append([4, 5, 1])  # side 4
            bFace.append([4, 2, 1])  # side 4
            bFace.append([0, 1, 6])  # side 5
            bFace.append([1, 5, 6])  # side 5
            bFace.append([3, 7, 2])  # side 6
            bFace.append([2, 7, 4])  # side 6

            # Add voxels
            bVox.append([0, 2, 3, 7])
            bVox.append([0, 1, 2, 7])
            bVox.append([0, 1, 6, 7])
            bVox.append([2, 4, 5, 7])
            bVox.append([1, 2, 5, 6])
            bVox.append([2, 4, 6, 7])

            # Create boundary mesh
            bVert = array(bVert)
            bFace = array(bFace)
            bVox = array(bVox)
            bMesh = pm.form_mesh(bVert, bFace, bVox)
            #pm.save_mesh('bMesh.obj', bMesh)

            return bMesh

        def getVolMesh(newMesh, bottomFaces, z):

            # Prepare to create volume mesh
            newMeshVert = newMesh.vertices
            volVert = []
            volFace = []
            volVox = []

            # Create volume mesh from bottom part of mesh
            for f in bottomFaces:
                iVer = len(volVert)

                oldVerts = []
                newVerts = []
                for v in f:
                    oldVerts.append(newMeshVert[v])
                    newV = array([newMeshVert[v][0], newMeshVert[v][1], z])
                    newVerts.append(newV)

                # Append vertices
                volVert += oldVerts
                volVert += newVerts

                # Append faces
                volFace.append([iVer, iVer + 1, iVer + 2])
                volFace.append([iVer + 3, iVer + 4, iVer + 5])

                # Append voxels
                volVox.append([iVer, iVer + 1, iVer + 2, iVer + 3])
                volVox.append([iVer + 1, iVer + 3, iVer + 4, iVer + 5])
                volVox.append([iVer + 1, iVer + 2, iVer + 3, iVer + 5])

            # Create volume mesh
            volVert = array(volVert)
            volFace = array(volFace)
            volVox = array(volVox)
            volMesh = pm.form_mesh(volVert, volFace, volVox)

            return volMesh

        def intersectAndBottomFaces(bMesh, z):
            warning = None

            # Make intersection with auto boolean engine
            newMesh = pm.boolean(mesh, bMesh, 'intersection')

            if newMesh.num_faces == 0:
                # Change boolean engine to Cork
                warning = 'Changing Boolean Engine to Cork!'
                print(warning)
                newMesh = pm.boolean(bMesh, mesh, 'difference', engine='cork')

            #pm.save_mesh('intMesh.obj', newMesh)

            # Get bottom part of mesh
            try:
                newSource = newMesh.get_attribute('source')
                newFace = newMesh.faces
                bottomFaces = []

                for i, s in enumerate(newSource):
                    if int(s) == 1:
                        bottomFaces.append(newFace[i])

                return newMesh, bottomFaces, warning

            except RuntimeError:
                # Try different approach to getting bottom faces
                newMesh.add_attribute('face_centroid')
                newFace = newMesh.faces
                # print('len newFace:',len(newFace))
                # print('first newFace:',newFace[0])
                newCen = newMesh.get_attribute('face_centroid')
                bottomFaces = []

                for newFaceIndex in range(len(newFace)):
                    newCenZ = newCen[newFaceIndex * 3 + 2]
                    if newCenZ < z:
                        bottomFaces.append(newFace[newFaceIndex])

                return newMesh, bottomFaces, warning

        # Volume function to solve
        def findHeight(z):
            #print('current z:',z)

            # Check if pools will overflow mesh
            if z > zMax:
                z = zMax

            # Create Bbox
            bMesh = createBbox(z)

            # Make intersection
            newMesh, bottomFaces, warning = intersectAndBottomFaces(bMesh, z)

            # Create volume mesh
            volMesh = getVolMesh(newMesh, bottomFaces, z)

            if z == zMax:
                return 0

            else:
                # Compute volume
                volMesh.add_attribute('voxel_volume')
                volVol = volMesh.get_attribute('voxel_volume')
                volVol = sum(list((map(abs, volVol))))

                #print('volume',volume)
                #print('volVol1',volVol)

                return volume - volVol

        # Get final height
        zFinal = newton(findHeight,Z)

        # Create final mesh
        def finalMesh(z):
            poolWarning = None

            # Check if pools will overflow mesh
            if z > zMax:
                z = zMax
                poolWarning = 'The pool have a greater volume than the mesh can contain. Pool set to fill entire mesh.'

            # Create Bbox
            bMesh = createBbox(z)

            # Make intersection
            newMesh, bottomFaces, boolWarning = intersectAndBottomFaces(bMesh, z)

            # Create volume mesh
            volMesh = getVolMesh(newMesh, bottomFaces, z)

            volMesh.add_attribute('voxel_volume')
            volVol = volMesh.get_attribute('voxel_volume')
            volVol = sum(list(map(abs, volVol)))

            # Clean up mesh
            volMesh, info = pm.remove_isolated_vertices(volMesh)
            #print('num vertex removed', info["num_vertex_removed"])
            volMesh, info = pm.remove_duplicated_faces(volMesh)

            return volMesh, volVol, poolWarning, poolWarning

        # Save final mesh
        #print('zFinal',zFinal,'type:',type(zFinal))
        finalMesh, finalVol, poolWarning, boolWarning = finalMesh(zFinal)
        meshName = "poolMesh_" + str(faceIndex) + ".obj"
        hullMesh = pm.compute_outer_hull(finalMesh)
        pm.save_mesh(meshName, hullMesh)

        print(' ')
        print('volume',"{0:.3f}".format(volume))
        print('computed volume',"{0:.3f}".format(finalVol))
        print('closed?',finalMesh.is_closed())
        print(' ')

        return meshName


    # Initialize pool-loop
    Z = []
    i = 0
    doneList = []
    notDoneList = list(range(0,len(pts)))
    loopLength = len(notDoneList)
    meshNames = []

    # Use pool function on each set of points
    while i < loopLength:
        I = notDoneList.pop(i)
        names = pool(fI[I],pts[I],vols[I])

        # Put meshNames in name list
        if names:
            if not names in meshNames:
                meshNames.append(names)
            else:
                pass

        doneList.append(i)
        loopLength = len(notDoneList)

    # Open InData and edit last line
    file_obj = open("InData.txt",'r')
    file = file_obj.readlines()
    file_obj.close()

    mNames = ''
    for n in meshNames:
        mNames += ',' + n

    file[6] = 'meshNames.txt' + mNames

    outfile_obj = open("InData.txt", 'w')
    outfile_obj.writelines(file)

    # Write meshNames.txt
    file_obj = open("meshNames.txt", 'w')
    file_obj.write(mNames)
    file_obj.close()

    #print('function warn', [boolWarning, poolWarning])
    if boolWarning or poolWarning:
        return [boolWarning, poolWarning]
    else:
        return None


class simple_rain():
    def __init__(self, cpus, precipitation, windSpeed, windDirection, testPoints, testVectors, context, temperature, k):
        self.prec = precipitation
        self.windSpeed = windSpeed
        self.windDir = windDirection
        self.testPts = testPoints
        self.testVecs = testVectors
        self.context = context
        self.temp = temperature
        self.kMiss = k[0]
        self.kHit = k[1]
        self.dirVec = False
        self.hourlyResult = False
        self.wdr = False
        self.xyAngles = []
        self.yzAngles = []
        self.cpus = int(cpus)

    # Final function
    def rainHits(self):
        from math import degrees, exp, log, acos, sqrt, pi, cos, radians
        from rhinoscriptsyntax import XformMultiply, VectorCreate, AddPoint, VectorTransform, XformRotation2
        from Rhino.Geometry.Intersect.Intersection import RayShoot
        from Rhino.Geometry import Ray3d
        import threading
        import Queue

        # Helper functions

        def rain_vector(Vw, regn):

            # Rain drop radius:
            A = 1.3
            p = 0.232
            n = 2.25

            if regn == 0:
                return 0

            def f_a(I):
                return A * I ** p

            a = f_a(regn)
            r = a * exp(log(-log(0.5)) / n) / 1000

            # Angle:
            # rho_L = 1.2
            # rho_w = 1000
            # g = 9.81
            # c = 0.3
            # alpha = 3*c*rho_L*Vw^2*r^2/sqrt(4*r^4*(9*Vw^2*c^2*rho_L^2+64*g^2*r^2*rho_w^2))
            # Simplified it becomes:

            a = (0.54 * Vw ** 2 * r ** 2) / sqrt(r ** 4 * (1.1664 * Vw ** 2 + 6.159110400 * 10 ** 9 * r ** 2))
            if a > 1:
                a = 1
            alpha = acos(a)

            return alpha

        def rotate_yz(angle):
            return XformRotation2(angle, [1, 0, 0], [0, 0, 0])

        def rotate_xy(angle):
            return XformRotation2(angle, [0, 0, 1], [0, 0, 0])

        # Correction of wind direction
        def B_wind(angle, direction):
            c = 360 - (angle + 90)
            # print(c)

            a = abs(c - direction)
            if a < 180:
                b = a
            else:
                b = 360 - a

            return radians(b)

        def rayShoot():
            """Build on: Ladybug - RayTrace"""

            # Initialize
            while not q.empty():
                numOfBounce = 1
                startPt, xyAngle, yzAngle = q.get()

                vector = direction_vector
                ray = Ray3d(startPt, vector)
                # Check the wind direction
                B = B_wind(xyAngle, self.windDir[i])
                # print(B)
                if B > pi / 2:
                    # print('more than 90')
                    hourly_rain.append(0)
                    hourly_result.append(False)

                else:
                    # print('less than 90')
                    # Compute rain amount
                    K_wind = cos(B) / sqrt(
                        1 + 1142 * (sqrt(self.prec[i]) / self.windSpeed[i] ** 4)) * exp(
                        -12 / (self.windSpeed[i] * 5 * self.prec[i] ** (0.25)))

                    # Shoot ray
                    intPt = RayShoot(ray, [self.context], numOfBounce)

                    # Check for intersection
                    if intPt:
                        # print('Intersection!')
                        hourly_result.append(True)
                        kRain = self.kHit
                        # verticalFactor = (1/(90*K_wind*kRain)-1/90)*yzAngle
                        # print(verticalFactor)
                        hourly_rain.append((K_wind * kRain * self.prec[i]))

                    else:
                        # print('No intersection!')
                        hourly_result.append(False)
                        kRain = self.kMiss
                        # verticalFactor = (1 / (90 * K_wind * kRain) - 1 / 90) * yzAngle
                        # print('vf',verticalFactor)
                        hourly_rain.append((K_wind * kRain * self.prec[i]))

                # print('done')
                q.task_done()

        result = []
        dirVec_hourly = []
        wdr = []

        for i in range(0, len(self.prec)):

            if self.temp[i] <= -2 or self.windSpeed[i] <= 0 or self.prec[i] <= 0:
                dirVec_hourly.append(None)
                result.append([False] * len(self.testPts))

            else:

                # Rotate vectors towards the sky
                R_v = rain_vector(self.windSpeed[i], self.prec[i])
                towards_sky = rotate_yz(degrees(R_v))

                # Rotate vectors towards the wind
                w_d = self.windDir[i]
                towards_wind = rotate_xy(w_d)

                # Combine:
                transformation = XformMultiply(towards_wind, towards_sky)
                north_vector = VectorCreate(AddPoint(0, 0, 0), AddPoint(0, -1, 0))
                direction_vector = VectorTransform(north_vector, transformation)
                hourly_result = []
                hourly_rain = []

                # Put jobs in queue
                q = Queue.Queue()
                for fi, pts in enumerate(self.testPts):
                    q.put((pts, self.xyAngles[fi], self.yzAngles[fi]))
                # Call task function

                for c in range(self.cpus):
                    t = threading.Thread(target=rayShoot)
                    t.setDaemon(True)
                    t.start()

                # break
                # Wait until all tasks in the queue have been processed
                q.join()

                dirVec_hourly.append(direction_vector)
                result.append(hourly_result)
                wdr.append(hourly_rain)

        self.hourlyResult = result
        self.dirVec = dirVec_hourly
        self.wdr = wdr

    def computeAngles(self):
        import Rhino.Geometry as rc
        from math import degrees

        # Construct planes
        zero = rc.Point3d(0, 0, 0)
        z = rc.Vector3d(0, 0, 1)
        x = rc.Vector3d(1, 0, 0)
        y = rc.Vector3d(0, 1, 0)
        xy = rc.Plane(zero, z).WorldXY
        yz = rc.Plane(zero, x).WorldYZ

        # Compute angles on the XY and YZ plane
        for fn in self.testVecs:
            self.xyAngles.append(degrees(rc.Vector3d.VectorAngle(fn, y, xy)))
            yz_tmp = degrees(rc.Vector3d.VectorAngle(fn, z, yz))

            # Correct angles
            if yz_tmp > 90:
                yz_tmp = 180 - yz_tmp
            elif yz_tmp > 180:
                yz_tmp = yz_tmp - 180
            elif yz_tmp > 270:
                yz_tmp = abs(yz_tmp - 360)
            elif yz_tmp < 0:
                yz_tmp = yz_tmp * (-1)

            self.yzAngles.append(yz_tmp)


def topographic_index(meshPath, drainCurvesPath):
    import numpy as np
    import GeometryClasses as gc
    import pymesh as pm

    # Load mesh and curves
    mesh = pm.load_mesh(meshPath)
    drainCurves = []
    file = open(drainCurvesPath, 'r')
    lines = file.readlines()
    for l in lines:
        drainCurves.append(int(i) for i in l.split(',')[1:-1])

    # Initilize mesh data
    mesh.add_attribute('face_normal')
    mesh.add_attribute('face_area')
    faceArea = mesh.get_attribute('face_area')
    fn = mesh.get_attribute('face_normal')
    faceNormal = []
    i = 0
    while i < len(fn):
        faceNormal.append(np.array([fn[i], fn[i+1], fn[i+2]]))
        i += 3
    drainArea = faceArea
    TI = []

    def topoIndex(a, beta):
        return np.log(a / np.tan(beta))

    def computeBeta(normal):
        z = np.array([0,0,1])
        return gc.angleBetweenVectors(z, normal, forceAngle='acute')[0]

    def processDrainCurve(curveIndex):
        """Processes a single drain curve"""

        A = 0

        for face in drainCurves[curveIndex]:
            a = faceArea[face]
            drainArea[face] += a
            A += a

        return True

    for curve in range(len(drainCurves)):
        processDrainCurve(curve)

    for face in range(mesh.num_faces):
        a = drainArea[face]
        b = computeBeta(faceNormal[face])
        TI.append(topoIndex(a,b))

    # Write topographic indices to file
    topoFile = open('topographicIndex.txt', 'w')

    for face in TI:
        topoFile.write(str(face) + '\n')
    topoFile.write('\n')

    topoFile.close()

