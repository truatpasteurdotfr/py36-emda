# Run codes for proshade
import numpy as np
import proshade

def proshade_overlay(map1, map2, fitresol=4.0):
    ps                = proshade.ProSHADE_settings()
    ps.task           = proshade.OverlayMap
    ps.verbose        = -1                                                              
    ps.setResolution  (fitresol)                                                           
    ps.addStructure   (map1)
    ps.addStructure   (map2)
    
    rn                = proshade.ProSHADE_run(ps)
    eulerAngles       = rn.getEulerAngles()
    rotMatrix    = rn.getOptimalRotMat()
    return rotMatrix

def get_symmops_from_proshade(mapname):
    import sys
    import numpy
    import proshade

    ### Create the settings object
    #pSet = proshade.ProSHADE_settings()
    #### Set settings values
    #pSet.task = proshade.Symmetry
    #pSet.verbose = 1
    #pSet.setResolution(7.0)
    #pSet.moveToCOM = False
    #pSet.changeMapResolution = True
    #pSet.changeMapResolutionTriLinear = False

    ### Create the structure object
    #pStruct = proshade.ProSHADE_data(pSet)

    ### Read in the structure
    #pStruct.readInStructure(mapname, 0, pSet)

    ### Process map
    #pStruct.processInternalMap(pSet)

    ### Map to spheres
    #pStruct.mapToSpheres(pSet)

    ### Compute spherical harmonics
    #pStruct.computeSphericalHarmonics(pSet)

    ### Compute self-rotation function
    #pStruct.computeRotationFunction(pSet)

    ### Detect symmetry
    #pStruct.detectSymmetryInStructurePython(pSet)
    #pStruct.detectSymmetryInStructure(pSet)
    #recSymmetryType = pStruct.getRecommendedSymmetryType(pSet)
    #recSymmetryFold = pStruct.getRecommendedSymmetryFold(pSet)
    ##recSymmetryAxes = proshade.getRecommendedSymmetryAxesPython(pStruct, pSet)
    ##recSymmetryAxes = pStruct.getRecommendedSymmetryAxes(pSet) # works
    ##recSymmetryAxes = pStruct.getAllCSyms(pSet)
    #recSymmetryAxes = pStruct.getAllCSyms(pSet)
    ##recSymmetryAxes = proshade.getSymmetryAxis(pStruct, pSet)


    """ Create the settings object """
    ps                                    = proshade.ProSHADE_settings ( )

    """ Set up the run """
    ps.task                               = proshade.Symmetry
    ps.verbose                            = -1;                      
    ps.setResolution                      ( 8.0 )                  
    ps.addStructure                       ( mapname )
    #ps.usePhase = False # to get the phaseless rotation function.

    """ Run ProSHADE """
    rn                                    = proshade.ProSHADE_run ( ps )

    """ Retrieve results """
    recSymmetryType = rn.getSymmetryType ( )
    recSymmetryFold = rn.getSymmetryFold ( )
    recSymmetryAxes = rn.getAllCSyms     ( )
    """ print("Detected " + str(recSymmetryType) + "-" + str(recSymmetryFold) + " symetry.")
    proshade_pg = str(recSymmetryType) + str(recSymmetryFold)
    print("Proshade point group: ", proshade_pg)
    print( "Found a total of " + str ( len ( allCs ) ) + " cyclic symmetries." )
    print( "Fold      x         y         z       Angle     Height   Avg. FSC" )
    print( "  %s    %+1.3f    %+1.3f    %+1.3f    %+1.3f    %+1.4f     %1.3f" % ( float ( recSymmetryAxes[0] ),
                                                                                float ( recSymmetryAxes[1] ),
                                                                                float ( recSymmetryAxes[2] ),
                                                                                float ( recSymmetryAxes[3] ),
                                                                                float ( recSymmetryAxes[4] ),
                                                                                float ( recSymmetryAxes[5] ),
                                                                                float ( recSymmetryAxes[6] ) ) ) """
    ### Print results
    print("Detected " + str(recSymmetryType) + "-" + str(recSymmetryFold) + " symetry.")
    proshade_pg = str(recSymmetryType) + str(recSymmetryFold)
    print("Proshade point group: ", proshade_pg)
    print("Fold      x         y         z       Angle     Height     Avg. FSC")
    print(type(recSymmetryAxes))
    for iter in range(0, len(recSymmetryAxes)):
        print(
            "  %s    %+1.3f    %+1.3f    %+1.3f    %+1.3f    %+1.4f    %1.3f"
            % (
                recSymmetryAxes[iter][0],
                recSymmetryAxes[iter][1],
                recSymmetryAxes[iter][2],
                recSymmetryAxes[iter][3],
                recSymmetryAxes[iter][4],
                recSymmetryAxes[iter][5],
                recSymmetryAxes[iter][6],
            )
        )
    fold, x, y, z, theta, peakh = [], [], [], [], [], []
    for row in recSymmetryAxes:
        fold.append(int(row[0]))
        x.append(row[1])
        y.append(row[2])
        z.append(row[3])
        theta.append(row[4])
        peakh.append(row[5])
    return [fold, x, y, z, peakh, proshade_pg]

