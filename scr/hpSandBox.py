def skinSur(
              VolumeNode="croppedROI",
              MiniThresd="150",
              modName="boneTempt",
              color="blue",
              opacity=0.8,
              ):

    masterVolumeNode = slicer.util.getNode(VolumeNode)

    # Create segmentation
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
    addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("skin")
    maximumHoleSizeMm = .5

    # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass(
      "vtkMRMLSegmentEditorNode"
    )
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)

    # Thresholding
    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold", MiniThresd)
    #   effect.setParameter("MaximumThreshold","695")
    effect.self().onApply()

    segmentEditorWidget.setActiveEffectByName("Margin")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MarginSizeMm",str(maximumHoleSizeMm))
    effect.self().onApply()
  
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
    effect.self().onApply()
  
    # Invert the segment
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "INVERT")
    effect.self().onApply()
    # Remove islands in inverted segment (these are the holes inside the segment)
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
    effect.self().onApply()
    # Grow the inverted segment by the same margin as before to restore the original size
    segmentEditorWidget.setActiveEffectByName("Margin")
    effect = segmentEditorWidget.activeEffect()
    effect.self().onApply()
    # Invert the inverted segment (it will contain all the segment without the holes)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "INVERT")
    effect.self().onApply()
  
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
    effect.self().onApply()
    # # Add it to the allVertebraeSegment
    # segmentEditorWidget.setCurrentSegmentID(addedSegmentID)
    # segmentEditorWidget.setActiveEffectByName("Logical operators")
    # effect = segmentEditorWidget.activeEffect()
    # effect.setParameter("Operation", "UNION")
    # effect.setParameter("ModifierSegmentID", segmentID)
    # effect.self().onApply()
    # Clean up
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    # Make segmentation results visible in 3D
    segmentationNode.CreateClosedSurfaceRepresentation()

    polyData = vtk.vtkPolyData()

    # Make sure surface mesh cells are consistently oriented
    surfaceMesh = segmentationNode.GetClosedSurfaceRepresentation(addedSegmentID, polyData)
    normals = vtk.vtkPolyDataNormals()
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.SetInputData(polyData)
    normals.Update()
    surfaceMesh = normals.GetOutput()

    vtkNode = slicer.modules.models.logic().AddModel(surfaceMesh)
    vtkNode.SetName(modName)

    # segmentationNode.GetDisplayNode().SetVisibility(False)
    modelDisplay = vtkNode.GetDisplayNode()
    modelDisplay.SetColor(Helper.myColor(color))  # yellow
    modelDisplay.SetOpacity(opacity)
    modelDisplay.SetBackfaceCulling(0)
    # modelDisplay.SetVisibility(1)
    modelDisplay.SetSliceIntersectionVisibility(True)

    slicer.mrmlScene.RemoveNode(segmentationNode)
    return
