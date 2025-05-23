# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Slicer 5.8
#     language: python
#     name: slicer-5.8
# ---

# # slicer imports

#%%
import collections
# import shelve
import csv
import datetime
import json
import logging
import traceback
# import pandas as pd
# import string
import math
import os
import re
import time
import unittest
from abc import ABCMeta
from collections import defaultdict as dfDic
from errno import ESTALE
# from msilib.schema import Dialog
from operator import truediv
from pprint import pp
from xml.etree.ElementTree import TreeBuilder
# import matplotlib
# import ctk
import DICOM
import numpy as np
import PythonQt
import qt
import slicer
import vtk
# import numpy.linalg as npl
from numpy.lib.format import write_array
from scipy.spatial.distance import cdist
from slicer import util as ut
from slicer.ScriptedLoadableModule import *
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import SimpleITK as sitk
import sitkUtils
# import trimesh
# import sys
# from itertools import repeat
# from Helper import *
# from viztracer import VizTracer
import numpy.typing as npt
import typing
from typing import Optional as Opt
from typing import Literal as Lit
from typing import Any
from typing import Iterable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable as Cal
from typing import Sequence as Seq
from typing import DefaultDict as dfDic
# PS = Opt[npt.ArrayLike]
ut  = slicer.util
MOD = 'vtkMRMLModelNode'
OB = vtk.vtkOBBTree
NOD = slicer.vtkMRMLTransformableNode
VOL = slicer.vtkMRMLVolumeNode
VPD = vtk.vtkPolyData
SMD = slicer.vtkMRMLModelNode
LR = 'LR'
LT = 'LT'
MTHR = 100
SCENE = slicer.mrmlScene
ndA = np.asanyarray
npl = np.linalg
eps = 1e-6
PS = Union[ Seq[Tuple[float, float]],
            Seq[Tuple[float, float, float]]]

# 使用
points_2d: PS = [(1.0, 2.0), ndA([3.0, 4.0])]

points_3d: PS = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


# hp = Helper

# from chpoints import watch
# from viztracer import log_sparse
# from objprint import objjson

# import vtkSlicerCombineModelsModuleLogicPython as vtkbool

# # [[Helper]]类
# - Helper

class Helper(object):
    """
    classdocs
    """


    # ### SetBgFgVolumes
    # - SetBgFgVolumes
    # hp = Helper
    @staticmethod
    def SetBgFgVolumes(bg):

        appLogic = slicer.app.applicationLogic()
        selectionNode = appLogic.GetSelectionNode()
        selectionNode.SetReferenceActiveVolumeID(bg)
        appLogic.PropagateVolumeSelection()

    @staticmethod
    def p3Nor(p1: PS,
              p2: PS,
              p0: PS,
              unit: bool = True
              ) -> Opt[np.ndarray]:
        '''p3Nor 三点定平面

        非共线三点定平面的单位法向量

        Args:
            p1 (_type_): 点1
            p2 (_type_): 点2
            p0 (_type_): 点3

        Returns:
            - 平面的单位法向量

        Note:
            - 叉乘(n)等于0, 表示平行
        '''
        p1 = ndA(p1)
        p2 = ndA(p2)
        p0 = ndA(p0)
        n = np.cross(p1 - p0, p2 - p0) # Normal
        if unit == False:
            return n
        else:
            if npl.norm(n) != 0:
                return n/npl.norm(n) # unit normal vector;
            else:
                print("三点共线了吧...")
                return


# ### SetLabelVolume
# - SetLabelVolume
    @staticmethod
    def SetLabelVolume(lb):
        appLogic = slicer.app.applicationLogic()
        selectionNode = appLogic.GetSelectionNode()
        selectionNode.SetReferenceActiveLabelVolumeID(lb)
        appLogic.PropagateVolumeSelection()

    # ### findChildren
    # - findChildren

    @staticmethod
    def findChildren(widget=None, name="", text=""):
        """return a list of child widgets that match the passed name"""
        # TODO: figure out why the native QWidget.findChildren method
        # does not seem to work from PythonQt
        if not widget:
            widget = mainWindow()
        children = []
        parents = [widget]
        while parents != []:
            p = parents.pop()
            parents += p.children()
            if name and p.name.find(name) >= 0:
                children.append(p)
            elif text:
                try:
                    p.text
                    if p.text.find(text) >= 0:
                        children.append(p)
                except AttributeError:
                    pass
        return children

# ###
# -

    @staticmethod
    def ortMark(ortMarker, op = None, drts = None, coord=True, mNam='',**kw):
        # 如果coord为真,则创建坐标轴并合并到ortMarker中
        ortMarker = Helper.namNod(ortMarker)
        if coord:
            cNod = Helper.coord(op,drts)
            ortMarker = Helper.pdsAdd(ortMarker,cNod,mNam=mNam+'_ortMarker')
        viewNodes = slicer.util.getNodesByClass("vtkMRMLAbstractViewNode")
        for viewNode in viewNodes:
            viewNode.SetOrientationMarkerHumanModelNodeID(ortMarker.GetID())
            viewNode.SetOrientationMarkerType(viewNode.OrientationMarkerTypeHuman)
            viewNode.SetOrientationMarkerSize(viewNode.OrientationMarkerSizeMedium)
        return Helper.pdMDisp(ortMarker,mNam,**kw)

    @staticmethod
    def pdsAdd(*pdList, mNam='',**kw) -> VPD:
        '''pdsAdd 合并多个polyData

        将多个vtk.polyData合并为一个vtk.polyData

        Args:
            *pdList (tuple): 多个vtk.polyData

        Returns:
            - 合并后的vtk.polyData
        '''
        appendFilter = vtk.vtkAppendPolyData()
        for pd in pdList:
            pd = Helper.getPd(pd)
            appendFilter.AddInputData(pd)
        appendFilter.Update()
        return Helper.pdMDisp(appendFilter.GetOutput(),mNam,**kw)

# 生成函数coord, 生成三维坐标系的model, 参数包括: 原点坐标和xyz的正方向和长度,返回一个model: 三个圆柱为坐标系的轴, 用圆锥标识方向
    @staticmethod
    def coord(origin: PS = [0, 0, 0],
                drts = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                lens = [30, 50, 30],
                mNam: str = 'coord',
                **kw) -> VPD:
        '''coord 生成三维坐标系的model
        '''
        x,y,z = drts
        xLen,yLen,zLen = lens
        # 三轴终点
        xAx = Helper.p2pLn(origin, drt = x)[2]
        yAx = Helper.p2pLn(origin, drt = y)[2]
        zAx = Helper.p2pLn(origin, drt = z)[2]
        # 生成圆锥
        xCone = Helper.p3Cone(xAx(xLen), drt=x, rad=2, high=6, mNam=mNam + '_xCone', **kw)
        yCone = Helper.p3Cone(yAx(yLen), drt=y, rad=2, high=6, mNam=mNam + '_yCone', **kw)
        zCone = Helper.p3Cone(zAx(zLen), drt=z, rad=2, high=6, mNam=mNam + '_zCone', **kw)
        # 生成圆柱
        xCyl = Helper.p2pCyl(origin, xAx(xLen+1), rad=0.6, mNam=mNam + '_xCyl', **kw)
        yCyl = Helper.p2pCyl(origin, yAx(yLen+1), rad=0.6, mNam=mNam + '_yCyl', **kw)
        zCyl = Helper.p2pCyl(origin, zAx(zLen+1), rad=0.6, mNam=mNam + '_zCyl', **kw)
        # 合并
        return Helper.pdsAdd(xCone, yCone, zCone, xCyl, yCyl, zCyl, mNam=mNam, **kw)


    #tag mxNor 定向
    # @staticmethod
    # def mxNor(ps,
    #         cp    = None,
    #         nor   = [1,0,0],
    #         reC   = True,
    #         mNam  = ''):
    #   ''' 在某个方向找极值

    #   '''
    #   ps    = Helper.getArr(ps)
    #   if cp is None:
    #     cp  = np.mean(ps, 0) # 求点集中心
    #   nor   = ndA(nor)
    #   if not isUNor(nor):
    #     nor = p2pNor(cp, nor)[0]
    # #   pj    = np.dot(ps-cp, nor)
    #   pj    = np.dot(ps, nor)
    #   minP  = ps[np.argmin(pj)]
    #   maxP  = ps[np.argmax(pj)]
    #   norm  = p2pLn(minP, maxP, mNam=mNam)[1]
    #   if reC:
    #     if allClose(nor, norm):
    #       return norm
    #     return mxNor(ps, cp, norm)
    #   else:
    #     return norm, maxP, minP


    #tag p2pLn 点对点生成线
    def p2pLn(p: PS,
          pN: PS       = None,
          plus: float  = eps,
          mNam: str = '',
          dia: float   = 1.,
          **kw
                ):
      p         = ndA(p) *1.
      pN        = ndA(pN)*1.
      nor, dst  = p2pNor(pN)
      if not isUNor(pN):
        nor, dst= p2pNor(p, pN)
        dst    += plus
      else:
        dst     = plus
      pp        = p+nor*dst
      if mNam != "":
        marksNod = SCENE.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", mNam)
        marksNod.AddControlPoint(vtk.vtkVector3d(p))
        marksNod.AddControlPoint(vtk.vtkVector3d(pp))
        Helper.markDisp(mNam, gType=3, lDia=.2)
        p2pCone(pp, nor, mNam=mNam)
      return  pp, nor, dst, lambda x: p + nor * x

    #tag obbBox 生成obbBox
    def obBx(cpXyz,
          mNam: str="r",
          xyzSort = True,
          **kw: any
          )->any:
      '''obbBox 生成obbBox
      🧮: 生成obbBox

      🔱:
          ps (PS): 点集
          mNam (str, optional): 莫名. Defaults to "r".
          xyzSort (bool, optional): 🧭排序. Defaults to True.
          **kw: any: _description_. Defaults to any.

      🎁:
          - drts: xyz🧭
          - dsts: xyz距离
          - cpXyz: 中点集
          - cns: 顶点
          - cp: 中心点

      🚱:
      📝:
          - cns:  0 ---P:3   OIPR(ozyx: 0134)
                / |    / |
              /  |  7/  |
            R:4  1 ---  2
                /I     /
                /      /
              5      6
      '''
      cpXyz      = Helper.getArr(cpXyz)
      cov     = np.cov(cpXyz,
                    y      = None,
                    rowvar = 0,
                    bias   = 1)
    #   vT      = np.transpose(npl.eig(cov)[1])
      vT      = npl.eig(cov)[1].T
      psR     = np.dot(cpXyz,
                    npl.inv(vT))

      cMin    = np.min(psR, axis=0)
      cMax    = np.max(psR, axis=0)

      cMax   -= cMin
      cMax   /= 2
      cMin   += cMax
      dx, dy, dz\
              = cMax
      cx, cy, cz\
              = cMin
      cn      = ndA([
                    [cx - dx, cy - dy, cz - dz],
                    [cx - dx, cy + dy, cz - dz],
                    [cx - dx, cy + dy, cz + dz],
                    [cx - dx, cy - dy, cz + dz],
                    [cx + dx, cy - dy, cz - dz],
                    [cx + dx, cy + dy, cz - dz],
                    [cx + dx, cy + dy, cz + dz],
                    [cx + dx, cy - dy, cz + dz],
                    ])
      cns     = np.dot(cn  , vT)
      cp      = np.dot(cMin, vT)
      zP, zDrt, zDst, _\
              = p2pLn(cns[0], cns[4]) # , mNam=['zLn'+mNam, mNam][int(mNam=='')])
      xP, xDrt, xDst, _\
              = p2pLn(cns[0], cns[1]) # , mNam=['xLn'+mNam, mNam][int(mNam=='')])
      yP, yDrt, yDst, _\
              = p2pLn(cns[0], cns[3]) # , mNam=['yLn'+mNam, mNam][int(mNam=='')])
      cpXyz      = ndA([xP, yP, zP])
      drts    = ndA([xDrt, yDrt, zDrt])
      dsts    = ndA([xDst, yDst, zDst])
    #   dsts    = [max(1., value) for value in dsts] # 距离小于1则=1
      dsts = [__builtins__.max(1., value) for value in dsts]
      print(f'{dsts=}')

    #   if xyzSort:
    #     inx   = np.argmax(abs(drts), axis=0)
    #     cpXyz    = cpXyz[inx]
    #     dsts  = dsts[inx]
    #   if mNam != "":
      rNod = roi(dsts, drts, cp, f"{mNam}_roi")
    #     roi(cp, [dsts[0], dsts[1], 1      ], drts, mNam='z'+mNam)
    #     roi(cp, [dsts[0], 1      , dsts[2]], drts, mNam='x'+mNam)
    #     roi(cp, [1      , dsts[1], dsts[2]], drts, mNam='y'+mNam)
      return    drts, dsts, cns, cp, cpXyz, rNod

    #tag roi 生成ROI
    def roi(dim,
            drts: PS,
            cp = None,
            mNam: str ="",
            **kw
            ):
      '''roi 生成ROI

      生成ROI

      Args:
          cp (PS): 中心点
          dim (float): 尺寸
          drts (PS): 方向
          mNam (str, optional): _description_. Defaults to "".

      Returns:
          - _description_

      Note:
          - _description_vbs = sitkUtils.PullVolumeFromSlicer(vbs)
            vbs = sitkUtils.PullVolumeFromSlicer(vbs)

      '''
      roi = SCENE.AddNewNodeByClass(
          "vtkMRMLMarkupsROINode",mNam)
    #   roi.SetName("ROI")
      roi.SetSize(*dim)
      if cp is None:
        cp = 0.5*np.dot(dim,drts)
    #   roi.SetCenter(cp)
      b2Rt = np.row_stack((np.column_stack((drts[0], drts[1], drts[2], cp)),
                          (0, 0, 0, 1)))
      b2RtMtx = ut.vtkMatrixFromArray(b2Rt)
      roi.SetAndObserveObjectToNodeMatrix(b2RtMtx)
    #   Helper.markDisp(roi, **kw)
      Helper.nodsDisplay(roi,
                          opacity=0.2,
                          color="cyan",
                          gDia=2,
                          lock=False
                          )
      return roi




    @staticmethod
    #tag plnGrid 面网格
    def plnGrid(pd, pln=None, flat=True):
      def __gps(ps, dr, dt):
        if ps.ndim == 2:
          ps = ps.reshape(len(ps), 1, 3)
        dst = dr * np.arange(1, dt)[:, np.newaxis]
        dst = dst.reshape(1, len(dst), 3)
        return ps + dst
      pArr     = Helper.getArr(pd)
      if pln is None:
        dr, dt, cns = Helper.obBx(pArr)[:3]
      xgps     = __gps(cns[0], dr[0], dt[0])
      xgps     = xgps.reshape(len(xgps) , 3)
      print(f'{xgps.shape = }')
    #   xgps     = xgps.reshape(len(xgps), 1, 3)
      gps      = __gps(xgps, dr[1], dt[1])
      print(f'{gps.shape=}')
      if flat:
        gps    = gps.reshape(-1, 3)
      return gps
    # ### readFileAsString
    # - readFileAsString

    @staticmethod
    def readFileAsString(fname):
        s = ""
        with open(fname, "r") as f:
            s = f.read()
        return s

    @staticmethod
    def ndA(data: PS):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return data
    # ### lsDic
    # - lsDic

    @staticmethod
    def lsDic(dic, lDic, upDate=False) -> dict:
        '''lsDic 彪子店

        通过dfDic建list并可实现list.append

        Args:
            dic (dict): 输入字典
            lDic (dict, optional): 源字典. Defaults to None.

        Returns:
            - 将`dic`的内容append到`lDic`(其中value为list)

        Note:
            - 当`dic`内有和`lDic'相同的key时,append list
            - 否则, 创建新的
            - 先要创建一个孔子店(e.g. dic = ())
            - 当len(value)=1时,取值时加[0]解包
        '''
        ls2Arr = lambda x: np.array([np.array(i[0]) for i in x])
        if not lDic:
            lDic = dfDic(list)
        if upDate: # 更新则清空lDic中和dic相同key的值后加入新数据
            for k, v in dic.items():
                if k in lDic.keys():
                    lDic[k] = []
        for k, v in dic.items():
            data = [(k,v)]
            for (key, value) in data:
                lDic[key].append(value)
        return lDic

    @staticmethod
    def mName(mNod: str) ->str:
        '''mName 显名

        nod显名

        Args:
            mNod (scr): mNod

        Returns:
            - mNod

        Note:
            -
        '''
        # mNod
        return mNod if mNod != "" else mNod

# ### projPla
# - [[projPla]]
    @staticmethod
    def projPla(ps: np.ndarray,
                pm: np.ndarray =[0,0,0],
                norm: np.ndarray =[0,1,1],
                ) -> np.ndarray:
        ''' projPla 投影到平面
        将点投影到平面
        参:
            ps: 点集 (n,3)
            pm: 平面上一点 (3,)
            norm: 平面法向量 (3,)
        返:
            投影点 (n,3)
        '''
        ps      = np.asarray(ps)
        pm      = np.asarray(pm)
        norm    = np.asarray(norm)
        d       = np.dot((    ps
                            - pm),
                            norm)\
                / np.dot(   norm,
                            norm)
        p_proj  = ps\
                - d * norm
        return np.asarray(p_proj)

    # ### [[色术]]
    # - myColor

    @staticmethod
    def myColor(colorName):
        if isinstance(colorName, list):
            colorArr = colorName
        else:
            if colorName == "red":
                colorArr = [1, 0, 0]
            elif colorName == "green":
                colorArr = [0, 1, 0]
            elif colorName == "blue":
                colorArr = [0, 0, 1]
            elif colorName == "black":
                colorArr = [0, 0, 0]
            elif colorName == "white":
                colorArr = [1, 1, 1]
            elif colorName == "yellow":
                colorArr = [1, 1, 0]
            elif colorName == "pink":
                colorArr = [1, 0, 1]
            elif colorName == "cyan":
                colorArr = [0, 1, 1]
            elif colorName == "purple":
                colorArr = [0.6,  0.1,  0.6]
        return colorArr

    # ### nodesDisplay
    # - [[nodesDisplay]]

    @staticmethod
    def nodsDisplay(nods, modCls=None, **kw):
        """Model Display

        Param: Glyph = 0, 1-StarBurst2D, 2-Cross2D,3-CrossDot2D, 4-ThickCross2D, 5-Dash2D, 6-Sphere3D, 7-Vertex2D, 8-Circle2D,9-Triangle2D, 10-Square2D, Diamond2D, Arrow2D, ThickArrow2D,
        HookedArrow2D, GlyphType_Last

        """
        nodes = Helper.nameNods(nods)
        # if not isinstance(nodes, tuple):
        #     nodes = tuple(tuple)
        # for key in nodes.keys():
        for node in nodes:
            Helper.nodsDisp(node, 1)
            if not modCls:
                modCls,_ = Helper.getCls(node)
            # node.SetDisplayVisibility(1)
            if modCls == "mod":
                Helper.modDisp(node, **kw)
            elif modCls == "mark":
                Helper.markDisp(node, **kw)
            # elif modCls == "vol":
            #     Helper.markDisp(node, **kw)
        return nodes
    @staticmethod
    def getCls(nod=None, cls=None):
        """
        """
        dic = dict(mod = MOD,
                   mark = "vtkMRMLMarkup",
                #    vol = "vtkMRMLScalarVolumeNode"
                  )
        if cls:
            return cls, dic[cls]
        else:
            nodeCls = nod.GetClassName()
            for k, v in dic.items():
                if nodeCls.startswith(v):
                    return k, v
                elif nodeCls == v:
                    return k, v


    @staticmethod
    def nodsDisp(nods="", display=0):
        """Hide or disp
        parame:
            - nods: "marks","mods","",nods(str or nods)
            - display: False-->hide
        return:
        """
        def __disP(nods="", disp=display):
            nodes = Helper.nameNods(nods)
            #     nods.SetDisplayVisibility(display)
            for nod in nodes:
                if isinstance(nod, NOD):
                    nod.SetDisplayVisibility(disp)
            return
        if nods == "":
            __disP("vtkMRMLMarkups*")
            __disP(f"{MOD}*")
        elif nods == "marks":
            __disP("vtkMRMLMarkups*")
        elif nods == "mods":
            __disP(f"{MOD}*")
        else:
            __disP(nods, disp=display)
        return True

    @staticmethod
    def nameNods(nods: Union[str,  NOD],
                 cls=None
                 ) -> dict:
        """由nods返回nodes
        """
        nodsDic = collections.OrderedDict()
        if cls:
            _, nods = Helper.getCls(cls=cls) + "*"
            nodes = ut.getNodes(nods, useLists=True)
        else:
            if isinstance(nods, str):
                nodes = ut.getNodes(nods, useLists=True)
            else:
                nodes = collections.OrderedDict()
                nodes[nods.GetName()] = [nods]
        if len(nodes.values()) == 1:
            return list(nodes.values())[0]
        else:
            for k, node in nodes.items():
                if isinstance(node[0],  NOD):
                    n = tuple(node) # [0]
                    if len(n) > 1:
                        for nod in n:
                            nodsDic[nod.GetID()] = nod
                    else:
                        nodsDic[node[0].GetID()] = node[0]

            return list(nodsDic.values())

    @staticmethod
    def nameNodCls(nod, cls):
        if nod == "":
            return
        node = Helper.namNod(nod)
        nodCls,_ = Helper.getCls(node)
        if nodCls != cls:
            print("nod is not {cls}!")
            return 0
        return node
                # Hide measurement result while markup up

    @staticmethod
    def perMarkDic(markDn, **kw):
        mDic = dict(gType=markDn.GetGlyphType(),
                    SelectedColor = markDn.GetSelectedColor(),
                    color = markDn.GetColor(),
                    GlyphScale = markDn.GetGlyphScale(),
                    gDia = markDn.GetGlyphSize(),
                    texScal = markDn.GetTextScale(),
                    opacity = markDn.GetOpacity(),
                    SliceProjection = markDn.GetSliceProjection(),
                    SliceProjectionColor = markDn.GetSliceProjectionColor(),
                    SliceProjectionOpacity = markDn.GetSliceProjectionOpacity(),
                    lDia = markDn.GetLineDiameter(),
                   )

        perDic = {}
        for k,v in mDic.items():
            key = k
            k = kw.get(f"{key}", v)
            perDic[f"{key}"] = k
        return perDic
    @staticmethod
    def markDisp(nod, lock=1, handle=0, **kw):
        # if mNam != "":
        nod = Helper.nameNodCls(nod, "mark")
        nod.CreateDefaultDisplayNodes()
        markDn = nod.GetDisplayNode()
        perDic = Helper.perMarkDic(markDn, **kw)
        markDn.SetSelectedColor(Helper.myColor(perDic["SelectedColor"]) if isinstance(perDic["SelectedColor"], str) else perDic["SelectedColor"])
        markDn.SetGlyphType(perDic["gType"])
        markDn.SetCurveLineSizeMode(1)
        markDn.SetLineDiameter(perDic["lDia"])
        # markDn.UseGlyphScaleOn()
        markDn.UseGlyphScaleOff()
        markDn.SetGlyphSize(perDic["gDia"])
        # markDn.GetTextProperty().SetFontSize(perDic["FontSize"])

        markDn.SetTextScale(perDic["texScal"])
        markDn.SetOpacity(perDic["opacity"])
        slicer.modules.markups.logic().SetAllControlPointsLocked(nod,lock)
        markDn.SetHandlesInteractive(handle)
        markDn.SetTranslationHandleVisibility(handle)
        markDn.SetRotationHandleVisibility(handle)
        markDn.SetScaleHandleVisibility(handle)

    @staticmethod
    def perModDic(modDn, **kw):
        mDic = dict(color = modDn.GetColor(),
                    vis2D = modDn.GetVisibility2D(),
                    vis3D = modDn.GetVisibility3D(),
                    opacity = modDn.GetOpacity(),
                    lineW = modDn.GetLineWidth(),
                    egVis = modDn.GetEdgeVisibility(),
                    egC = modDn.GetEdgeColor(),
                    Lighting = modDn.GetLighting(),
                    projS = modDn.GetSliceDisplayMode()
                    )

        perDic = {}
        for k,v in mDic.items():
            key = k
            k = kw.get(f"{key}", v)
            perDic[f"{key}"] = k
        return perDic

    @staticmethod

    def pdMDisp(polydata, mNam = '', **kw):
        """polydata to model
        """
        if mNam=='':
            return polydata
        if isinstance(polydata, VPD):
            model = SCENE.AddNewNodeByClass(MOD, mNam)
            model.SetAndObservePolyData(polydata)
            Helper.modDisp(model, **kw)
            return model
        elif isinstance(polydata, vtk.vtkImageData):
            vol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            vol.SetIJKToRASMatrix(slicer.vtkMRMLSliceNode.GetXYToRAS())
            vol.SetAndObserveImageData(polydata)
            vol.CreateDefaultDisplayNodes()
            vol.CreateDefaultStorageNode()


    @staticmethod
    def modDisp(modNod,  mNam='', **kw):
        """
        """
        modNod = Helper.nameNodCls(modNod, mNam)
        if modNod == 0:
            return
        modNod.CreateDefaultDisplayNodes()
        modDn = modNod.GetDisplayNode()
        dnDic = Helper.perModDic(modDn, **kw)

        modDn.SetSliceDisplayMode(dnDic["projS"])
        modDn.SetColor(Helper.myColor(dnDic["color"]) if isinstance(dnDic["color"], str) else dnDic["color"])
        modDn.SetVisibility2D(dnDic["vis2D"])
        modDn.SetVisibility3D(dnDic["vis3D"])
        modDn.SetOpacity(dnDic["opacity"])
        modDn.SetLineWidth(dnDic["lineW"])
        modDn.SetEdgeVisibility(dnDic["egVis"])
        modDn.SetEdgeColor(Helper.myColor(dnDic["egC"]) if isinstance(dnDic["egC"], str) else dnDic["egC"])
        modDn.SetLighting(dnDic["Lighting"])

    # ### nod2Arr
    #
    @staticmethod
    def getArr(nod: Union[PS, NOD, VPD],
               order: bool = False,
               segLen: int = 1
               )-> np.ndarray:
        """
        获取array
        """
        def __reOrd(arr, order, segLen):
            arr = arr[0::segLen]
            if order:
                return np.concatenate((arr[0::2],arr[1::2][::-1]),axis=0) # 排序
            else:
                return arr
        if isinstance(nod, VPD):
            pd = nod.GetPoints().GetData()
            arr = vtk_to_numpy(pd)
            return __reOrd(arr, order, segLen)
        elif isinstance(nod, str):
            arr = ut.array(nod)
            return __reOrd(arr, order, segLen)
        elif isinstance(nod, NOD):
            arr = ut.array(nod.GetID())
            return __reOrd(arr, order, segLen)
        elif isinstance(nod, list):
            arr = np.asarray(nod)
            return __reOrd(arr, order, segLen)
        elif isinstance(nod, np.ndarray):
            arr = nod
            return __reOrd(arr, order, segLen)

    # ### p20exLine
    # - [[p2pexLine]]

    @staticmethod
    #tag p2pNor 点对点生成单位向量和距离
    def p2pNor(p0, pN = None):
      p0     = ndA(p0)
      n      = len(p0)
      if pN is not None:
        pN   = ndA(pN)
        if np.equal(p0, pN).all():
          raise ValueError('同一个点, 无法计算!')
        dst = npl.norm(pN - p0) # , axis=-1, keepdims=True)
        print(dst.shape)

      # DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      # roi.SetSize(*dim

        return (pN - p0)/dst, dst
      else:
        dst = npl.norm(p0) #     , axis=-1, keepdims=True)
        return   p0/dst+eps, dst

    @staticmethod
    # eps = 1e-6
    def p2pLn(p: PS,
        pV: PS       = None,
        plus: float  = eps,
        mNam: str = '',
        dia: float   = 1.,
        **kw
              ):

      """
      计算两点之间的线段，并可在3D视图中显示。

      Args:
        p      : 一个长度为3的数组，表示起点坐标。
        pV     : 一个长度为3的数组，表示终点坐标。默认为None，表示终点坐标为原点。
        plus   : 一个浮点数，表示线段长度的增量。默认为eps。
        mNam: 一个字符串，表示在3D视图中显示的线段名称。默认为空字符串。
        dia    : 一个浮点数，表示线段的宽度。默认为1。
        **kw   : 其他参数。

      Returns:
        pp     : 一个长度为3的数组，表示线段的终点坐标。
        pV     : 一个长度为3的数组，表示起点到终点的单位向量。
        dst    : 一个浮点数，表示起点到终点的距离。
        lambda x: p + pV * x: 一个函数，输入一个浮点数x，返回线段上距离起点x的点的坐标。

      Raises:
        Exception: 如果起点和终点是同一个点，则抛出异常。
      """
      p     = ndA(p)*1.
      pV    = ndA(pV)*1.
      pV   -= p
      nor, dst   = Helper.p2pNor(pV) # , axis=[None,1][isPs])
      if dst <= eps:
          raise Exception("p和pV是同一个点")
      dst  += plus
      pp    = p + nor * dst
      if mNam != "":
          marksNod = SCENE.AddNewNodeByClass(
              "vtkMRMLMarkupsLineNode", mNam
          )
          marksNod.AddControlPoint(vtk.vtkVector3d(p))
          marksNod.AddControlPoint(vtk.vtkVector3d(pp))
          Helper.markDisp(marksNod, lDia = dia, **kw)
      return  pp, nor, dst, lambda x: p + nor * x

    # - line2Cyl

    @staticmethod
    def line2Cyl(lNod, line2Cyl=False, Dia=None, Seg=12, mNam = ''):
        """Take the line coordinates

        Para: mNam

        return:  pos1, pos2, Dia, length, drt,modNode
        """
        p0, p1 = Helper.getArr(lNod)
        length = Helper.p2pDst(p1,  p0)
        drt = (p1 - p0) / length
        lNod = Helper.namNod(lNod)
        dn = lNod.GetDisplayNode()
        if Dia is None:
            Dia = dn.GetLineDiameter()
        if mNam == '':
            ln2CyName = f"l2Cyl_{mNam}"
        else:
            ln2CyName = mNam
        if line2Cyl is True:
            # Helper.nodesDel(nodName)
            Helper.nodsDel(lNod)
            lNod = Helper.p2pCyl(
                p0, p1, rad=Dia/2, mNam=ln2CyName, Seg=Seg)
        return p0, p1, Dia, length, drt, lNod

    @staticmethod
    def addFids(arr, mNam="N", Dia = .3, lableName = "", **kw):
        '''addFids 加点s

        添加Fids

        Args:
            data (arr or list): 点坐标
            mNam (str, optional): 莫名. Defaults to "N".
            Dia (float, optional): 直径. Defaults to .3.
            lableName (str, optional): 标签,默认None标签自动排序. Defaults to None.

        Returns:
            - fidList Node
        '''
        #   lableName="1"
        # if data is not None:p
        #     # this_type_str = type(arr)
        arr = ndA(arr)
        if not bool(ut.getNodes(mNam)):  # 没有名
            fidList = slicer.vtkMRMLMarkupsFiducialNode()
            fidList.SetName(mNam)
        else:  # 有名
            fidList = ut.getNode(mNam)
            if not isinstance(fidList, slicer.vtkMRMLMarkupsFiducialNode):
                fidList = slicer.vtkMRMLMarkupsFiducialNode()
                fidList.SetName(mNam)
        fidList.AddFiducialFromArray(arr)
        fidList = SCENE.AddNode(fidList)
        Helper.markDisp(fidList, mNam, gDia = Dia, **kw)
        return fidList
    # ### [[两点平面聚焦术]]
    # - P2SliceFocus

    @staticmethod
    def P2SliceFocus(p1, p2, p3=0):
        P1 = p1 + [0, 0, 0.1]
        P2 = p2 + [0.1, 0, 0]

        if p3 == 0:
            P3 = p1 + [0, 0, 0.1]
        else:
            P3 = p3 + [0, 0, 0.1]
        Helper.p2pCyl(
            P3, p2, 10, plus=0.1 - Helper.p2pDst(p2,  P3), Seg=3, mNam="cyl"
        )
        P4 = Helper.cyl2Line("cyl")[2][0] + [0.1, 0, 0]
        Helper.nodsDel("cyl")

        redSD = [P1, P2, [P1[0], P2[1], P2[2]]]
        yellowSD = [[P2[0], P2[1], P1[2]], P1, P2]
        greenSD = [P3, [P3[0], P3[1], P4[2]], P4]

        points = np.array([list(zip(*redSD))])[0]
        sliceNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()
        plnPos = points.mean(axis=1)
        plnNor = Helper.p3Nor(points[:, 1] , points[:, 2] , points[:, 0], False)
        #   (ppln1, ppln2) = (1, 2)
        plnX = points[:, 1] - points[:, 2]
        sliceNode.SetSliceToRASByNTP(
            plnNor[0],
            plnNor[1],
            plnNor[2],
            plnX[0],
            plnX[1],
            plnX[2],
            plnPos[0],
            plnPos[1],
            plnPos[2],
            0,
        )

        ypoints = np.array([list(zip(*yellowSD))])[0]
        ysliceNode = slicer.app.layoutManager().sliceWidget("Yellow").mrmlSliceNode()
        yplnPos = ypoints.mean(axis=1)
        yplnNor = Helper.p3Nor(ypoints[:, 1] , ypoints[:, 2] , ypoints[:, 0], False)
        #   (ppln1, ppln2) = (2, 1)
        yplnX = ypoints[:, 1] - ypoints[:, 2]
        ysliceNode.SetSliceToRASByNTP(
            yplnNor[0],
            yplnNor[1],
            yplnNor[2],
            yplnX[0],
            yplnX[1],
            yplnX[2],
            yplnPos[0],
            yplnPos[1],
            yplnPos[2],
            0,
        )

        gpoints = np.array([list(zip(*greenSD))])[0]
        gsliceNode = slicer.app.layoutManager().sliceWidget("Green").mrmlSliceNode()
        gplnPos = gpoints.mean(axis=1)
        gplnNor = Helper.p3Nor(gpoints[:, 1] , gpoints[:, 2] , gpoints[:, 0], False)
        #   (ppln1, ppln2) = (2, 1)
        gplnX = gpoints[:, 1] - gpoints[:, 2]
        gsliceNode.SetSliceToRASByNTP(
            gplnNor[0],
            gplnNor[1],
            gplnNor[2],
            gplnX[0],
            gplnX[1],
            gplnX[2],
            gplnPos[0],
            gplnPos[1],
            gplnPos[2],
            0,
        )

# ### p3Cone
# - [[p3Cone]]
    @staticmethod
    def p3Cone(bP, drt=None, mNam='',rad=1, high=3, seg=12, hP=None,rP=None, *kw):
        # 设置bP为椎底坐标, rP为椎底圆边任一点, hP椎顶坐标
        if drt is None:
            rad = npl.norm(rP-bP)
            high = npl.norm(hP-bP)
            drt = (hP-bP)/high
        cone = vtk.vtkConeSource()
        cone.SetResolution(100)
        cone.SetCenter(bP)
        cone.SetRadius(rad)
        cone.SetHeight(high)
        cone.SetDirection(drt)
        cone.Update()
        return Helper.pdMDisp(cone.GetOutput(), mNam, *kw)

    @staticmethod
    def cnnExVol(vol, lbl):
        """
        Extrac the largest connected region of a vtk image

        Args:
            vtk_im: vtk image
            label_id: id of the label
        Return:
            new_im: processed vtk image
        """

        fltr = vtk.vtkImageConnectivityFilter()
        fltr.SetScalarRange(lbl, lbl)
        fltr.SetExtractionModeToLargestRegion()
        fltr.SetInputData(vol)
        fltr.Update()
        new_im = fltr.GetOutput()
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        py_im = vtk_to_numpy(vol.GetPointData().GetScalars())
        py_mask = vtk_to_numpy(new_im.GetPointData().GetScalars())
        mask = np.logical_and(py_im==lbl, py_mask==0)
        py_im[mask] = 0
        vol.GetPointData().SetScalars(numpy_to_vtk(py_im))
        return vol

    # ### [[2点棱柱术]]
    # - p2pCyl

    @staticmethod
    def p2pCyl(startPoint,
               endPoint=None,
               rad: float =1.,
               mNam: str ="Cyl",
               plus: float =0,
               drt: PS=None,
               Seg: int =12,
               RotY=0,
               Tx=0,
               Tz=0,
               ) -> any:
        """点对点棱柱"""
        startPoint = ndA(startPoint)
        if endPoint is None:
            endPoint = Helper.p2pLn(startPoint, drt=drt,plus=plus)[0]
        else:
            endPoint = ndA(endPoint)
        cylinderSource = vtk.vtkCylinderSource()
        cylinderSource.SetRadius(rad)
        cylinderSource.SetResolution(Seg)

        # Generate a random start and end point
        # startPoint = [0] * 3
        # endPoint = [0] * 3

        length, matrix = Helper.crtMatx(startPoint, endPoint, plus, drt)
        # Apply the transforms
        pd = Helper.cylTrans(startPoint, RotY, Tx, Tz, cylinderSource, length, matrix)
        # vtkNode = slicer.modules.models.logic().AddModel(pd)
        # vtkNode.SetName(mNam)
        # if transed:
        #     return
        # else:
        return Helper.pdMDisp(pd, mNam)

    @staticmethod
    def crtMatx(startPoint, endPoint, plus=0, drt=None):
        # 创建一个用于产生随机数的序列
        rng = vtk.vtkMinimalStandardRandomSequence()
        # 设置一个用于测试的固定种子
        rng.SetSeed(8775070)
        # 计算基础坐标
        normalizedX = [0] * 3  # x坐标
        normalizedY = [0] * 3  # y坐标
        normalizedZ = [0] * 3  # z坐标
        # 用结束坐标减去起始坐标，计算x坐标
        vtk.vtkMath.Subtract(endPoint, startPoint, normalizedX)
        # 计算x坐标的长度，若有方向参数，则加上plus
        length = [vtk.vtkMath.Norm(normalizedX) + plus, plus][int(drt is not None)]
        # 用正则化函数归一化x坐标
        vtk.vtkMath.Normalize(normalizedX)
        # 使用一个任意的向量来计算z坐标，向量的三个坐标从-10到10中随机取值
        arbitrary = [0] * 3
        for i in range(0, 3):
            rng.Next()
            arbitrary[i] = rng.GetRangeValue(-10, 10)
        # 计算z坐标，用x和任意向量的叉乘
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        # 用正则化函数归一化z坐标
        vtk.vtkMath.Normalize(normalizedZ)
        # 计算y坐标，用z和x的叉乘
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
        # 创建一个阵列用于构造方向余弦阵
        matrix = vtk.vtkMatrix4x4()
        # 创建方向余弦阵
        matrix.Identity()
        for i in range(0, 3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])
        # 返回x坐标的长度和方向余弦阵
        return length, matrix


    @staticmethod
    def cylTrans(startPoint, RotY, Tx, Tz, cylinderSource, length, matrix):
        transform = vtk.vtkTransform()
        transform.Translate(startPoint)  # translate to starting point
        transform.Concatenate(matrix)  # apply drt cosines
        transform.RotateZ(-90.0)  # align cylinder to x axis
        transform.Scale(1.0, length, 1.0)  # scale along the height vector
        transform.Translate(0, 0.5, 0)  # translate to start of cylinder
        transform.RotateY(RotY)
        # transform.Translate(Tx, 0, Tz)

        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(cylinderSource.GetOutputPort())
        transformPD.Update()
        pd = transformPD.GetOutput()
        return pd

    @staticmethod

    def angle3P(p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0
        return np.arccos(np.dot(v1, v2) / (npl.norm(v1) * npl.norm(v2))) * 180. / np.pi

    @staticmethod

    def modsBool(modA, operation, modB, modOut="modBool"):
        '''
        Param:Add:A+B;Sub:A-B;Ins:A&B
        '''
        import vtkSlicerCombineModelsModuleLogicPython as vtkbool
        boolFilter = vtkbool.vtkPolyDataBooleanFilter()

        ModA = Helper.namNod(modA)
        ModB = Helper.namNod(modB)

        if operation == '+':
            boolFilter.SetOperModeToUnion()
        elif operation == 'x':
            boolFilter.SetOperModeToIntersection()
        elif operation == '-':
            boolFilter.SetOperModeToDifference()
        else:
            raise ValueError("Invalid operation: " + operation)

        boolFilter.SetInputConnection(0, ModA.GetPolyDataConnection())

        boolFilter.SetInputConnection(1, ModB.GetPolyDataConnection())

        boolFilter.Update()

        outputModel = SCENE.AddNewNodeByClass(MOD)
        outputModel.SetName(modOut)

        outputModel.SetAndObservePolyData(boolFilter.GetOutput())
        outputModel.CreateDefaultDisplayNodes()
        # The filter creates a few scalars, don't show them by default,
        # as they would be somewhat distracting
        outputModel.GetDisplayNode().SetScalarVisibility(False)
        return outputModel

    @staticmethod
    def namNod(namNod: Union[str, NOD])-> NOD:
        ''
        # return_list
        if isinstance(namNod,str):
            namNod=ut.getNode(namNod)
        return namNod

    # ### [[返棱柱术]]
    # - cyl2Line

    @staticmethod
    def cyl2Line(mNam = "", nodName="", cyl2line=False, changeCyl=False, r=0, plus=0, RotY=0, Seg=12):
        """获取棱柱的数据; 转线; 变棱柱

        Param: mNam

        return: 0. np.array([axisMed0, axisMed1]), 1. np.around(rad,2), 2. np.array([P0,P1]), 3. NumP // 4, 4. modNode[1](drt), 5. modNode

        """
        modNode = Helper.namNod(mNam)
            # 读取节点(module)
        sr = modNode.GetPolyData()  # module转多边形

        numP = sr.GetNumberOfPoints()//2  # 多面体的点数
        ps = Helper.getArr(modNode)[numP:]
        P1s = ps[numP//2:]
        P0s = ps[:numP//2]
        axisMed0 = np.mean(P1s, axis=0)
        axisMed1 = np.mean(P0s, axis=0)
        rad = Helper.p2pDst(axisMed0,  P0s[0])
        Drt = Helper.p2pLn(axisMed0, axisMed1, plus=0)[1]
        if cyl2line is True:
            modNode = Helper.p2pLn(
                    axisMed0, axisMed1, mNam=nodName, plus=0, Dia=rad*2)[-1]
        if changeCyl is True:
            Helper.nodsDel(modNode)
            rad = rad if r == 0 else r
            modNode = Helper.p2pCyl(
                axisMed0, axisMed1, rad, mNam=f"nodName", plus=plus, RotY=RotY, Seg=Seg)
        return np.array([axisMed0, axisMed1]), np.around(rad, 2), np.array([P0s, P1s]), numP // 4, Drt, modNode

    # ### [[删节点术]]
    # - delNodes

    @staticmethod
    def nodsDel(nods=None, cls=None):
        """delete nodes"""
        if cls:
            nodes = Helper.nameNods(cls=cls)
        else:
            nodes = Helper.nameNods(nods)
        for nod in nodes:
            SCENE.RemoveNode(nod)
    # ### [[标点成阵术]]
    # - Pdata

    @staticmethod
    def Pdata(fidNode="T"):
        """Read the value of the fidlist"""
        fidList = ut.getNode(fidNode)
        if fidList.GetClassName() == "vtkMRMLMarkupsFiducialNode":
            numFids = fidList.GetNumberOfFiducials()
            Mdata = [0, 0, 0]
            for i in range(numFids):
                ras = [0, 0, 0]
                #         Mdata = [0,0,0]
                fidList.GetNthFiducialPosition(i, ras)
                if i == 0:
                    Mdata = np.array(ras)
                else:
                    Mdata = np.append(Mdata, np.array(ras), axis=0)
            data = np.array(Mdata).ravel()
            if len(data) > 3:
                data = np.reshape(data, (len(data)//3, 3))
        elif fidList.GetClassName() == "vtkMRMLMarkupsCurveNode":
            data = vtk_to_numpy(fidList.GetCurvePointsWorld().GetData())
        # zcount = int(data.size / (3 * groups))
        # data = np.array(data.reshape(zcount, (groups, 3)))
        # # ras = [-1, -1, 1] * data  # ras coordinate
        return data

    # ### [[otThred]]
    # - otThred

    @staticmethod
    def otThred(vNod):

        vNod = Helper.namNod(vNod)
        skImg = sitkUtils.PullVolumeFromSlicer(vNod.GetID())
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        seg = otsu_filter.Execute(skImg)
        return otsu_filter.GetThreshold()
    @staticmethod
    def boneSeg(vol, mNam='skBone'):
        vol = Helper.namNod(vol)
        S_img = sitkUtils.PullVolumeFromSlicer(vol.GetID())
        # 中值滤波
        median_filter = sitk.MedianImageFilter()
        median_filter.SetRadius(1)
        M_img = median_filter.Execute(S_img)

        # 闭运算
        cFilter = sitk.GrayscaleMorphologicalClosingImageFilter()
        cFilter.SetKernelRadius(3)
        c_img = cFilter.Execute(M_img)
        stFilter = sitk.SubtractImageFilter()
        st_img = stFilter.Execute(c_img, M_img)
        st2_img = stFilter.Execute(M_img, st_img)
        sgFilter = sitk.SmoothingRecursiveGaussianImageFilter()
        sgFilter.SetSigma(1) # 设置sigma
        sgImg = sgFilter.Execute(st2_img) # 将vol进行Otsu阈值分割

        st2Arr =  sitk.GetArrayFromImage(st2_img)
        st2Arr[st2Arr <= 100] = 360 # 将st2Arr中小于100的像素值设置为100
        # 用st2Arr生成新的sitk image, 但是要求image的origin, spacing, direction必须和st2_img一致
        stImg = sitk.GetImageFromArray(st2Arr, isVector=False)
        stImg.SetOrigin(st2_img.GetOrigin())
        stImg.SetSpacing(st2_img.GetSpacing())
        stImg.SetDirection(st2_img.GetDirection())

        osFilter = sitk.OtsuMultipleThresholdsImageFilter()
        # osFilter.SetNumberOfThresholds(2) # 设置阈值个数
        osFilter.SetValleyEmphasis(True) #
        ossImg = osFilter.Execute(sgImg) # 执行分割
        # vols = []
        # for i in range(1, 3):
        #     ths = sitk.ThresholdImageFilter()
        #     ths.SetUpper(i)
        #     ths.SetLower(i)
        #     osImg = ths.Execute(ossImg)
            # osImgs.append(osImg)
        cnnFilter = sitk.ConnectedComponentImageFilter()
        cnnImg = cnnFilter.Execute(ossImg)

        fhFilter = sitk.BinaryFillholeImageFilter()
        fhFilter.SetForegroundValue(1)
        fhImg = fhFilter.Execute(cnnImg)
        gdImg = sitk.InvertIntensity(fhImg,1)

        return sitkUtils.PushVolumeToSlicer(fhImg, name=mNam), sitkUtils.PushVolumeToSlicer(gdImg, name=mNam+'gd')
    # ### [[幻影术]]
    # - skinSur

    @staticmethod
    def segMod(vNod,
                mNam: str="boneTempt",
                minThred: float=1.,
                color: str="blue",
                opacity: float=0.8,
                skV: bool=True,
                edit: bool=False,
                # invert: bool=False
                ):
        # if skV:
        #     vNod, gdNod = Helper.boneSeg(vNod, mNam+'_sk')
        #     print(f"{vNod=}")
        #     # vNod是一个volume节点
        # else:
        #     vNod = Helper.namNod(vNod)
            #    minThred = 100
        # Create segmentation
        vNod = Helper.namNod(vNod)
        segNods = SCENE.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segNods.CreateDefaultDisplayNodes()  # only needed for display
        segNods.SetReferenceImageGeometryParameterFromVolumeNode(vNod) # 设置参考图像
        segId = segNods.GetSegmentation().AddEmptySegment("skin")
        maximumHoleSizeMm = .1

        # Create segment editor to get access to effects
        segEd = slicer.qMRMLSegmentEditorWidget()
        segEd.setMRMLScene(SCENE)
        segEdNod = SCENE.AddNewNodeByClass(
            "vtkMRMLSegmentEditorNode"
        )
        segEd.setMRMLSegmentEditorNode(segEdNod)
        segEd.setSegmentationNode(segNods)
        segEd.setSourceVolumeNode(vNod)

        # Thresholding
        segEd.setActiveEffectByName("Threshold")
        effect = segEd.activeEffect()
        effect.setParameter("MinimumThreshold", minThred)
        #   effect.setParameter("MaximumThreshold","695")
        effect.self().onApply()

        segEd.setActiveEffectByName("Smoothing")
        effect = segEd.activeEffect()
        effect.setParameter("SmoothingMethod", "MEDIAN")
        effect.setParameter("KernelSizeMm", 1)
        effect.self().onApply()
        if edit is True:
            # Invert the segment
            segEd.setActiveEffectByName("Logical operators")
            effect = segEd.activeEffect()
            effect.setParameter("Operation", "INVERT")
            effect.self().onApply()
            # Remove islands in inverted segment (these are the holes inside the segment)

            # Smoothing
            segEd.setActiveEffectByName("Smoothing")
            effect = segEd.activeEffect()
            effect.setParameter("SmoothingMethod", "MEDIAN")
            effect.setParameter("KernelSizeMm", 1)
            effect.self().onApply()

            # Invert the segment
            segEd.setActiveEffectByName("Logical operators")
            effect = segEd.activeEffect()
            effect.setParameter("Operation", "INVERT")
            effect.self().onApply()

            # Smoothing
            segEd.setActiveEffectByName("Smoothing")
            effect = segEd.activeEffect()
            effect.setParameter("SmoothingMethod", "MEDIAN")
            effect.setParameter("KernelSizeMm", 1)
            effect.self().onApply()

            # Islands
            segEd.setActiveEffectByName("Islands")
            effect = segEd.activeEffect()
            effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
            effect.self().onApply()

            # Margin
            segEd.setActiveEffectByName("Margin")
            effect = segEd.activeEffect()
            effect.setParameter("MarginSizeMm", str(maximumHoleSizeMm))
            effect.self().onApply()    # # Add it to the allVertebraeSegment


            # if invert:
            #     # Invert the segment

        meshPd = Helper.surMesh(segNods, segId)
        # segmentEditorWidget.setActiveEffectByName("Logical operators")
        # effect = segmentEditorWidget.activeEffect()
        # effect.setParameter("Operation", "INVERT")
        # effect.self().onApply()
        # invertMesh = Helper.surMesh(segNod, addedSegmentID)

        modNod = slicer.modules.models.logic().AddModel(meshPd)
        SCENE.RemoveNode(segNods)
        segEd = None
        SCENE.RemoveNode(segEdNod)
        # segmentationNode.GetDisplayNode().SetVisibility(False)
        if mNam != "":
            modNod.SetName(mNam)
            modelDisplay = modNod.GetDisplayNode()
            modelDisplay.SetColor(Helper.myColor(color))  # yellow
            modelDisplay.SetOpacity(opacity)
            modelDisplay.SetBackfaceCulling(0)
            # modelDisplay.SetVisibility(1)
            modelDisplay.SetVisibility2D(True)

        return modNod, Helper.modObb(meshPd)

    @staticmethod
    def surMesh(segNod, addedSegmentID=None):
        if addedSegmentID is not None:
            addedSegmentID = segNod.GetSegmentation().AddEmptySegment("segNode")
        segNod.CreateClosedSurfaceRepresentation()
        polyData = VPD()
        surfaceMesh = segNod.GetClosedSurfaceRepresentation(
            addedSegmentID, polyData)
        normals = vtk.vtkPolyDataNormals()
        normals.AutoOrientNormalsOn()
        normals.ConsistencyOn()
        normals.SetInputData(polyData)
        normals.Update()
        surfaceMesh = normals.GetOutput()
        # SCENE.RemoveNode(segNod)
        return surfaceMesh

    # ### [[射线穿透术]]
    # - modObb

    @staticmethod
    def modObb(mNod: Union[str, VPD, MOD, OB])-> OB:

        # get model"tempt'
        if not isinstance(mNod,OB):
            if not isinstance(mNod, VPD):
                mNod = Helper.namNod(mNod)
                if not isinstance(mNod,  SMD):
                    return False
                mNod = mNod.GetPolyData()  # node to PolyData
        obbtree = vtk.vtkOBBTree()
        obbtree.SetDataSet(mNod)
        obbtree.BuildLocator()
        return obbtree

    @staticmethod
    def modIn(p1: PS=None,
              p2: PS=None,
              oNod: Union[MOD,OB]='',
              drt: PS=None,
              plus: float=0.,
              mNam: str=""
              )-> Tuple[int, vtk.vtkPoints]:
        '''modIn 判断p0位于mod内外

        判断p0位于mod内外

        Args:
            p0 (PS): 源点
            p (PS): 靶点
            oNod (Union[MOD,OB]): 模特
            plus (float, optional): _description_. Defaults to 0..

        Returns:
            - code: -1->内;0->射线无交点;1->外
            - sectP: 交点

        Note:
            - _description_
        '''
        # get model"tempt'
        if not isinstance(oNod, OB):
            oNod = Helper.modObb(oNod)
        if drt is not None and plus==0.:
            plus = 60.
        if plus >= 0.:
            p2 = Helper.p2pLn(p1, p2, plue=plus, mNam=mNam)[0]
        else: # 若plus<0,则p1,p2位置互换
            p_2 = np.copy(p1)
            if drt:
                p1 = Helper.p2pLn(p1, drt, plue=abs(plus), mNam=mNam)[0]
            else:
                p1 = Helper.p2pLn(p1, p2, plue=abs(plus), mNam=mNam)[0]
            p2 = p_2
        sectP = vtk.vtkPoints()
        return oNod.IntersectWithLine(p1, p2, sectP, None), sectP

    # ### [[射线穿透术]]
    # - rayCast

    @staticmethod
    def rayCast(p1: PS=None,
                p2: PS=None,
                oNod: Union[MOD,OB]='',
                drt: PS=None,
                plus: float=0.,
                oneP: bool=True,
                mNam: str=""
                # inP=False
                ): # , cell=False, ):
        '''

        '''
        code, sectP = Helper.modIn(p1, p2, oNod, drt, plus, mNam)
        if code == 0:
            Helper.addLog(f"No intersection points found")
            return np.array([])
        else:
            inScsData = sectP.GetData()
            noSect = inScsData.GetNumberOfTuples()
            psInsec = []
            for idx in range(noSect):
                _tup = inScsData.GetTuple3(idx)
                psInsec.append(_tup)
            if oneP == True:
                # 假设源点在model内,且只有一个交点
                return np.array(psInsec)[0]
            else:
                # Helper.addFids(np.array(psInsec),mNam)
                return np.array(psInsec)
    @staticmethod
    def adScrew(adLineName, obbtree):
        """Adjust the angle diameter and length of the screw

        param:

        return: PZ, PB
        """

        lineN = Helper.line2Cyl(adLineName,)
        PB0 = lineN[0]
        PA = lineN[1]
        if PB0[1] > PA[1]:
            PB0 = lineN[1]
            PA = lineN[0]

        adDia = lineN[2]

        Helper.nodsDel(adLineName)

        PBs = Helper.rayCast(Helper.p2pLn(PA, PB0, plus=10, mNam="l1")[
                             0], Helper.p2pLn(PB0, PA, plus=20, mNam="l2")[0], obbtree=obbtree, oneP=False)
        print(PBs)
        PB = PBs[0]
        Helper.addFid(PB, mNam="Pb")
        PA = PBs[-1]
        Helper.addFid(PA, mNam="Pbb")
        Length = Helper.p2pDst(PA,  PB)*.9 // 5 * 5
        line = Helper.p2pLn(
            PB,
            PA,
            plus=Length - Helper.p2pDst(PB,  PA),
            Dia=adDia,
            mNam=adLineName,
            lock=False,
        )
        # Helper.line2Cyl(adLineName,True)
        return PB, PA, line

    # ### [[柱角术]]
    # - screwAngle

    @staticmethod
    def screwAngle(mNam):
        """Screw angle"""
        lineN = Helper.line2Cyl(mNam)
        Pz = lineN[0]
        Pa = lineN[1]
        Lscrew = lineN[2]
        x = [Pa[0], Pz[1], Pz[2]]
        xy = [Pa[0], Pa[1], Pz[2]]
        y = [Pz[0], Pa[1], Pz[2]]
        yz = [Pz[0], Pa[1], Pa[2]]
        """
  1. TPA: x_o_xy_Angle((PA[0],PZ[1],PZ[2]),PZ,(PA[0],PA[1],PZ[2]))
  2. SPA: y_o_yz_Angle((PZ[0],PA[1],PZ[2]),PZ,(PZ[0],PA)[1],PA[2]))
  3. xyy: x*tan(SPA+3*sn)
  4. yzz: y*tan(TPA+3*tn)
  """

        TPA = Helper.p3Angle(x, Pz, xy)  # Coronary Angle (PSA)
        # logging.debug("SPA:{}".format(SPA))
        SPA = Helper.p3Angle(y, Pz, yz)  # Syryatic angle (PTA)
        # logging.debug("TPA:{}".format(TPA))
        return np.around(TPA), np.around(SPA)

# ### p3Angle
# - [[p3Angle]]

    @staticmethod
    def p3Angle(P0, P, P2, delN=True, mNam="Angle", color="purple"):
        """3 Points Angle

        param: P0,P,P2,delN=True, mNam="Angle",color="red"

        return: ∠P
        """
        markupsNode = SCENE.AddNewNodeByClass(
            "vtkMRMLMarkupsAngleNode")
        markupsNode.AddControlPoint(vtk.vtkVector3d(P0))
        markupsNode.AddControlPoint(vtk.vtkVector3d(P))
        markupsNode.AddControlPoint(vtk.vtkVector3d(P2))
        markupsNode.SetName(mNam)
        markupsNode.CreateDefaultDisplayNodes()
        dn = markupsNode.GetDisplayNode()
        dn.SetSelectedColor(Helper.myColor(color))

        measurement = markupsNode.GetMeasurement("angle").GetValue()
        if delN == True:
            Helper.nodsDel(mNam)
        return measurement

    @staticmethod
    def p3Box(p0, p1, p2, mNam = "",**kw):
        """p3Box _summary_

        _extended_summary_

        Arguments:
            p0 {_type_} -- _description_
            p1 {_type_} -- _description_
            p2 {_type_} -- _description_

        Keyword Arguments:
            mNam {str} -- _description_ (default: {""})

        Returns:
            _type_ -- _description_
        """
        polydata = VPD()
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(p0)
        plane.SetPoint1(p1)
        plane.SetPoint2(p2)
        plane.Update()
        polydata = plane.GetOutput()
        return Helper.getArr(polydata), Helper.pdMDisp(polydata, mNam,**kw)

    @staticmethod
    def plaModCk(oMod: MOD,
                 norm: PS=None,
                 cp: PS=None,
                 mNam: str="",
                 inOut: bool=1,
                 ck: bool=True,
                 **kw
                 )-> any:
        """ps2Contour 点云轮廓

        从点云arr提取轮廓

        Args:
            arr (arr): 点云
            norm (arr): 平面方向
            cp (arr): 平面中点
            mNam (str, optional): 模名. Defaults to "".
            inOut (str, optional): 内外轮廓?. Defaults to "in".

        Returns:
            - _description_
        """
        obb = Helper.mod2Obb(oMod)
        mArr = Helper.getArr(oMod)
        cp = np.mean(mArr,axis=0)
        px = Helper.closestP(cp,mArr,'max')
        norm = Helper.p3Nor(cp,mArr[0],mArr[len(mArr)//3])
        if norm != Helper.p3Nor(cp,mArr[0],mArr[len(mArr)//3*2]):
            Helper.addLog('这不是平面model')
            return False
        crArr = Helper.p3Cir(cp, rad = 1, norm = norm, sn = 90, mNam=f"{mNam}cri") # ref垂直旋转圆
        cps = []
        for i,p in enumerate(crArr):
            p = Helper.rayCast(cp, px ,obb, plus=3, oneP=False)
            if len(p)>0:
                cps.append(p[0,-1][inOut])
            else:
                if ck:
                    return False
        if mNam!="":
            return Helper.pds2Mod(np.array(cps),mNam=mNam,**kw)
        return cps


        return return_list

    @staticmethod
    def p3Cir(pc: PS,
                norm: PS = None,
                rad: float=0,
                p1: PS = None,
                p2: PS = None,
                sn: int=120,
                mNam: str = "",
                **kw
                )-> np.ndarray:
        """三点圆
            - parame:
            - return: cnnArr
            - Note:默认p0为圆心, norm为法向量, 或者半径1p,pc距离
        """
        polydata = VPD()
            # planePosition = points.mean(axis=1)
        if norm is None:
            norm = Helper.p3Nor(p1, p2, pc)
        if rad == 0: # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            rad = npl.norm(pc-p1)
        cirSoc = vtk.vtkRegularPolygonSource()
        cirSoc.SetNumberOfSides(sn)
        cirSoc.SetRadius(rad)
        cirSoc.SetGeneratePolygon(False)
        cirSoc.SetNormal(norm)
        cirSoc.SetCenter(pc)
        cirSoc.Update()
        polydata = cirSoc.GetOutput()
        data = polydata.GetPoints().GetData()
        cnnArr = vtk_to_numpy(data)
        if mNam != "":
            Helper.pdMDisp(polydata, mNam,**kw)
        return cnnArr
    @staticmethod
    def p3Tri(p0, p1, p2, mNam = "", **kw):
        polydata = VPD()

        Points = vtk.vtkPoints()
        Triangles = vtk.vtkCellArray()

        Points.InsertNextPoint(p0)
        Points.InsertNextPoint(p1)
        Points.InsertNextPoint(p2)

        Triangle = vtk.vtkTriangle()
        Triangle.GetPointIds().SetId(0, 0)
        Triangle.GetPointIds().SetId(1, 1)
        Triangle.GetPointIds().SetId(2, 2)
        Triangles.InsertNextCell(Triangle)

        polydata.SetPoints(Points)
        polydata.SetPolys(Triangles)
        return Helper.pdMDisp(polydata, mNam, **kw)
        # return np.array([p0,p1,p2])

# ### p2pAngle
# - [[p2pAngle]]

    @staticmethod
    def p2pAngle(PA, PL):
        lx = [PA[0], PL[1], PL[2]]
        lxy = [PA[0], PA[1], PL[2]]
        ly = [PL[0], PA[1], PL[2]]
        lyz = [PL[0], PA[1], PA[2]]

        lSPA = Helper.p3Angle(lx, PL, lxy, mNam="SPA")  # 冠状角(PSA)
        lTPA = Helper.p3Angle(ly, PL, lyz, mNam="TPA")  # 矢状角(PTA)
        return lSPA, lTPA

# ### [[叉面坐标术]]
    @staticmethod
    def getSP3(ps: PS=None,
               setSPs: bool=False
               )->PS:
        '''getSP3 获取叉面

        _extended_summary_

        Args:
            ps (PS, optional): 三点(中右前三点,或中点). Defaults to None.
            setSPs (bool, optional): 是否Set. Defaults to False.

        Returns:
            - 返回叉点和位置(法向量)

        Note:
            - _description_
        '''
        # Compute the center of rotation (common intersection point of the three planes)
        # http://mathworld.wolfram.com/Plane-PlaneIntersection.html
        if ps is not None and ps.ndim==2: # 三点获取
                cp = ps[0]
                nors = np.asarray([Helper.p3Nor(ps[1], ps[2], ps[0]),
                                   Helper.p2pLn(cp,ps[2])[1],
                                   Helper.p2pLn(cp,ps[1])[1],
                                   ])
        else:
            rNode = SCENE.GetNodeByID('vtkMRMLSliceNodeRed')
            yNode = SCENE.GetNodeByID('vtkMRMLSliceNodeYellow')
            gNode = SCENE.GetNodeByID('vtkMRMLSliceNodeGreen')

            snodR2ras = rNode.GetSliceToRAS()
            # axR = [snodR2ras.GetElement(0,0),snodR2ras.GetElement(1,0),snodR2ras.GetElement(2,0)]
            norR = [snodR2ras.GetElement(0,2),snodR2ras.GetElement(1,2),snodR2ras.GetElement(2,2)]
            if ps is None:
                pR = [snodR2ras.GetElement(0,3),snodR2ras.GetElement(1,3),snodR2ras.GetElement(2,3)]

            snod2Ras = gNode.GetSliceToRAS()
            # axG = [snod2Ras.GetElement(0,0),snod2Ras.GetElement(1,0),snod2Ras.GetElement(2,0)]
            norG = [snod2Ras.GetElement(0,2),snod2Ras.GetElement(1,2),snod2Ras.GetElement(2,2)]
            if ps is None:
                pG = [snod2Ras.GetElement(0,3),snod2Ras.GetElement(1,3),snod2Ras.GetElement(2,3)]

            snod2Ras = yNode.GetSliceToRAS()
            # axY = [snod2Ras.GetElement(0,0),snod2Ras.GetElement(1,0),snod2Ras.GetElement(2,0)]
            norY = [snod2Ras.GetElement(0,2),snod2Ras.GetElement(1,2),snod2Ras.GetElement(2,2)]
            if ps is None:
                pY = [snod2Ras.GetElement(0,3),snod2Ras.GetElement(1,3),snod2Ras.GetElement(2,3)]
            # print([pR, pG, pY])
            # Computed intersection point of all planes
                cp = [0,0,0]
                n2_xp_n3 = [0,0,0]
                x1_dp_n1 = vtk.vtkMath.Dot(pR,norR)
                vtk.vtkMath.Cross(norG,norY,n2_xp_n3)
                vtk.vtkMath.MultiplyScalar(n2_xp_n3, x1_dp_n1)
                vtk.vtkMath.Add(cp,n2_xp_n3,cp)
                n3_xp_n1 = [0,0,0]
                x2_dp_n2 = vtk.vtkMath.Dot(pG,norG)
                vtk.vtkMath.Cross(norY,norR,n3_xp_n1)
                vtk.vtkMath.MultiplyScalar(n3_xp_n1, x2_dp_n2)
                vtk.vtkMath.Add(cp,n3_xp_n1,cp)
                n1_xp_n2 = [0,0,0]
                x3_dp_n3 = vtk.vtkMath.Dot(pY,norY)
                vtk.vtkMath.Cross(norR,norG,n1_xp_n2)
                vtk.vtkMath.MultiplyScalar(n1_xp_n2, x3_dp_n3)
                vtk.vtkMath.Add(cp,n1_xp_n2,cp)
                # print(cp)
                norMatx = vtk.vtkMatrix3x3()
                norMatx.SetElement(0,0,norR[0])
                norMatx.SetElement(1,0,norR[1])
                norMatx.SetElement(2,0,norR[2])
                norMatx.SetElement(0,1,norG[0])
                norMatx.SetElement(1,1,norG[1])
                norMatx.SetElement(2,1,norG[2])
                norMatx.SetElement(0,2,norY[0])
                norMatx.SetElement(1,2,norY[1])
                norMatx.SetElement(2,2,norY[2])
                norMatxDeterminant = norMatx.Determinant()
                if abs(norMatxDeterminant)>0.01:
                    # there is an intersection point
                    vtk.vtkMath.MultiplyScalar(cp, 1/norMatxDeterminant)
                else:
                    # no intersection point can be determined, use just the position of the axial slice
                    cp = pR
            else:
                cp = ps
            cp, nors = ndA(cp),np.asarray([norY, norG, norR]) # ,np.asarray([axR, axG, axY])
            # 验证并排序
        nor = Helper.rasCk(nors)
        if setSPs is True:
            Helper.restoreViews()
            Helper.setSP3(cp, nor)
        return cp, nor

    @staticmethod
    def getPtri(obb: OB = None,
                ps: PS = None,
                mNam: str='l',
                setSPs: bool=False
                )-> dfDic:
        '''getPtri 获取椎三角

        _extended_summary_

        Args:
            obb (OB, optional): 定向树. Defaults to None.
            ps (PS, optional): 参考点. Defaults to None.
            mNam (str, optional): 莫名. Defaults to 'l'.
            setSPs (bool, optional): 设吗. Defaults to False.

        Returns:
            - 返回椎三角字典

        Note:
            - _description_
        '''
        ppDic = {}
        sPp, nors = Helper.getSP3(ps, setSPs) # 叉面数据
        print(f'{sPp=}')
        pps = []
        ras = 'RAS'
        for i, d in enumerate(nors):
            pps.append(Helper.p2pLn(sPp,drt=d,mNam=[f'{ras[i]}_{mNam}',''][int(mNam=='')])[:2])
        ppDic = dict(zip(list(ras),pps))
        ppDic['sPp'] = sPp
        ppDic['pNorm'] = nors
        if obb is not None:
            bPs = [Helper.rayCast(sPp, drt=-ppDic['R'][1], oNod=obb, oneP=False),
                   Helper.rayCast(sPp, drt= ppDic['R'][1], oNod=obb, oneP=False)
                   ]
            if len(bPs) > 1:
                ppDic['P0s'] = [bPs[0][0], bPs[1][0]]
                ppDic['Pm'] = np.mean(ppDic['P0s'], axis=0)
                ppDic['P1s'] = [bPs[0][-1], bPs[1][-1]]
            else:
                Helper.addLog('sPp not in Canal...')
                return ppDic
        if setSPs:
            Helper.p3SP3([[ppDic['sPp'], ppDic['Pm']][int(obb!=None)], ppDic['R'][0],ppDic['A'][0]])
        return ppDic

    @staticmethod
    def rasCk(nors):
        dic={}
        ras = 'RAS'
        for n in nors:
            ind = np.argmax(abs(n))
                # print(ind)
            dic[ras[ind]] = n*[1,-1][int(n[ind]<0)]
        nors = np.array([dic['R'],dic['A'],dic['S']])
        return nors

    @staticmethod
    def setSP3(cp:PS=None,
               sNrs:PS=None,
               )->any:
        '''setSP3 叉面设置

        _extended_summary_

        Args:
            sPos (PS, optional): 叉面交点. Defaults to None.
            sNor (PS, optional): 叉面方向. Defaults to None.

        Returns:
            - 至目标叉面

        Note:
            - _description_
        '''
        sNods = [SCENE.GetNodeByID("vtkMRMLSliceNodeRed"),
                 SCENE.GetNodeByID("vtkMRMLSliceNodeGreen"),
                 SCENE.GetNodeByID("vtkMRMLSliceNodeYellow")
                 ]
        def __setSP(sNod, sNor, cp):
            dfUp = [0,0,1]
            drtR = [-1,0,0]
            if sNor[1]>=0:
                sNorm = sNor
            else:
                sNorm = [-sNor[0], -sNor[1], -sNor[2]]
            upAng = vtk.vtkMath.AngleBetweenVectors(sNorm, dfUp)
            minAng = 0.75 # about 45 degrees
            if upAng > minAng and upAng < vtk.vtkMath.Pi() - minAng:
                upDrt = dfUp
                saxY = upDrt
                saxX = [0, 0, 0]
                vtk.vtkMath.Cross(saxY, sNorm, saxX)
            else:
                saxX = drtR
            sNod.SetSliceToRASByNTP(sNorm[0], sNorm[1], sNorm[2],
                                    saxX[0], saxX[1], saxX[2],
                                    cp[0], cp[1], cp[2],
                                    0
                                    )
        if cp is None:
            cp,sNrs= Helper.getSP3()

        for i, nod in enumerate(sNods):
            __setSP(nod,sNrs[2-i],cp)
        return cp[0],sNrs

    @staticmethod
    def p3SP3(mraP):
        sNods = [SCENE.GetNodeByID("vtkMRMLSliceNodeRed"),
                 SCENE.GetNodeByID("vtkMRMLSliceNodeGreen"),
                 SCENE.GetNodeByID("vtkMRMLSliceNodeYellow")
                 ]
        def __p3SP(ps, sNod):
            n = Helper.p3Nor(ps[1],ps[2],ps[0])# plane normal direction
            # print(f"{n=}")
            t = np.cross([0.0, 0.0, 1], n) # plane transverse direction
            t = t/npl.norm(t)
            # Set slice plane orientation and position
            sNod.SetSliceToRASByNTP(n[0], n[1], n[2],
                                    t[0], t[1], t[2],
                                    ps[0][0], ps[0][1], ps[0][2],
                                    0
                                    )
            return n
        mraP = Helper.getArr(mraP)
        mraP = np.append(mraP, np.array([Helper.p3Nor(mraP[1],mraP[2],mraP[0])]),axis=0)
        ps = [np.asarray([mraP[0],mraP[1],mraP[2]]),
                np.asarray([mraP[0],mraP[2],mraP[3]]),
                np.asarray([mraP[0],mraP[1],mraP[3]]),
                ]
        i=0
        nrs=[]
        for i, p in enumerate(ps):
            nrs.append(__p3SP(p, sNods[i]))
            print(f"{nrs=}")
        return mraP[0], np.asarray(nrs)

    @staticmethod
    def jumpS2P(p):
        Helper.restoreViews()
        slicer.vtkMRMLSliceNode.JumpAllSlices(
            SCENE, *p, slicer.vtkMRMLSliceNode.CenteredJumpSlice)

    @staticmethod
    def jump2Lay(drt: Union[Lit['X','Y','Z',None], PS]=None,
                 pls: Opt[float]=None,
                 p: PS=None
                 )-> np.ndarray:
        '''jump2Lay 跳层

        _extended_summary_

        Args:
            drt (Lit[&#39;R&#39;,&#39;G&#39;,&#39;Y&#39;,None], optional): 方向. Defaults to None.
            pls (Opt, optional): 距离. Defaults to None.
            p (PS, optional): 直接跳点. Defaults to None.

        Returns:
            - 目的点

        Note:
            - _description_
        '''
        xyz = dict(X = np.array([1,0,0]),
                   Y = np.array([0,1,0]),
                   Z = np.array([0,0,1]))
        if p is not None:
            p = Helper.getArr(p)
        else:
            pls = [drt,xyz[drt]][isinstance(drt, str)]*pls
        ps=[]
        for i, v in enumerate("YGR"):
            Lay = SC[v].sliceLogic().GetSliceOffset()
            # p[i] = p[i]*[1,-1][int(Lay==p[i])]
            SC[v].sliceLogic().SetSliceOffset([p[i],Lay+pls[i]][int(p==None)])
            ps.append(Lay+p[i])
        return np.asarray(ps)

    @staticmethod
    def rotOrtSlice(insP=None, angle=10.):
        """
        Rotate the orthogonal nodes around the common intersection point, around the normal of the slice node
        """
        if insP is None:
            insP = Helper.getSlicesPIP()[0]
        Helper.restoreViews()
        axNode = SCENE.GetNodeByID("vtkMRMLSliceNodeRed")
        saNode = SCENE.GetNodeByID("vtkMRMLSliceNodeYellow")
        coNode = SCENE.GetNodeByID("vtkMRMLSliceNodeGreen")

        sToRas = axNode.GetSliceToRAS()
        rotTf = vtk.vtkTransform()
        rotTf.RotateZ(angle)
        #rotatedAxialSliceToRas = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(sToRas, rotTf.GetMatrix(), sToRas)
        sNor = [-sToRas.GetElement(0, 2), -
                sToRas.GetElement(1, 2), -sToRas.GetElement(2, 2)]
        sAX = [-sToRas.GetElement(0, 0), -sToRas.GetElement(1,
                                                            0), -sToRas.GetElement(2, 0)]
        saNode.SetSliceToRASByNTP(
            sNor[0], sNor[1], sNor[2], sAX[0], sAX[1], sAX[2], insP[0], insP[1], insP[2], 1)
        coNode.SetSliceToRASByNTP(
            sNor[0], sNor[1], sNor[2], sAX[0], sAX[1], sAX[2], insP[0], insP[1], insP[2], 2)
        return True

    @staticmethod
    def restoreViews():
        views = slicer.app.layoutManager().sliceViewNames()
        for view in views:
            sliceNode = slicer.app.layoutManager().sliceWidget(view).mrmlSliceNode()
            sliceNode.SetOrientationToDefault()
            for axis in ("X", "Y", "Z"):
                attributeName = "current" + axis + "Rotation"
                sliceNode.SetAttribute(attributeName, "0.0")


# ### gradeScrew
# - [[gradeScrew]]

    @staticmethod
    def gradeScrew(mNam = "", volumeNode=None, getPos=True, minThred=None, modNode=None, obbtree=None, P0=[0], centroid=False):
        """灰阶化模特

        return: sumVox, Parr

        """
        volume_node = ut.getNode(volumeNode)
        voxels = Helper.getArr(
            volume_node)  # 获取像numpy阵列一样的voxels
        pvDic = {}
        parr = []
        if mNam != "":
            Pdata = Helper.cyl2Line(mNam)
        else:
            Pdata = Helper.cyl2Line(modNode=modNode)
            mNam = modNode.GetName()
        Seg = Pdata[3]

        totalCount = int(np.around(Helper.p2pDst(Pdata[0][1],  Pdata[0][0])))
        P1arr = Pdata[2][0]
        P2arr = Pdata[2][1]
        p0s = []
        for i in range(Seg):
            P1 = P1arr[i]
            P2 = P2arr[i]
            if getPos is True:
                p = Helper.p2pLn(P1, (P1+P2)/2, plus=0)[0]
                p0s.append(abs(p[0]-P0[0]))
        minInx = p0s.index(min(p0s)) if getPos is True else 0
        sumVox = 0
        bVox = 0
        for i in range(Seg):
            P1 = P1arr[i]
            P2 = P2arr[i]
            ii = i-minInx if i >= minInx else i+minInx
            if mNam[-1] == "R":
                ii = Seg-1-i
            else:
                ii = i
            for j in range(1, totalCount):
                jj = totalCount//2
                if centroid is True:
                    P = Helper.p2pLn(P1, P2, plus=jj-totalCount)[0]
                    pvDic[i] = P
                else:
                    P = Helper.p2pLn(P1, P2, plus=j-totalCount)[0]
                    volumeRasToIjk = vtk.vtkMatrix4x4()
                    volume_node.GetRASToIJKMatrix(volumeRasToIjk)
                    point_Ijk = [0, 0, 0, 1]
                    volumeRasToIjk.MultiplyPoint(np.append(P, 1.0), point_Ijk)
                    point_Ijk = [int(round(c)) for c in point_Ijk[0:3]]
                    voxelValue = voxels[point_Ijk[2],
                                        point_Ijk[1], point_Ijk[0]]
                    sumVox += voxelValue
                    if minThred is not None:
                        if voxelValue < minThred and obbtree.InsideOrOutside(P) != -1:
                            if ii in [0, 1, 11]:
                                Helper.addFids(
                                    P, Dia=3, mNam=f"dp_{mNam}", lableName=f"{jj}_{ii}_vox={voxelValue}", color="red", fontScale=0, GlyphType=9)
                                Helper.nodsDisp(f"dp_{mNam}", display=False)
                                bVox += voxelValue
                            elif ii in [5, 6, 7]:
                                Helper.addFids(
                                    P, Dia=3, mNam=f"dp_{mNam}", lableName=f"{jj}_{ii}_vox={voxelValue}", color="red", fontScale=0, GlyphType=9)
                                parr.append(P)
                            if len(parr) > 90:
                                return Helper.addLog("那么多坏点,我不干了...")
                            Helper.nodsDisp(f"dp_{mNam}", display=False)
                            bVox += voxelValue
                            meanVlist = np.around((np.array(VVlist).mean(axis=0)), 0)
        if centroid is True:
            ps = []
            for k, v in pvDic.items():
                kk = (k+6) % 12
                vv = Helper.p2pLn(
                    v, pvDic[kk], mNam=f"{k}-{k+6}_l", Dia=.1, plus=10)[0]
                p0 = Helper.rayCast(vv, v, obbtree)
                Helper.addFids(p0)
                ps.append(p0)
            parr = np.array(ps)
            print(parr)
            cp = np.average(parr, axis=0)
            Helper.addFid(cp, mNam=f"{mNam}_cp")
            i = 1
            for p0 in ps:
                Helper.p2pLn(
                    cp, p0, Dia=.2, plus=0, mNam=f'cline{i}', color=Helper.myColor("blue"))
                i += 1
        return sumVox, bVox

    @staticmethod
    def maxVox_pos(vList, pList):
        vSubs = (vList[1:]-vList[-1:]).tolist()
        i = vSubs.index(max(vSubs))
        return pList[i]

    @staticmethod
    def filterIsthm(vList, pList):
        '''
        '''

        vlistT = vList(map(vList, zip(*vList)))
        plistT = vList(map(pList, zip(*pList)))
        for i, ar in enumerate(vlistT):
            if min(ar) > 1:
                vlistT = vlistT[i:]
                break
        vlistT = vlistT[::-1]
        for i, ar in enumerate(vlistT):
            if min(ar) > 1:
                vlistT = vlistT[i:]
                break
        vlistT = vlistT[::-1]

    @staticmethod
    def plane2Angle():
        angleDegs = []
        sliceNormalVector = []
        rSlice = "vtkMRMLSliceNodeRed"
        ySlice = "vtkMRMLSliceNodeYellow"
        gSlice = "vtkMRMLSliceNodeGreen"

        ysliceToRAS = SCENE.GetNodeByID(ySlice).GetSliceToRAS()
        ysliceNormalVector = [
            [
                ysliceToRAS.GetElement(0, 2),
                ysliceToRAS.GetElement(1, 2),
                ysliceToRAS.GetElement(2, 2),
            ]
        ]
        yangleRad = vtk.vtkMath.AngleBetweenVectors(
            ysliceNormalVector[0], [0, 1, 0])
        yangleDeg = np.around(vtk.vtkMath.DegreesFromRadians(yangleRad) - 90)

        rsliceToRAS = SCENE.GetNodeByID(rSlice).GetSliceToRAS()
        rsliceNormalVector = [
            [
                rsliceToRAS.GetElement(0, 2),
                rsliceToRAS.GetElement(1, 2),
                rsliceToRAS.GetElement(2, 2),
            ]
        ]
        rangleRad = vtk.vtkMath.AngleBetweenVectors(
            rsliceNormalVector[0], [1, 0, 0])
        rangleDeg = np.around(90 - vtk.vtkMath.DegreesFromRadians(rangleRad))

        gsliceToRAS = SCENE.GetNodeByID(gSlice).GetSliceToRAS()
        gsliceNormalVector = [
            [
                gsliceToRAS.GetElement(0, 2),
                gsliceToRAS.GetElement(1, 2),
                gsliceToRAS.GetElement(2, 2),
            ]
        ]
        gangleRad = vtk.vtkMath.AngleBetweenVectors(
            gsliceNormalVector[0], [0, 0, 1])
        gangleDeg = np.around(90 - vtk.vtkMath.DegreesFromRadians(gangleRad))

        return [yangleDeg, rangleDeg, gangleDeg]

    @staticmethod
    def sVB_Norm():
        cp = (p0+p1)/2
        cyl = Helper.p2pCyl(cp, drt=[0,0,1],rad=10, plus=10)
        Helper.cropVcanal(vol,cyl)

    @staticmethod
    def cropVcanal(Vol="croppedROI", modNod="L1_RXXX", miniTresd=100, mNam=''):
        modNod = Helper.namNod(modNod)
        volume = Helper.namNod(Vol)

        segmentationNode = SCENE.AddNewNodeByClass(
            "vtkMRMLSegmentationNode")
        segmentationNode.SetName("Segmentation")

        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(
            modNod, segmentationNode
        )

        segmentationNode = ut.getNode("Segmentation")

        import SegmentStatistics
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter(
            "Segmentation", segmentationNode.GetID())
        segStatLogic.getParameterNode().SetParameter(
            "LabelmapSegmentStatisticsPlugin.obb_origin_ras.enabled", str(True))
        segStatLogic.getParameterNode().SetParameter(
            "LabelmapSegmentStatisticsPlugin.obb_diameter_mm.enabled", str(True))
        segStatLogic.getParameterNode().SetParameter(
            "LabelmapSegmentStatisticsPlugin.obb_drt_ras_x.enabled", str(True))
        segStatLogic.getParameterNode().SetParameter(
            "LabelmapSegmentStatisticsPlugin.obb_drt_ras_y.enabled", str(True))
        segStatLogic.getParameterNode().SetParameter(
            "LabelmapSegmentStatisticsPlugin.obb_drt_ras_z.enabled", str(True))
        segStatLogic.computeStatistics()
        stats = segStatLogic.getStatistics()

        for segmentId in stats["SegmentIDs"]:
            obb_origin_ras = np.array(
                stats[segmentId, "LabelmapSegmentStatisticsPlugin.obb_origin_ras"])
            obb_diameter_mm = np.array(
                stats[segmentId, "LabelmapSegmentStatisticsPlugin.obb_diameter_mm"])
            obb_drt_ras_x = np.array(
                stats[segmentId, "LabelmapSegmentStatisticsPlugin.obb_drt_ras_x"])
            obb_drt_ras_y = np.array(
                stats[segmentId, "LabelmapSegmentStatisticsPlugin.obb_drt_ras_y"])
            obb_drt_ras_z = np.array(
                stats[segmentId, "LabelmapSegmentStatisticsPlugin.obb_drt_ras_z"])
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            roi = SCENE.AddNewNodeByClass("vtkMRMLMarkupsROINode")
            roi.GetDisplayNode().SetHandlesInteractive(
                False)
            roi.SetSize(obb_diameter_mm)
            obb_center_ras = obb_origin_ras+0.5 * \
                (obb_diameter_mm[0] * obb_drt_ras_x + obb_diameter_mm[1]
                 * obb_drt_ras_y + obb_diameter_mm[2] * obb_drt_ras_z)
            boundingBoxToRasTransform = np.row_stack((np.column_stack(
                (obb_drt_ras_x, obb_drt_ras_y, obb_drt_ras_z, obb_center_ras)), (0, 0, 0, 1)))
            boundingBoxToRasTransformMatrix = ut.vtkMatrixFromArray(
                boundingBoxToRasTransform)
            roi.SetAndObserveObjectToNodeMatrix(
                boundingBoxToRasTransformMatrix)
            cropLogic = slicer.modules.cropvolume.logic()
            crop_module = slicer.vtkMRMLCropVolumeParametersNode()
            crop_module.SetROINodeID(roi.GetID())
            crop_module.SetInputVolumeNodeID(volume.GetID())
            cropLogic.Apply(crop_module)
            croppedVolume = SCENE.GetNodeByID(
                crop_module.GetOutputVolumeNodeID())
            nodeName = "vertebral canal_{}".format(modNod)
            croppedVolume.SetName(nodeName)
            if mNam != '':
                Helper.segMod(nodeName, mNam=nodeName)
            Helper.nodsDel("Segmentation")
            Helper.nodsDel("Transform*")
        return

    @staticmethod
    def jump2Lay(drt: Union[Lit['X','Y','Z',None], PS]=None,
                dst: float=0.,
                )-> np.ndarray:
        SC = {}
        SC['R'] = slicer.app.layoutManager().sliceWidget('Red')
        SC['Y'] = slicer.app.layoutManager().sliceWidget('Yellow')
        SC['G'] = slicer.app.layoutManager().sliceWidget('Green')
        xyz = dict(X = np.array([1,0,0]),
                Y = np.array([0,1,0]),
                Z = np.array([0,0,1]))
        pls = [drt,xyz[drt]][drt in 'PS']*dst
        ps=[]
        for i, v in enumerate("YGR"):
            Lay = SC[v].sliceLogic().GetSliceOffset()
            SC[v].sliceLogic().SetSliceOffset(Lay+pls[i])
            ps.append(Lay+p[i])
        return np.asarray(ps)

    @staticmethod
    def arrVox(arr, Vol=None):
        if Vol is None:
            vol = Helper.getFirstNodByCls('vtkMRMLVolumeNode')
        else:
            vol = Helper.namNod(Vol)
        voxels = Helper.getArr(vol)
        voxVals=[]
        for p in arr:
            volRas2Ijk = vtk.vtkMatrix4x4()
            vol.GetRASToIJKMatrix(volRas2Ijk)
            pIjk = [0, 0, 0, 1]
            volRas2Ijk.MultiplyPoint(np.append(p, 1.0), pIjk)
            pIjk = [int(round(c)) for c in pIjk[0:3]]
            voxVals.append(voxels[pIjk[2], pIjk[1], pIjk[0]])
        return np.mean(np.asarray(voxVals),axis=0)

    @staticmethod
    def psFromVol(vol: VOL,
                  cond: bool=True,
                  arr: PS=None
                  )->PS:
        volumeArray = slicer.util.arrayFromVolume(vol)
        if cond:
            inds = np.where(volumeArray<100)
            arr = np.asarray(inds).T[:,::-1]
        else:
            vBds = np.zeros(6)
            vol.GetBounds(vBds)
            p0 = np.asarray(vBds[0::2])
            arr = np.zeros(3)
        ps = []
        for p in arr:
            volIjk2Ras = vtk.vtkMatrix4x4()
            vol.GetIJKToRASMatrix(volIjk2Ras)
            pRas = [0, 0, 0, 1]
            volIjk2Ras.MultiplyPoint(np.append(p,1.0), pRas)
            ps.append(pRas[0:3])
        return np.asarray(ps)

    @staticmethod
    def skBiOp(vol0, vol1, skOp='OR', vol=None, mNam=None):
        sImg0 = sitkUtils.PullVolumeFromSlicer(vol0)
        sImg1 = sitkUtils.PullVolumeFromSlicer(vol1)
        sType = sImg0.GetPixelIDValue()
        sImg1 = sitk.Cast(sImg1, sType)
        sImg1 = sitk.Resample(sImg1, sImg0, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sImg0.GetPixelIDValue())

        if skOp == 'OR':
            SandvImg = sitk.Or(sImg0, sImg1)
        elif skOp == 'AND':
            SandvImg = sitk.And(sImg0, sImg1)
        elif skOp == 'XOR':
            SandvImg = sitk.Xor(sImg0, sImg1)
        SandvImg = sitk.And(sImg0, sImg1)
        sitkUtils.PushVolumeToSlicer(SandvImg, vol,name=mNam)

    @staticmethod
    def ps2vol(ps, mNam):
        ps = np.asarray(ps)
        dimX = ps[:, 0].max() - ps[:, 0].min()
        dimY = ps[:, 1].max() - ps[:, 1].min()
        dimZ = ps[:, 2].max() - ps[:, 2].min()
        imageSize = [dimX, dimY, dimZ]
        voxelType=vtk.VTK_UNSIGNED_CHAR
        imageOrigin = [ps[:, 0].min(), ps[:, 1].min(), ps[:, 2].min()]
        imageSpacing = [1.0, 1.0, 1.0]
        imageDirections = [[1,0,0], [0,1,0], [0,0,1]]
        fillVoxelValue = 0
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageSize)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        for point in ps:
            i, j, k = int(point[0]), int(point[1]), int(point[2])
            imageData.SetScalarComponentFromFloat(i, j, k, 0, 1)
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", mNam)
        volumeNode.SetOrigin(imageOrigin)
        volumeNode.SetSpacing(imageSpacing)
        volumeNode.SetIJKToRASDirections(imageDirections)
        volumeNode.SetAndObserveImageData(imageData)
        volumeNode.CreateDefaultDisplayNodes()
        volumeNode.CreateDefaultStorageNode()
        return volumeNode

    @staticmethod
    def probeVol(p: PS=None,
                 drt: PS=None,
                 Vol: VOL="8: Unnamed Series",
                 n = 3,
                 ck: bool=False
                 ) ->np.ndarray:
        if Vol is None:
            vNode = Helper.getFirstNodByCls('vtkMRMLVolumeNode')
        vNode = Helper.namNod(Vol)
        voxels = Helper.getArr(vNode)
        if p is None:
            p,nor=Helper.getSP3()
            drt = -nor[1]
        def __meanVox(arr):
            voxVals=[]
            for P in arr:
                volumeRasToIjk = vtk.vtkMatrix4x4()
                vNode.GetRASToIJKMatrix(volumeRasToIjk)
                point_Ijk = [0, 0, 0, 1]
                volumeRasToIjk.MultiplyPoint(np.append(P, 1.0), point_Ijk)
                point_Ijk = [int(round(c)) for c in point_Ijk[0:3]]
                voxVals.append(voxels[point_Ijk[2], point_Ijk[1], point_Ijk[0]])
            return np.mean(np.asarray(voxVals),axis=0)
        px = Helper.p2pLn(p,drt=drt)[2]
        arr = np.concatenate([Helper.p3Cir(px( 1), drt, rad=3, sn=6),
                              Helper.p3Cir(px(-1), drt, rad=3, sn=6)
                              ])
        meanThr = __meanVox(arr)
        if ck is False:
            return meanThr
        else:
            xx = 0
            for i in range(n, -1, -1):
                meanThr1 = __meanVox(arr)
                out1 = int(meanThr1<MTHR)
                ps = Helper.p2pLn(p, drt=drt, plus=2**i*[1,-1][out1==1])
                p = ps[0]
                arr = Helper.p3Cir(p, drt, 3, sn=12, mNam=f'{i}_Cir')
                print(f'{out1=}')
                xx += out1
            if 0<xx<2**n:
                return meanThr1, p
            else:
                Helper.addLog("没变化呀")
                return False

    @staticmethod
    def pla2D(arr):
        zN = np.array([0,0,1])
        a = np.array([1,1,1])
        U_Z = np.dot(a,zN)
        X = a-np.dot(U_Z, zN)
        X = X/(X[0]**2+X[1]**2+X[2]**2)**.5
        Y = np.cross(zN,X)
        ds=[]
        for u in arr:
            ds.append([np.dot(u,X),np.dot(u,Y),0])
        Helper.addFids(np.array(ds))
        return np.array(ds)

    @staticmethod
    def threProbe(modNode: str, volumeNode="croppedROI"):
        probe = slicer.modules.probevolumewithmodel
        parameters = {}
        arr = np.array()
        parameters["InputModel"] = ut.getNode(modNode)
        outModel =  SMD()
        outModel.SetName("grade_{}".format(modNode))
        SCENE.AddNode(outModel)
        parameters["InputVolume"] = ut.getNode(volumeNode)
        parameters["OutputModel"] = outModel
        slicer.cli.runSync(probe, arr, parameters)

        outModel.CreateDefaultDisplayNodes()
        modDn = outModel.GetDisplayNode()
        modDn.ScalarVisibilityOn()
        modDn.SetActiveScalarName("NRRDImage")
        modDn.SetScalarRangeFlag(1)
        modDn.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
        return arr

    @staticmethod
    def addLog(text="Done", x=False, color="purple", dem=2, sec=.5):
        pyConsole = ut.mainWindow().pythonConsole()
        if x:
            text = f">>>>>>text={np.round(text,dem)}"
        pyConsole.printMessage("\n{}\n>>>>>>".format(text), qt.QColor(color))
        time.sleep(sec)
        slicer.app.processEvents()
        logging.info(text)
        return

    @staticmethod
    def normalP3(oP, xP, yP, length=None, norm = False, mNam = "", mid = False):
        if length is None:
            length = Helper.p2pDst(xP,  oP)
        if norm is False:
            zNor = Helper.p3Nor(xP,yP,oP)
            na = True in np.isnan(zNor)
            if na is True:
                print("好像是:三点在一条直线上...")
                return
            if norm is None:
                zP = oP + zNor / npl.norm(zNor) * length
                yNor = Helper.p3Nor(zP,x,oP)
                yP = oP + yNor / npl.norm(yNor) * length
                z = Helper.p2pLn(oP,zP,plus=0,mNam="{mNam}_z" if mNam != "" else mNam)
                x = Helper.p2pLn(oP,xP,plus=0,mNam="{mNam}_x" if mNam != "" else mNam)
                y = Helper.p2pLn(oP,yP,plus=0,mNam="{mNam}_y" if mNam != "" else mNam)
                return x,y,z
            else:
                zP = oP + zNor / npl.norm(zNor) * length
                z = Helper.p2pLn(oP,zP,plus=0,mNam=f"{mNam}_z" if mNam != "" else mNam)
                return z
        else:
            zNor = yP
            oP = (oP+xP)/2 if mid is True else oP
            zP = oP + zNor / npl.norm(zNor) * length
            yNor = Helper.p3Nor(zP,x,oP)
            yP = oP + yNor / npl.norm(yNor) * length
            y= Helper.p2pLn(oP,yP,plus=0,mNam=f"{mNam}_y" if mNam != "" else mNam)
            return y

    @staticmethod
    def p3PIP(oP, xP=None, yP=None, lNode=None, norm = None,slice="All", setSlices=True):
        if lNode is not None:
            ps = ut.array(lNode)
            xP = ps[0]
            yP = ps[1]

        rp = Helper.p2pLn(oP,xP)[1]
        ap = Helper.p2pLn(oP, yP)[1]
        sp = Helper.p3Nor(xP, yP, oP)

        nors =  np.array([sp, ap, rp])
        axs = np.array([rp, rp, sp])
        if setSlices is True:
            Helper.restoreViews()
            Helper.sPIP(oP, nors, axs, sPIP = True)
        return np.array(oP), nors, axs

    @staticmethod
    def zoom(factor, sliceNodes=None):
        if not sliceNodes:
            sliceNodes = ut.getNodes('vtkMRMLSliceNode*')
        for sliceNode in list(sliceNodes.values()):
            if factor == "Fit":
                sliceWidget = slicer.app.layoutManager().sliceWidget(
                    sliceNode.GetLayoutName())
                if sliceWidget:
                    sliceWidget.sliceLogic().FitSliceToAll()
            else:
                newFOVx = sliceNode.GetFieldOfView()[0] * factor
                newFOVy = sliceNode.GetFieldOfView()[1] * factor
                newFOVz = sliceNode.GetFieldOfView()[2]
                sliceNode.SetFieldOfView(newFOVx, newFOVy, newFOVz)
                sliceNode.UpdateMatrices()

    @staticmethod
    def csvWriter(filePath, input, fieldName=None):
        if isinstance(input, dict):
            np.save(filePath+'.npy')
        else:
            try:
                f = open(filePath,'a',encoding='utf8',newline='')
            except:
                os.chmod(filePath, 0o777)
                f = open(filePath, 'a', encoding='utf8', newline='')

            writer = csv.writer(f, delimiter='|')
            writer.writerow(input)
            f.close()

    @staticmethod
    def uidDcmDic():
        db = slicer.dicomDatabase
        pList = db.patients()
        idPathDic = {}
        regex = r"[^\/]+?$"
        subst = ""
        for p in pList:
            sList = db.studiesForPatient(p)
            for s in sList:
                serList = db.seriesForStudy(s)
                for ser in serList:
                    s = db.filesForSeries(ser)[0]
                    pName = slicer.dicomDatabase.fileValue(
                        ut.longPath(s), "0010,0010")
                    path = re.sub(regex, subst, s)
                    idPathDic[f'{p}_{pName}'] = path
        return idPathDic

    @staticmethod
    def getFirstNodByCls(className='vtkMRMLVolumeNode'):
        return SCENE.GetFirstNode(None, className, False, False)

    @staticmethod
    def getTwoClosestElements(arr):
        seq = sorted(arr)
        dif = float('inf')

        for i, v in enumerate(seq[:-1]):
            d = abs(v - seq[i+1])
            if d < dif:
                first, second, dif = v, seq[i+1], d
        return (first, second)

    @staticmethod
    def nodeClone(mNam = "", modNode=None, cloneAffix="ad"):
        if mNam != "":
            modNode = ut.getNode(mNam)
        else:
            mNam = modNode.GetName()
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
            SCENE)
        itemIDToClone = shNode.GetItemByDataNode(modNode)
        clonedItemID = slicer.modules.subjecthierarchy.logic(
        ).CloneSubjectHierarchyItem(shNode, itemIDToClone)
        clonedNode = shNode.GetItemDataNode(clonedItemID)
        clonedNode.SetName(f"{cloneAffix}_{mNam}")
        return clonedNode
# -

    @staticmethod
    def nodesClone(mNam = "", modNode=None, cloneAffix="ad"):
        """clone nodes

        """
        cloneDic = {}
        if modNode is not None:
            nodes = collections.OrderedDict()
            nodes[modNode.GetName()] = modNode
        else:
            if mNam != "":
                nodes = ut.getNodes(mNam)
        for key, value in nodes.items():
            modCls = value.GetClassName()
            modClone = SCENE.AddNewNodeByClass(modCls)
            modClone.CopyContent(value, False)
            cloneName = f"{cloneAffix}_{key}"
            modClone.SetName(cloneName)
            modClone.GetDisplayNode().CopyContent(value.GetDisplayNode(), False)
            modClone.GetDisplayNode().SetVisibility3D(True)
            modClone.GetDisplayNode().SetVisibility2D(True)
            cloneDic[cloneName] = modClone
        return cloneDic

    @staticmethod
    def fids2Mod(fids, curve=True):
        fids = Helper.namNod(fids)
        outputModel = SCENE.AddNode( SMD())
        outputModel.CreateDefaultDisplayNodes()
        outputModel.GetDisplayNode().SetIntersectingSlicesVisibility(True)
        outputModel.GetDisplayNode().SetColor(1,0,0)

        markupsToModel = SCENE.AddNode(slicer.vtkMRMLMarkupsToModelNode())

        markupsToModel.SetAutoUpdateOutput(True)
        markupsToModel.SetAndObserveModelNodeID(outputModel.GetID())
        markupsToModel.SetAndObserveInputNodeID(fids.GetID())
        if curve:
            markupsToModel.SetModelType(0)
            markupsToModel.SetTubeRadius(0.01)
            markupsToModel.SetTubeLoop(1)
            markupsToModel.SetCurveType(0)
        return outputModel

    @staticmethod
    def multiSelMsgbox(orgList, refVal, lnName):
        '''

        para:
        return:
        '''
        varList = []
        # diaList = []
        n = len(orgList)
        Helper.nodsDel(lnName)
        for v in orgList:
            Helper.addFids(v[1], Dia=1.5, mNam=f"{lnName}_ps", color="red")
            length = Helper.p2pDst(v[1], v[0])
            varList.append(abs(refVal-length))
            varListSort = sorted(varList)
            # if len(v)==3:
            #   diaList.append(cosA*length)
            #   diaListSort = sorted(diaList)
        l = 0
        while l < n:
            messageBox = qt.QMessageBox()
            messageBox.setWindowTitle(" /!\ WARNING /!\ ")
            messageBox.setIcon(messageBox.Warning)
            # if i == 3:
            #   messageBox.setText(f"Found {n} {lnName}s, is this {lnName}? /n If no {lnName} is found, select a 最接近的那个. ")

            var = orgList[varList.index(varListSort[l])]
            Helper.p2pLn(var[0], var[1], Dia=1., mNam=lnName, plus=0,
                             color="green", GlyphType=4, scaleGly=3, project=True, fontScale=0)
            if len(orgList[0]) == 3:
                ldName = f"{lnName}d"
                Helper.p2pLn(var[0], var[2], Dia=1., mNam=ldName, plus=0,
                                 color="green", GlyphType=4, scaleGly=3, project=True, fontScale=0)
                length = Helper.line2Cyl(ldName)[3]
            else:
                length = Helper.line2Cyl(lnName)[3]
            # lf = Helper.p2pDst(var[1], var[0])
            messageBox.setText(
                f"Found {n} {lnName}s, is this {lnName}(={length})? ")
            messageBox.setInformativeText(
                "Yes:is Pdw;No:continue;Reset:select again. ")
            messageBox.setStandardButtons(
                messageBox.No | messageBox.Yes | messageBox.Reset | messageBox.Retry)
            messageBox.setDefaultButton(messageBox.Yes)
            choice = messageBox.exec_()
            if choice == messageBox.Yes:
                Helper.nodsDisplay(lnName, lDia=1., color="purple")
                if len(orgList[0]) == 3:
                    Helper.nodsDisplay(ldName, lDia=1., color="purple")

                Helper.nodsDel(f"{lnName}_ps")
                return length
            elif choice == messageBox.Reset:
                l = -1
                Helper.nodsDel(f'{lnName}*')
            elif choice == messageBox.Retry:
                length = Helper.adjLine(lnName)
                if len(orgList[0]) == 3:
                    length = Helper.adjLine(lnName)
                return
            else:
                Helper.nodsDel(f'{lnName}*')
                if l == n-1:
                    l = -1
    #         continue
            l += 1

    @staticmethod
    def lineChange(mNam, changeStp=1, changeDrt=False):

        modNode = ut.getNode(mNam)
        p0 = np.zeros(3)
        p1 = np.zeros(3)
        modNode.GetNthControlPointPositionWorld(0, p0)
        modNode.GetNthControlPointPositionWorld(1, p1)
        Helper.nodsDel(mNam)
        if changeDrt is True:
            Helper.p2pLn(p1, p0, Dia=1., mNam=mNam, plus=changeStp,
                             color="green", GlyphType=4, scaleGly=3, project=True)
        else:
            Helper.p2pLn(p0, p1, Dia=1., mNam=mNam, plus=changeStp,
                             color="green", GlyphType=4, scaleGly=3, project=True)
        return

    @staticmethod
    def adjLine(lnName, n=1.0):
        # n = 1
        # data = Helper.line2Cyl("L")
        # p0 = data[0]
        # p1 = data[1]
        while n == 1.0:
            messageBox = qt.QMessageBox()
            messageBox.setWindowTitle(" /!\ WARNING /!\ ")
            messageBox.setIcon(messageBox.Warning)
    #     lineChange(lnName,changeStp=10)
            messageBox.setText(f"Adjust Line")
            messageBox.setInformativeText(
                "Yes:is Pdw;No:continue;OK:select again. ")
            messageBox.setStandardButtons(
                messageBox.No | messageBox.Yes | messageBox.Ok | messageBox.Cancel)
            messageBox.setDefaultButton(messageBox.Ok)
            choice = messageBox.exec_()
            if choice == messageBox.Cancel:
                Helper.lineChange(lnName, changeDrt=True, changeStp=n)
            elif choice == messageBox.Yes:
                Helper.lineChange(lnName, changeStp=n)
            elif choice == messageBox.No:
                Helper.lineChange(lnName, changeStp=-1.0*n)
            else:
                n = 0
                return Helper.line2Cyl(lnName)[3]

    @staticmethod
    def getThrs(p1, p2, vol, rad=2, delSeed=True):
        '''阈值种子
        para: - pos: position;
              - vol: volume;
              - rad: the rad of the Cylinder;
              - h: the High of the Cylinder;
              - delSeed: delete the Seed or not.
        return:mean,min,max,the thredsheld array of the Cylinder.
        '''
        Helper.p2pCyl(p1, p2, rad=rad, plus=5, mNam="cCyl", Seg=3)
        # Helper.line2Cyl("cLine",True)
        thrDic = Helper.gradeScrew("cCyl", vol.GetID(), getPos=False)[0]
        voxList = []
        for key in thrDic.keys():
            voxList.append(thrDic[key]["Vox"])
        voxArray = np.array(voxList)
        if delSeed is True:
            Helper.nodsDel("cCyl")
        # np.around(np.mean(voxArray)),min(voxList),max(voxList),
        return voxArray

    @staticmethod
    def p2lineProj(p = None, p1=[0, 0, 0], p2=[0, 5, 5], lNod=None, drt=None):

        if lNod is not None:
            p1, p2 = ut.array(lNod)
        elif drt is not None:
            p2 = Helper.p2pLn(p1, drt=drt,)[0]
        p_p1 = Helper.p2pDst(p1, p)
        p_p2 = Helper.p2pDst(p2, p)
        p1_p2 = Helper.p2pDst(p2, p1)
        d = npl.norm(np.cross(p2-p1,p-p1)/Helper.p2pDst(p2, p1))
        if p_p1 > p_p2:
            P1 = p1
            P2 = p2
            Pl = p_p1
        else:
            P1 = p2
            P2 = p1
            Pl = p_p2
        p_d = (Pl**2-d**2)**.5
        if p_d <= p1_p2:
            dd = Helper.p2pLn(P2, P1, plus=-p_d,mNam="dd")[0]
        else:
            dd = Helper.p2pLn(P1, P2, plus=p_d-p1_p2,mNam="dd")[0]
        # dd1 = Helper.p2pexLine(p,dd,plus=0 ,mNam="dd1")
        return dd, Helper.p2pLn(p,dd)[1]

    @staticmethod
    def lines2P(p0, p01, p1, p11):
        '''lines2P 2线交点

        '''
        drt00 = np.array([p01[0]])
        drt01 = np.array([p01[1]])
        drt02 = np.array([p01[2]])
        drt10 = np.array([p11[0]])
        drt11 = np.array([p11[1]])
        p0x = np.array([p0[0]])
        p0y = np.array([p0[1]])
        p0z = np.array([p0[2]])
        p1x = np.array([p1[0]])
        p1y = np.array([p1[1]])
        t = (drt10 * (p1y - p0y) + drt11 * (p0x - p1x)) / ((drt10 * drt01) - (drt00 * drt11))
        x = p0x + drt00 * t
        y = p0y + drt01 * t
        z = p0z + drt02 * t
        return np.array([x[0], y[0], z[0]])


    @staticmethod
    def p2pDst(p1,p2):
        """
        2点的距离
        """
        p1 = ndA(p1)
        p2 = ndA(p2)
        return npl.norm(p1 - p2)
    @staticmethod
    def flat_key(layer):
        """ Example: flat_key(["1","2",3,4]) -> "1[2][3][4]" """
        if len(layer) == 1:
            return layer[0]
        else:
            _list = [f"${k}" for k in layer[1:]]
            return layer[0] + "".join(_list)

    @staticmethod
    def loadNpy(filepath):
        x = np.load(filepath, mmap_mode='r')
        return x


    @staticmethod
    def lineFitPoints(psarr, cP=None, mNam='axis'):
        '''
        '''
        psarr = ndA(psarr)
        # 1. 求质心
        ct = np.mean(psarr, axis=0)
        # 2. 减质心
        data = psarr-ct
        # 3. SVD
        _, _, vv = npl.svd(data)
        # 4. 取线的矢量
        # 最大奇异值对应的右奇异向量...😓🙅🏻‍♀️不知道啥意思
        drt = vv[0]
        # 5. 点(质心)+矢量得到直线
        # length = len(psarr)+2  # 线的长度
        length = max(psarr.max() - psarr.min())
        if cP is None:
            cP = ct
        p3 = cP + vv[0] / npl.norm(vv[0]) * length
        line = Helper.p2pLn(p3, cP, mNam=mNam,
                                plus=length/2)  # Slicer项目里面专用
        return line

    @staticmethod
    def flatDict(_dict):
        if not isinstance(_dict, dict):
            raise TypeError(
                "argument must be a dict, not {}".format(type(_dict)))

        def __flat_dict(pre_layer, value):
            result = {}
            for k, v in value.items():
                layer = pre_layer[:]
                layer.append(k)
                if isinstance(v, dict):
                    result.update(__flat_dict(layer, v))
                else:
                    result[Helper.flat_key(layer)] = v
            return result
        return __flat_dict([], _dict)

# ## 常用`Fun`

# #### vtkCutPla Fun

# +

    @staticmethod
    def vtkCutPla(p0=None,
                  p1=None,
                  sp = None,
                  mNam = "",
                  rad = 15,
                  drt = None,
                  normal = None,
                  cPName=None,
                  way = "lP",
                  p3 = None ,
                  modPd=None,
                  show = False,
                  **kw
                  ):
        '''vtkCutPla vtk切片

        vtk切片

        Args:
            p0:cyl p0(定义圆柱)
            p1:cyl p1(决定圆柱的向量)
            pp:closestP in plane(取最近点)
            nor: 改变截面方向(默认的截面为圆柱圆面)
            mNam:object mNam
            p3:
        return: way=="lP" PA; way == "cP" centroidP; way == "mP"
                cP, cnnArr, xxx, modDp
        '''
        xxx = []
        wDic=dict(cP='large',lP='all')
        if p1 is not None:
            drt = Helper.p2pLn(p0,p1)[1]
        if normal is None:
            normal = drt
        if modPd is None:
            modPd = Helper.cylClip(p0, drt, mNam, rad)
        cnnPd = Helper.plaCut(p0, modPd, sp, normal, cPName, show, color = "blue", lineW = 8)

        # connPd = cnnPd.GetPoints().GetData()
        cnnArr = Helper.getArr(cnnPd)
        cP = np.mean(cnnArr,axis=0)
        if way == "cP":
            cP = np.average(cnnArr, axis=0)
            # _, xxx = Helper.closestP(p0, cnnArr) #
            if show:
                # if cP is not None:
                Helper.addFids(cP,mNam = f"{cPName}_cp")
        else:
            arr = Helper.getArr(cnnArr, segLen=5)
            cps = Helper.closestP(arr[0], arr[1:],sortArr=True)[1][:11]
            cps = np.asarray([cps[0].tolist()]+cps.tolist())
            cP = Helper.closestP(p3, cps)
            if show:
                Helper.addFids(cP,mNam = "lP")
                Helper.addFids(cps,mNam = "lPs")
            # cnnArr = axx
        return cP, cnnArr, modPd

# ### closestP
# - [[closestP]]
    @staticmethod
    def closestP(ps0: PS,
                 ps1: PS,
                 minAx: Lit['min','max'] = 'min',
                 indP: bool=False
                 )-> Union[float, PS]:
        """Calculates the distance between points in a given point cloud.

        Args:
            ps0 (PS): Source point
            ps1 (PS): Target point cloud
            minAx (str, optional): Specifies whether to look for 'min' or 'max'. Defaults to 'min'.
            idP (bool): Specifies whether to return the index in addition to the result.

        Returns:
            The nearest/farthest point, or the index of the result minus the source point

        Note:
            - _description_
        """
        ps0 = np.asarray(ps0)
        ps1 = np.asarray(ps1)
        xx = npl.norm(ps1 - ps0, axis = 1)
        ind = [np.argmin(xx), np.argmax(xx)][int(minAx == 'max')]
        if indP:
            return ps1[ind], ind
        else:
            return ps1[ind]


    @staticmethod
    def plaClip(mPd: NOD,
                norm: PS=None,
                cp: PS=None,
                pla: NOD=None,
                mNam: str=""
                )->NOD:
        """Plane clip a model with a plane.
        这段代码的作用是使用vtkClipPolyData函数对输入的模型mod进行裁剪，根据输入的法向量norm和中心点cp（如果pla不为None，则从pla中获取norm和cp）来定义裁剪平面，并将裁剪后的结果存储在clippedDp中。
        Parameters:
        mod (NOD): The vtkMRMLModelNode object to be clipped.
        norm (PS, optional): The normal of the plane, default to None.
        cp (PS, optional): The center point of the plane, default to None.
        pla (NOD, optional): Same as norm and cp, instead of giving those two you can give a vtkMRMLPlaneWidget directly.
        mNam (str, optional): Result's name, default to "clipMod"

        Return:
        NOD: The clipped model.
        """
        mPd = Helper.getPd(mPd)
        if pla is not None:
            pla = Helper.namNod(pla)
            norm = np.asarray(pla.GetNormal())
            cp = np.asarray(pla.GetCenter())
        plane = vtk.vtkPlane()
        plane.SetNormal(norm) # 注意法向量和留下的部分一致
        plane.SetOrigin(cp)

        clip = vtk.vtkPolyDataPlaneClipper()
        clip.SetClipFunction(plane)
        clip.SetInputData(mPd)
        clip.Update()
        clippedDp = clip.GetOutput(0)
        return Helper.pdMDisp(clippedDp, mNam)

    @staticmethod
    def plaCut(p0: PS,
               modPd: Union[MOD, vtk.vtkPolyData],
               pp: PS=None,
               normal:PS=None,
               exMode: Lit['All','Lg','Clo', None]='Lg',
               mNam: str="",
               **kw)-> VPD:
        '''plaCut 切平面

        用平面切取

        Args:
            p0 (_type_): point in pla
            modPd (polydata): model Pd
            pp (_type_): a point in vertical linevertical line with p0
            nor (_type_): the plane normal
            cPName (str): 莫名

        Returns:
            - _description_

        Note:
            - _description_
        '''
        modPd = Helper.getPd(modPd)
        print(normal)
        plane = vtk.vtkPlane()
        plane.SetOrigin(p0)
        plane.SetNormal(normal)
        cutter = vtk.vtkCutter()
        cutter.SetInputData(modPd)
        cutter.SetCutFunction(plane)
        cutter.Update()

        conn = cutter.GetOutput()

        if pp is None:
            cnnPd = Helper.cnnEx(conn, exMode=exMode)
        else:
            cnnPd = Helper.cnnEx(conn, pp)
        return Helper.pdMDisp(cnnPd, mNam)

    @staticmethod
    def cnnEx(pD,
              sp = None,
              exMode: Lit['All','Lg','Clo', None]='Lg',
              clean = False,
              mNam=''
              ):
        # NOTE: preventive measures: clean before connectivity filter
        # to avoid artificial regionIds
        # It slices the surface down the middle
        pD = Helper.getPd(pD)
        if clean:
            surfer = vtk.vtkDataSetSurfaceFilter()
            surfer.SetInputData(pD)
            surfer.Update()

            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(surfer.GetOutput())
            cleaner.Update()
            pD = cleaner.GetOutput()

        cnn = vtk.vtkPolyDataConnectivityFilter()
        cnn.SetInputData(pD)
        if sp is not None and exMode == 'Clo':
            cnn.SetClosestPoint(sp)
            cnn.SetExtractionModeToClosestPointRegion()
        else:
            if exMode == "Lg":
                cnn.SetExtractionModeToLargestRegion()
            elif exMode == "All":
                cnn.SetExtractionModeToAllRegions()
            elif exMode == "All":
                cnn.SetExtractionModeToAllRegions()
                # regions = connect.GetNumberOfExtractedRegions()
                # connect.SetExtractionModeToSpecifiedRegions()
                # cnn = []
                # for i in range(0, regions, 1):
                #     print("region: ", i)
                #     connect.InitializeSpecifiedRegionList()
                #     connect.AddSpecifiedRegion(i)
                #     connect.Update()
                #     cnn.append(connect.GetOutput())
                # print(f"{len(cnn)=}")
        cnn.Update()
        # leaves phantom points ....
        if clean:
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(cnn.GetOutput())
            cleaner.Update()
            return cleaner.GetOutput()
        return Helper.pdMDisp(cnn.GetOutput(), mNam)

    @staticmethod
    def cylClip(p0: PS,
                drt: PS,
                # pd: VPD,
                mPd: NOD,
                rad: float,
                crpIn=True,
                # **kw
               )->VPD:
        pd = Helper.getPd(mPd)
        vCyl = vtk.vtkCylinder()
        vCyl.SetRadius(rad)
        vCyl.SetCenter(p0)
        vCyl.SetAxis(drt)
        vClip = vtk.vtkClipPolyData()
        vClip.SetInputData(pd)
        vClip.SetClipFunction(vCyl)
        if crpIn:
            vClip.InsideOutOn()
            vClip.GenerateClippedOutputOn()
        vClip.Update()
        return vClip.GetOutput()
        # cnn = Helper.cnnEx(vCliPd, sp, exMode)
        # Helper.pdMDisp(cnn, mNam, **kw)
        # return vCliPd, cnn

    # -

    # ### obbBox

    @staticmethod
    def obbBox(nodArr: Union[NOD, np.ndarray],
            mNam: str="r",
            xyzSort = False,
            mtype: Lit["pla","roi"]="roi",
            grid: bool=False,
            **kw: any
            )->any:
        """get obbBox from ndarray
        para:
        return:
            - cp,cn,cns,xyz,gNode,pxyz
        """
        nodArr = Helper.getArr(nodArr)
        gNod = None
        # if nodArr.shape[1] == 2:
        #     nodArr = nodArr[..., np.newaxis]
        cov = np.cov(nodArr,
                y      = None,
                rowvar = 0,
                bias   = 1)
        _, vect = npl.eig(cov)
        tvect = np.transpose(vect)
        points_r = np.dot(nodArr,
                    npl.inv(tvect))

        co_min = np.min(points_r, axis=0)
        co_max = np.max(points_r, axis=0)

        xmin, xmax = co_min[0], co_max[0]
        ymin, ymax = co_min[1], co_max[1]
        zmin, zmax = co_min[2], co_max[2]

        x_x = xmax - xmin
        y_y = ymax - ymin
        z_z = zmax - zmin

        xdif = (x_x) * 0.5
        ydif = (y_y) * 0.5
        zdif = (z_z) * 0.5

        # xDim = npl.norm(x_x)
        # yDim = npl.norm(y_y)
        # zDim = npl.norm(z_z)
        # dia = min(xDim,zDim)

        cx = xmin + xdif
        cy = ymin + ydif
        cz = zmin + zdif

        corners = np.array([
                            [cx - xdif, cy - ydif, cz - zdif],
                            [cx - xdif, cy + ydif, cz - zdif],
                            [cx - xdif, cy + ydif, cz + zdif],
                            [cx - xdif, cy - ydif, cz + zdif],
                            [cx + xdif, cy - ydif, cz - zdif],
                            [cx + xdif, cy + ydif, cz - zdif],
                            [cx + xdif, cy + ydif, cz + zdif],
                            [cx + xdif, cy - ydif, cz + zdif],
                            ])
        cn = np.dot(corners, tvect)
        cp = np.dot([cx, cy, cz], tvect)
        _, xDrt, xDst, _= Helper.p2pLn(cn[0], cn[4], mNam=f"{mNam}_x")
        _, yDrt, yDst, _= Helper.p2pLn(cn[0], cn[1], mNam=f"{mNam}_y")
        _, zDrt, zDst, _= Helper.p2pLn(cn[0], cn[3], mNam=f"{mNam}_z")
        drts= ndA([xDrt, yDrt, zDrt])
        dsts= ndA([xDst, yDst, zDst])
        if xyzSort:
            drtInx = np.argmax(abs(drts), axis=1)
            dsts = dsts[drtInx]
            drts = drts[drtInx]
        gArr = None
        if grid is True:
            gArr = Helper.gridPS(cn[0], np.array([cns[0][1],cns[1][1]]), np.array(drts[:2]))

            # Helper.addLog(f"{len(gArr)=}")
        if mNam != "":
            # if mtype == 'roi':
            gNod = Helper.roi(cn[0], dsts, drts, f"{mNam}_roi")
            zR = Helper.roi(cp, [dsts[0], dsts[1], 1      ], drts, color='blue')
            yR = Helper.roi(cp, [dsts[0], 1      , dsts[2]], drts, color='blue')
            zR = Helper.roi(cp, [1      , dsts[1], dsts[2]], drts, color='blue')
        return drts, dsts, cn, cp # , gNod, gArr

#%%
    @staticmethod
    def pdHull(dataPs: Union[PS,NOD,VPD],
               mNam: str='',
               **kw
               ):
        dataPs = Helper.pds2Mod(dataPs)
        pa = vtk.vtkPassArrays()
        pa.SetInputConnection(reader.GetOutputPort())
        pa.AddArray( 0, 'Array1Name' )
        convexHull = vtk.vtkDelaunay3D()
        convexHull.SetInputData(pa.GetOutput())
        outerSurface = vtk.vtkGeometryFilter()
        outerSurface.SetInputConnection(convexHull.GetOutputPort())
        outerSurface.Update()
        pd = outerSurface.GetOutput()
        return pd, Helper.pdMDisp(pd,mNam, **kw)

    @staticmethod
    def pdPush(dataPs,
                drtP:PS=None,
                mNam: str='',
                order: bool=False,
                hull: bool=False,
                drt: PS=None,
                dist: float=0,
                sid2: bool=False,
                **kw
                ):
        dataPs = Helper.pds2Mod(dataPs, order=order, inOut='out')
        # dataPs = dataPs - np.mean(dataPs,axis=0)
        if drtP is None:
            drt = Helper.getArr(drt)
            drtP = drt*dist
        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputData(dataPs)
        extrude.SetExtrusionTypeToNormalExtrusion()
        extrude.SetVector(tuple(drtP))
        extrude.Update()
        pd = extrude.GetOutput()
        if hull is True:
            pd, _ = Helper.pdHull(pd)
        model = Helper.pdMDisp(pd,mNam, **kw)
        if sid2:
            pd = Helper.modTrans(model,moveArr = drtP *-.5)
        return pd

    @staticmethod
    def pds2Mod(dataP,
                mNam: str='',
                patch: bool=False,
                hull: bool=False,
                order: bool=False,
                **kw
                )-> VPD:
        '''pds2Mod 点数据模特

        点数据形成VPD模特

        Args:
            dataP (Union[PS,NOD,VPD]): 点数据(arr及nod点)
            mNam (str, optional): 莫名. Defaults to ''.

        Returns:
            - VPD模特

        Note:
            - _description_
        '''
        dataP = Helper.getArr(dataP, order)
        points = vtk.vtkPoints()
        # [TODO]model 2 mesh
        pg = vtk.vtkPolygon()
        polygon_pid = pg.GetPointIds()
        if patch:
            points, pg = Helper.patchPla(dataP)
        else:
            for i, p in enumerate(dataP):
                points.InsertNextPoint(*p.tolist())
                polygon_pid.InsertNextId(i)
        polygon_cell = vtk.vtkCellArray()
        polygon_cell.InsertNextCell(pg)
        pd = VPD()
        pd.SetPoints(points)
        pd.SetPolys(polygon_cell)
        if hull is True:
            pd = Helper.pdHull(pd)
        return Helper.pdMDisp(pd,mNam, **kw)
        # return pd


    @staticmethod
    def patchPla(dataP
                 ):

        iPlus=0
        points = vtk.vtkPoints()
        pg = vtk.vtkPolygon()
        polygon_pid = pg.GetPointIds()
        for i, p in enumerate(dataP):
            ii = (i+1)%len(dataP)
            p_p1 = Helper.p2pDst(dataP[ii],p)
            if i == 0:
                bdl = p_p1
            budao = p_p1/bdl
            if budao > 2:
                p_p1P = Helper.p2pLn(p, dataP[ii], plus=0)[2]
                # Helper.addFids([p, dataP[ii]],'Ls')
                for iP in range(int(budao)):
                    points.InsertNextPoint(*p_p1P(bdl*iP).tolist())
                    polygon_pid.InsertNextId(i+iP+iPlus)
                iPlus += int(budao)-1
            else:
                points.InsertNextPoint(*p.tolist())
                polygon_pid.InsertNextId(i+iPlus)
        return points, pg

    @staticmethod
    def plasCut(arrs: PS,
                mPd: Union[NOD,VPD,str],
                cP: PS=None,
                mNam: str='',
                surClose: bool=False
                )->VPD:
        '''plaCut 裁切
        多平面裁切
        Args:
            arrs (list): 多面阵(含: 矢和靶)
            mPd (NOD): 大模
            cP (PS, optional): 中心点. Defaults to None.
            mNam (str, optional): 莫名. Defaults to ''.
            surClose (bool, optional): 闭裁. Defaults to False.
        Returns:
            形数(VTK PolyData object): 多平面裁切后的VTK PolyData
        '''
        # -预 E: preprocessing
        mPd = Helper.getPd(mPd) # -转形数 E: convert to polydata
        if cP is None: # -无中 E: no center
            cP = np.mean(Helper.getArr(mPd), axis=0) # -中心 E: center
        def __closeSeg(mPd, plas):
            '''__closeSeg 闭裁'''
            clip = vtk.vtkClipClosedSurface() # -闭裁 E: clip
            clip.SetInputData(mPd) # -输入 E: input data
            clip.SetClippingPlanes(plas) # -裁切 E: cut
            clip.Update() # -更新 E: update
            return clip.GetOutput() # -返形数 E: return polydata
        # -多面裁 E: segment with planes
        norm = Helper.p2pLn(arrs[0], arrs[1])[1] # -矢 E: vector
        if surClose: # -闭裁 E: cut mPd with close surface
            plas = vtk.vtkPlaneCollection() # -面集 E: plane collection
            for i, arr in enumerate(arrs): # -迭面 E: loop planes
                pla = vtk.vtkPlane() # -设面 E: set vtkPlane
                # 确定normal, arrs为2个点的坐标, 保证normal朝向cP
                if i == 1:
                  norm *= -1
                # norm = arr*np.sign(np.dot(arrs[0],cP-arrs[1])) #
                pla.SetNormal(norm) # -矢 E: vector
                pla.SetOrigin(arr) # -靶 E: target
                plas.AddItem(pla) # -集面 E: add plane
            mPd = __closeSeg(mPd, plas) # -闭裁 E: cut mPd with close surface
        else: # -切片 E: cut onebyone
            while arrs: # -迭面 E: loop planes
                arr = arrs.pop(-1) # -平面 E: plane
                if len(arr) == 1:
                  norm *= -1
                mPd = Helper.plaClip(mPd, norm, arr) # -直裁 E: cut mPd with plane
        # -修选形数 E: simpify and extract the polydata
        cvPd = Helper.cnnEx(mPd,exMode='Lg')
        # -返数模 E: return polydata or model
        return Helper.pdMDisp(cvPd, mNam, color='yellow')

    @staticmethod
    def roi(cp: PS,
            dim: float,
            drts: PS,
            mNam: str ="",
            **kw
            ):
      '''roi 生成ROI

      生成ROI

      Args:
          cp (PS): 中心点
          dim (float): 尺寸
          drts (PS): 方向
          mNam (str, optional): _description_. Defaults to "".

      Returns:
          - _description_

      Note:
          - _description_vbs = sitkUtils.PullVolumeFromSlicer(vbs)
vbs = sitkUtils.PullVolumeFromSlicer(vbs)

      '''
      roi = SCENE.AddNewNodeByClass(
          "vtkMRMLMarkupsROINode",mNam)
      roi.SetName("ROI")
      roi.SetSize(dim)
      # roi.SetCenter(centerP)
      obbRAS = cp+0.5*(dim[0] * drts[0] + dim[1] * drts[1] + dim[2] * drts[2])
      b2Rt = np.row_stack((np.column_stack((drts[0], drts[1], drts[2], obbRAS)),
                          (0, 0, 0, 1)))
      b2RtMtx = ut.vtkMatrixFromArray(b2Rt)
      roi.SetAndObserveObjectToNodeMatrix(b2RtMtx)
      Helper.nodsDisplay(roi,
                          opacity=0.2,
                          color="cyan",
                          gDia=2,
                          lock=False
                          )
      return roi


    @staticmethod
    def p2PlaDst(ps: PS,
                    norm: PS=None,
                    cp: PS=None,
                    dst: bool=False,
                    plNod: NOD=None,
                    )-> any:
        '''p2PlaDst 点到平面的距离

        点的平面的距离

        Args:
            plNod (planeNode): 平面节点
            ps (PS): 探测点

        Returns:
            - 点到平面的距离逻辑值(1:平面前,0:上,-1:后)或者距离值

        Note:
            - 返回点到平面的距离（正数表示点在平面法向量一侧；反之相反）
        '''
        if plNod is not None:
            pla = Helper.namNod(plNod)
            norm = np.asarray(pla.GetNormal())
            cp = np.asarray(pla.GetCenter())
        plane = vtk.vtkPlane()
        plane.SetOrigin(cp)
        plane.SetNormal(norm)
        return [plane.EvaluateFunction(ps), plane.DistanceToPlane(ps)][int(dst)]

    @staticmethod
    def psFitPla(arr: PS,
                 mNam: str="",
                 )->any:
        '''psFitPla 点云拟合平面

            根据输入的PS坐标数组，计算平面的法向量和中心位置，
            如果输入mNam参数，则会在Slicer中创建一个PlaneNode节点，
            并将计算出来的法向量和中心位置赋值到该节点上。

        Args:
            arr (PS): 点云.
            mNam (str, optional): 是否显示平面. Defaults to "".

        Returns:
            - 平面(正方向)

        Note:
            - _description_
        '''
        arr = np.asarray(arr)
        cp = np.mean(arr,axis=0)
        norm = npl.svd((arr-cp).T)[0][:,-1]
        norm = norm*[1,-1][int(norm[2]<0)]
        if mNam!="":
            Helper.getPla(norm, cp, mNam)
        return norm, cp

    @staticmethod
    def getPla(norm: PS,
               cp: PS,
               mNam: str=""
               )->any:
        plaNod = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsPlaneNode')
        plaNod.SetCenter(cp)
        plaNod.SetNormal(norm)
        plaNod.SetName(mNam)

    @staticmethod
    def gridPS(ps: PS, spcs   = 1
                    )-> PS:
        p0, p1     = ps.min(), ps.max()
        dim        = ps.shape[0]
        spcs       = (spcs, spcs, spcs)\
                        if isinstance(spcs,int)\
                        else spcs
        lSpcs      = []
        for i     in range(dim):
            lSpcs += [np.linspace(p0[i], p1[i],
                            (p1[i] - p0[i] + 1)\
                        /   spcs[i])]
        xyz        = np.meshgrid(*lSpcs)
        xyz        = np.array(xyz).T.reshape(-1, dim)
        return xyz

    @staticmethod
    def gridMod(mNam, grid, pxyz, cn, xyz, cns, **kw): # , xDs, yDs, zDs, xDist, yDist, zDist):
      """2d grid and mod
      """
      # if grid:
      for ii in range(int(xyz[1])+2):
          if ii != int(xyz[1]+1):
              cnny = Helper.p2pLn(cn[0],drt=cns[1][1])[2](ii)
          else:
              cnny = Helper.p2pLn(cn[0],drt=cns[1][1])[2](xyz[1])
          for iii in range(int(xyz[2])+2):
              if iii != int(xyz[2]+1):
                  pxyz.append(Helper.p2pLn(cnny,drt=cns[2][1])[2](iii))
              else:
                  pxyz.append(Helper.p2pLn(cnny,drt=cns[2][1])[2](xyz[2]))
      return np.asarray(pxyz)
      #  else: # [TODO] 未测试
          #  if mNam != "":
              #  plaDate = ut.array(Helper.p3Box(cn[0].tolist(),cns[0][0].tolist(),cns[1][0].tolist(), mNam=mNam).GetID())
              #  gNode = Helper.pdP'u's'h(plaData, cns(2))
          #  if grid:
              #  for i in range(int(xDist)+1):
              #      cnn = Helper.p2pexLine(cn[0],drt=xDs[1])[2](i) # 沿着x轴遍历
              #      for ii in range(int(yDist)+1):
              #          cnny = Helper.p2pexLine(cnn,drt= yDs[1])[2](ii) # 沿着y轴遍历
              #          for iii in range(int(zDist)+1):
              #              pxyz.append(Helper.p2pexLine(cnny,drt=zDs[1])[2](iii)) # 沿着z轴遍历
              #  if show:
              #      Helper.addFids(pxyz,Dia=.3,mNam=f"{mNam}_grid")
      # return # gNode

    @staticmethod
    def psCont(plaMod: Opt[MOD]=None,
                norm: PS=None,
                cp: PS=None,
                mNam: str="",
                inOut: Lit['in','out']='in',
                sn: int= 120,
                cArr: bool=True,
                **kw
                )-> np.ndarray:
        """psCont 判断闭合+生成mod轮廓

        从点云arr提取轮廓

        Args:
            mod:
            arr (arr): 点云
            norm (arr): 平面方向
            cp (arr): 平面中点
            mNam (str, optional): 模名. Defaults to "".
            inOut (str, optional): 内外轮廓?. Defaults to "in".

        Returns:
            - cps (np.ndarray)
        """
        pData = Helper.getPd(plaMod)
        if cArr:
            arr = Helper.getArr(plaMod)
        # Helper.addLog(f'{type(pData)=}')
            if cp is None:
                cp = np.mean(arr,axis=0)
            if norm is None:
                norm = Helper.p3Nor(cp,arr[0], arr[len(arr)//3])
        crArr = Helper.p3Cir(cp, rad = 1, norm = norm, sn=sn, mNam='cir') # ref垂直旋转圆
        cps = []
        for p in crArr:
            p0 = Helper.p2pLn(cp,p)[0]
            data = Helper.p2pCylClip(cp,p0,.5, pData,exMode="All")[int(inOut=='out')]
            if data is not None:
                cps.append(data)
            # except:
            #     pass
        if cps is None:
            Helper.addLog('没数据呀')
            return
        Helper.addLog(f"{len(cps)=}")
        cps = np.asarray(cps)
        cp = np.mean(np.unique(cps, axis=0), axis=0)
        # Helper.addLog(f'pas = {np.round(pas, 2)}')
        return cps, cp, Helper.pds2Mod(cps,mNam=mNam)

    @staticmethod
    def psMod(plaMod: MOD=None,
              arr: PS=None,
              norm: PS=None,
              cp: PS=None,
              ind0: int=0,
              mNam: str="",
              sn: int=88,
              inOut: bool=True,
              ck: bool=True,
              **kw
              )-> Union[bool,Tuple[PS, PS, bool]]:
        """psMod 判断闭合+生成mod轮廓

        从点云arr提取轮廓

        Args:
            mod:
            arr (arr): 点云
            norm (arr): 平面方向
            cp (arr): 平面中点
            mNam (str, optional): 模名. Defaults to "".
            inOut (str, optional): 内外轮廓?. Defaults to "in".
            ck(bool, optional): 判断闭合吗?

        Returns:
            - _description_

        Notes:
            - 无法射到平面模特
        """
        if arr is None:
            arr = Helper.getArr(plaMod)
        if cp is None:
            cp = np.mean(arr,axis=0)
        if norm is None:
            norm = Helper.p3Nor(cp,arr[0], arr[len(arr)//3])
        if plaMod is None:
            plaMod = Helper.getPd(arr)
        crArr = Helper.p3Cir(cp, rad = 1, norm = norm, sn=sn) # ref垂直旋转圆
        maxL = Helper.closestP(cp, crArr,'max')
        maxDst = npl.norm(cp-maxL)
        cpsIn = []
        cpsOut = []
        # for p in enumerate(crArr):
        for p in crArr:
            # p = crArr[ind0+[1,-1][int(i%2)]]
            p0 = Helper.rayCast(cp, p , plaMod, plus = maxDst, oneP = False)
            if len(p0)==0 and ck == 1:
                return False
            cpsIn.append(p0[0])
            cpsOut.append(p0[-1])
        if ck == 1:
            return True
        if mNam != "":
            return Helper.pds2Mod(np.array([cpsOut, cpsIn][int(inOut)]),mNam=mNam, hull=0)

        return cpsIn, cpsOut, len(cpsIn)==sn

    @staticmethod
    def inMod(points,modNode):
        """ Check that the points of X are inside modNode
        如果是一个点返回逻辑值
        否则返回在mod内的点

        """
        inPs=[]
        modNode = Helper.namNod(modNode)
        if isinstance(modNode,  SMD):
            modNode = modNode.GetPolyData()
        loc=vtk.vtkCellLocator()
        loc.SetDataSet(modNode)
        loc.BuildLocator()
        points = ndA(points)
        if points.ndim==1:
            inOut = loc.FindCell(points)> -1
            return inOut
        else:
            for k, point in enumerate(points):
                inside = loc.FindCell(point)> -1
                if inside is True:
                    inPs.append(point)
            return inPs


    @staticmethod
    def getPd(aNod: Union[PS, str, MOD, VPD,VOL]
              )->VPD:
        """
        Converts Numpy Array to vtkPolyData

        Parameters
        ----------
        coords: array, shape= [number of points, point features]
            array containing the point cloud in which each point has three coordinares [x, y, z]
            and none, one or three values corresponding to colors.

        Returns:
        --------
        PolyData: vtkPolyData
            concrete dataset represents vertices, lines, polygons, and triangle strips

        """
        if isinstance(aNod, np.ndarray):
            Npts, _ = np.shape(aNod)
            Points = vtk.vtkPoints()
            # ntype = vtk.util.numpy_support.get_numpy_array_type(Points.GetDataType())
            coords_vtk = numpy_to_vtk(aNod, deep=1)
            Points.SetNumberOfPoints(Npts)
            Points.SetData(coords_vtk)

            Cells = vtk.vtkCellArray()
            ids = np.arange(0,Npts, dtype=np.int64).reshape(-1,1)
            IDS = np.concatenate([np.ones(Npts, dtype=np.int64).reshape(-1,1), ids],axis=1)
            ids_vtk = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(IDS, deep=True)

            Cells.SetNumberOfCells(Npts)
            Cells.SetCells(Npts,ids_vtk)
            pd = VPD()
            pd.SetPoints(Points)
            pd.SetVerts(Cells)
            return pd
        elif isinstance(aNod, str):
            return Helper.namNod(aNod).GetPolyData()
        elif isinstance(aNod, SMD):
            return Helper.namNod(aNod).GetPolyData()
        elif isinstance(aNod, VPD):
            return aNod
        elif isinstance(aNod, VOL):
            return Helper.namNod(aNod).GetImageData()
        else:
            raise TypeError(f"type {type(aNod)} not supported")

    @staticmethod
    def p2pCylClip(p0: PS,
            p1: PS,
            pN: PS = None,
            rad: float = 1,
            mPd=None,
            normal=None,
            mNam="",
            exMode: Lit['All','Lg','Clo', None]='Lg',
            sp = None,
            rot = 0
            ):
        '''cylClip 短柱切

        点点柱切

        Args:
            p0 (_type_): p0
            p1 (_type_): 01
            rad (float): 柱半径
            dataset (vtk): 料
            normal (_type_, optional): 方向. Defaults to None.
            mNam (str, optional): 莫名. Defaults to "".
            exMode (str, optional): 切取方式(largest,all). Defaults to "largest".
            rot (int, optional): 旋转角度. Defaults to 0.
        return:
            inArr,outArr,mod,cnnArr
        '''
        mod = None
        angle = lambda v1, v2:\
                    math.acos(np.dot(v1, v2)\
                  / (npl.norm(v1) * npl.norm(v2)))\
                  * (180 / math.pi)
        if normal is None:
            normal = Helper.p2pLn(p0,p1)[1]
        if pN is None:
          pN = [0,1,0]
        rotationaxis = np.cross(pN, normal)
        rotationangle =  angle(pN, normal) + rot

        transform = vtk.vtkTransform()
        transform.Translate(p0)
        transform.RotateWXYZ(rotationangle, rotationaxis)
        transform.Inverse()

        mPd = Helper.getPd(mPd)

        cylinder = vtk.vtkCylinder()
        cylinder.SetRadius(rad)
        cylinder.SetTransform(transform)

        plane0 = vtk.vtkPlane()
        plane0.SetOrigin(p0)
        plane0.SetNormal([-x for x in normal])
        plane1 = vtk.vtkPlane()
        plane1.SetOrigin(p1)
        plane1.SetNormal(normal)

        clipfunction = vtk.vtkImplicitBoolean() # 定义一个合集隐式布尔函数
        clipfunction.SetOperationTypeToIntersection()
        clipfunction.AddFunction(cylinder)
        clipfunction.AddFunction(plane0)
        clipfunction.AddFunction(plane1)

        clipper = vtk.vtkClipPolyData() # vtkClipPolyData()的结果是一个多边形
    #     clipper.SetInput(dataset)

        clipper.SetInputData(mPd)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOn()

        clipper.SetClipFunction(clipfunction)
        clipper.Update()
        pd = clipper.GetOutput()
        vData = Helper.cnnEx(pd, exMode = exMode,clean = 1)
        if vData.GetPoints() is None:
            print("no point vData")
            return None, None,None,None
        connPd = vData.GetPoints().GetData()
        cnnArr = vtk_to_numpy(connPd)
        xxx = cdist(cnnArr, np.array([p0], dtype='float32'))
    #     minDist.append(in(xxx[:,0]))
        inArr  = cnnArr[xxx[:,0].tolist().index(min(xxx[:,0]))]
        outArr = cnnArr[xxx[:,0].tolist().index(max(xxx[:,0]))]

        if mNam != "":
            mod = Helper.pdMDisp(vData, mNam = mNam, color="blue")
            # Helper.p2pexLine(p0, p1, dia = rad*2, plus=0,mNam=f"{mNam}_cyl", opacity = .3)
            # Helper.addFids(cnnArr, f'{mNam}_ps')

        return inArr,outArr,mod,cnnArr

    # #### mpdsIst
    # - mpdsIst polyData交集
    def mpdsIst(mPd1, mPd2, mNam='', **kw):
        '''mpdsIst polyData交集

        Args:
            *mPd (vtk): polyData
            mNam (str, optional): 莫名. Defaults to ''.
        '''
        mPd1 = Helper.getPd(mPd1)
        mPd2 = Helper.getPd(mPd2)
        istFilter = vtk.vtkIntersectionPolyDataFilter()
        istFilter.SetInputData(0, mPd1)
        istFilter.SetInputData(1, mPd2)
        istFilter.Update()
        return Helper.pdMDisp(istFilter.GetOutput(), mNam=mNam, **kw)
        # sitk的布尔运算方法包括: AND, OR, XOR, NOT, NAND, NOR, and XNOR
        # 比如交集的代码:
        # sitk.AndImageFilter().Execute(img1, img2)
        # cylinder of sitk
    @staticmethod
    def modTrans(smPd, moveArr=np.zeros(3),scale=(1,1,1), agl = 0, nor = (0,0,1), delMod=True, mNam=''):
        """
        Applies scaling, translation on a given vtk actor.
        # [TODO]rx=0,ry=0,rz=0,
        """
        pd = Helper.getPd(smPd)
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Scale(*scale)
        dx,dy,dz = moveArr
        transform.Translate(dx, dy, dz)
        transform.RotateXYZ(agl, nor)
        transformfilter = vtk.vtkTransformFilter()
        transformfilter.SetTransform(transform)
        transformfilter.SetInputData(pd)
        transformfilter.Update()
        pd = transformfilter.GetOutput()

        if isinstance(smPd, SMD):
            if mNam == '':
                mNam = smPd.GetName()
            if delMod:
                SCENE.RemoveNode(smPd)
            Helper.pdMDisp(pd, mNam)
        else:
            Helper.pdMDisp(pd, mNam)
        return pd

    @staticmethod
    def rotNod(p0, p1, normal, rot, __angle):
        """旋转nod
        """
        def __angle(v1, v2):
            norVect = lambda x: math.sqrt(np.dot(v, v))
            return math.acos(np.dot(v1, v2) / (norVect(v1) * norVect(v2)))
        if normal is None:
            normal = Helper.p2pLn(p0,p1)[1]
        rotationaxis = np.cross([0, 1, 0], normal)
        rotationangle = (180 / math.pi) * __angle([0, 1, 0], normal) + rot

        transform = vtk.vtkTransform()
        transform.Translate(p0)
        transform.RotateWXYZ(rotationangle, rotationaxis)
        transform.Inverse()
        return transform

# ### anoPrc
# - [[anoPrc]]
    @staticmethod
    def anoPrc(txt = None, dic=None, txtP = None, side = "L",
               dispMods = None, hidMods = None, sec=1):
        """字幕事件(3dslice字幕+显示目标显示)
            - param:

            - return:

            - None:
                - 字幕(可选):
                    - 左上滚动字幕-->先定义空字典(xdic={}),
                      然后(xdic=anoPrc(...dic=xdic))
                - 显示(可选):
        """
        if dispMods:
            Helper.nodsDisp(dispMods, True)
        if hidMods:
            Helper.nodsDisp(hidMods)
        view=slicer.app.layoutManager().threeDWidget(0).threeDView()
        if not dic:
            dic = {}
            view.cornerAnnotation().ClearAllTexts()

        def __annSet(side, txt, color = 'pink', bold = 0, size = 18, dic=None):
            cLocs = {
                    # 'lL': vtk.vtkCornerAnnotation.LowerLeft,
                    # 'lR': vtk.vtkCornerAnnotation.LowerRight,
                    'L': vtk.vtkCornerAnnotation.UpperLeft,
                    'R': vtk.vtkCornerAnnotation.UpperRight,
                    'lw': vtk.vtkCornerAnnotation.LowerEdge,
                    'up': vtk.vtkCornerAnnotation.UpperEdge,
                    # 'L': vtk.vtkCornerAnnotation.LeftEdge,
                    # 'R': vtk.vtkCornerAnnotation.RightEdge,
                    }
            view.cornerAnnotation().SetText(cLocs[side],txt)
            view.cornerAnnotation().GetTextProperty().SetColor(Helper.myColor(color))
            view.cornerAnnotation().GetTextProperty().SetBold(bold)
            view.cornerAnnotation().SetMaximumFontSize(size)
            # if not txtDic:
            dic = Helper.lsDic({side : txt}, lDic = dic)
            print (dic)
            return dic

        # Set text to "Something"
        if txt:
            __annSet("lw", txt)
            dic = __annSet(side, txt + "\n", dic=dic)
            for k in dic.keys():
                __annSet(k, dic[k])
        if txtP:
            __annSet("up", txtP,"pink", 1, 28)
            #uLtext = ((textP) if textP else "  " )+ text + "\n"
        # __annSet(cLoc, text, "purple", 0, 18)
        # Update the view
        view.forceRender()
        slicer.app.processEvents()
        time.sleep(sec)
        return dic

# ### modFoc
# - [[modFoc]]
    @staticmethod
    def modFoc(mod=None, factor = 1.0, sNod = None):
        if mod:
            mod = Helper.namNod(mod)
            bds=[0,0,0,0,0,0]
            mod.GetRASBounds(bds)
            nodCen=[(bds[1]+bds[0])*.5,
                    (bds[3]+bds[2])*.5,
                    (bds[5]+bds[4])*.5,]
            x = bds[1]-bds[0]
            y = bds[3]-bds[2]
            z = bds[5]-bds[4]
        # if sNod != None:
            # sliceNodes = ut.getNodes(volName)
        sNods = ut.getNodes('vtkMRMLSliceNode*')
        if sNod is not None:
            sNods = ut.getNodes(sNods[sNod])
        for sNod in list(sNods.values()):
            sliceWidget = slicer.app.layoutManager().sliceWidget(sNod.GetLayoutName())
            sliceWidget.sliceLogic().FitSliceToAll()
            newFOVx = (sNod.GetFieldOfView()[0] if not mod else x) / factor
            newFOVy = (sNod.GetFieldOfView()[1] if not mod else y) / factor
            newFOVz = (sNod.GetFieldOfView()[2] if not mod else z) / factor
            sNod.SetFieldOfView(newFOVx, newFOVy, newFOVz)
            sNod.UpdateMatrices()
        if mod:
            slicer.vtkMRMLSliceNode.JumpAllSlices(
            SCENE, *nodCen, slicer.vtkMRMLSliceNode.CenteredJumpSlice)


    # ### closedCheck(arr)
    # > 判断是否闭合
    def shouGongSha(arr,
                    ps = True,
                    order = True,
                    segLen = 9,
                    cP=None,
                    ):
        """
        """
        # [TODO]是否考虑凹多边形质心出墙的情况?
        if order is True:
            arr5 = arr[0::segLen]
            arr = np.concatenate((arr5[0::2],arr5[1::2][::-1]),axis=0) # 排序
            if ps:
                return arr
        if cP is None:
            cP = np.mean(arr, axis = 0) # 质心
        # 取角
        angles = []
        for i,v in enumerate(arr):
            if i == 0:
                Nor = Helper.pn3Angle(v, cP, arr[len(arr)//3])[1] # 平面法向量
                angle = 0
            if i > 0:
                angle = Helper.pn3Angle(arr[0], cP, v, Nor)[0]
            angles.append(angle)
        # 依据角排序
        cpsSorarr = Helper.arrSortArr(arr, angles)
        cpsSorarr.dtype = 'float32'
        sgs = np.array_equal(np.round(arr,2),np.round(cpsSorarr,2)) # 判断出轨
        return cpsSorarr, sgs, arr

    @staticmethod
    def arrSortArr(arr1: PS,
                   arr0: PS,
                   rev: bool=False
                   )->PS:
        '''arrSortArr 跟哥走

        按照arr0排序arr1

        Args:
            arr1 (PS): 弟弟
            arr0 (PS): 哥哥

        Returns:
            - 排好的弟弟

        Note:
            - _description_
        '''
        if not isinstance(arr1,list):
            arr1 = arr1.tolist()
        if not isinstance(arr0,list):
            arr0 = arr0.tolist()
        zip_a_b = zip(arr0,arr1)
        sorted_zip = sorted(zip_a_b, key=lambda x:x[0], reverse=rev)
        result = zip(*sorted_zip)
        _, cpsSort = [list(x) for x in result]
        return np.array(cpsSort)
    # +

    @staticmethod
    def sectP(ps,anglex=10):
        # angles = []
        vps = []
        cp = np.mean(ps,axis=0)
    #     Angles = []
        for i,v in enumerate(ps):
            if i == 0:
                Nor = Helper.pn3Angle(v, cp, ps[1])[1] # 平面法向量
                Angle_s = [0,]
            if i > 0:
                angle = Helper.pn3Angle(ps[i+1 if i < len(ps)-1 else 0], v, ps[i-1], Nor)[0]
                Angle_s.append(angle)
                if angle>180+anglex and Angle_s[-2]>90:
                    vps.append(v)
        pss = ps.tolist()
        vPs = []
        for i, v in enumerate(vps): # 去掉连续点留下最后那个
            if abs(pss.index(v.tolist())-pss.index(vps[i-1].tolist()))>1:
                vPs.append(v)
        return np.array(vPs)

    # -
    # ### p3Angle
    # > 三点求夹角

    # +

    @staticmethod
    def pn3Angle(p_a, p_b, p_c, N=None):
        """3点求角度(3D)
        parame:
        return: ∠abc(0°-360°)
        """
        p_a = p_a.astype(np.float64)
        p_b = p_b.astype(np.float64)
        p_c = p_c.astype(np.float64)
        # p1 = Helper.p2pexLine(p_b,p_a)[1]
        # p2 = Helper.p2pexLine(p_b,p_c)[1]
        if N is None:
            N = Helper.p3Nor(p_a,p_c,p_b) # 3点平面的法向量
            # N = n/npl.norm(n)

        a_x, b_x, c_x = p_a[0], p_b[0], p_c[0]  # 点a、b、c的x坐标
        a_y, b_y, c_y = p_a[1], p_b[1], p_c[1]  # 点a、b、c的y坐标
        a_z, b_z, c_z = p_a[2], p_b[2], p_c[2]  # 点a、b、c的z坐标

        # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
        x1,y1,z1 = (a_x-b_x),(a_y-b_y),(a_z-b_z)
        x2,y2,z2 = (c_x-b_x),(c_y-b_y),(c_z-b_z)

        dot = x1*x2 + y1*y2 + z1*z2
        det = x1*y2*N[2] + x2*N[1]*z1 + N[0]*y1*z2 - z1*y2*N[0] - z2*N[1]*x1 - N[2]*y1*x2
        angle = math.atan2(det, dot)

        deg = np.degrees(angle)

        return deg + 360 if deg < 0 else deg , N

    # ### createScrew
    @staticmethod
    def createScrew(p,drt = None, pt = None, dia = None, obbtree = None, mNam = None, **kw):
        '''createScrew 生螺钉

        2点或点+方向生成螺钉

        Args:
            p (_type_): p
            drt (_type_, optional): 方向. Defaults to None.
            pt (_type_, optional): p1. Defaults to None.
            dia (_type_, optional): diameter. Defaults to None.
            obbtree (vtk, optional): obbtree. Defaults to None.
            mNam (str, optional): 莫名. Defaults to None.

        Returns:
            - Dia, Length, Pb, Pbb
            -

        Note:
            - 生成螺钉和最窄区毛豆
        '''
        # ,mNam=f"{x}_axepP")[0]
        Dia = ((dia-1)*.8 // 0.5) * 0.5
        if drt is not None:
            P = Helper.p2pLn(p, drt=drt, mNam=mNam + "L")[2]
            Pb = Helper.rayCast(P(54), p, obbtree)
            Helper.addFids(Pb,mNam+"b")
            Pbb = Helper.rayCast(P(-54), p, obbtree)
            Helper.addFids(Pbb,mNam+"bb")
        else:
            Pb = Helper.rayCast(Helper.p2pLn(
                pt, p)[0], p, obbtree=obbtree)
            Pbb = Helper.rayCast(Helper.p2pLn(
                p, pt)[0], p, obbtree=obbtree)
        length = Helper.p2pDst(Pb,  Pbb)
        Length = (length * .9) // 1
        line = Helper.p2pLn(
                                Pbb,
                                drt=drt,
                                # PA,
                                dia=Dia,
                                plus=Length,
                                mNam=mNam,
                                lock=True,
                                )
        pta, psa = Helper.screwAngle(mNam)

        return Dia, Length, pta, psa, Pb, Pbb, line[-1]

    @staticmethod
    def minProj(pm: PS,
                segArr: np.ndarray,
                norm: PS,
                mNam: str="mProj"
                ) -> any :
        """最小截面
            pararm:
                - pm(arr): p on pla
                - segArr(arr): arr from nod
                - norm(arr): norm of pla
                - pp(arr): will be projed
                - inOut(bool): 'in' or 'out'
            return:
                - contPs(arr): ps of sectPla
        """
        aps = Helper.projPla(segArr,pm, norm, )
        # cp0 = Helper.p2pexLine(pm, aps[0])[2]
        Helper.p3Cir(pm, rad = 1, norm = norm, mNam=f"{mNam}cri") # ref垂直旋转圆
        # self.tDic({'len(aps)': len(aps)})
        Helper.addLog(f"apslen={len(aps)}")
        return aps, Helper.psCont(aps, norm=norm, order=False, cp = pm, mNam=f"mproj"),  # 沿着垂圆遍历找最近点,并生成轮廓

        Helper.testMs()

    @staticmethod
    def rot3Dveiw(axis = "X", rotEnd = 90, rotStart = 0):
        """
        Acquire a set of screenshots of the 3D view while rotating it.
        rotAxis = 0 --> yaw(x); =1 --> pitch(y); =2 --> roll(z)
        """
        drtDic = dict(Z = 0, X = 1, Y = 2)
        rotAxis = drtDic[axis]
        lm, viewNod = Helper.reset3Dview()
        renView = None
        for widgetIndex in range(lm.threeDViewCount):
            view = lm.threeDWidget(widgetIndex).threeDView()
            if viewNod == view.mrmlViewNode():
                renView = view
                break
        if not renView:
            raise ValueError('Selected 3D view is not visible in the current layout.')
        # renderView = viewFromNode(viewNode)
        # Save original orientation and go to start orientation
        oriInc = renView.pitchRollYawIncrement
        oriDrt = renView.pitchDirection
        renView.setPitchRollYawIncrement(-rotStart)
        if rotAxis == 0:
            renView.yawDirection = renView.YawRight
            renView.yaw()
        elif rotAxis == 1:
            renView.pitchDirection = renView.PitchDown
            renView.pitch()
        else:
            renView.rollDirection = renView.RollRight
            renView.roll()
        # Rotate step-by-step
        rotationStepSize = rotEnd - rotStart
        renView.setPitchRollYawIncrement(rotationStepSize)
        if rotAxis == 0:
            renView.yawDirection = renView.YawLeft
        elif rotAxis == 1:
            renView.pitchDirection = renView.PitchUp
        else:
            renView.rollDirection = renView.RollLeft
        # for offsetIndex in range(steps):
        if rotAxis == 0:
            renView.yaw()
        elif rotAxis == 1:
            renView.pitch()
        else:
            renView.roll()
        # Restore original orientation and rotation step size & direction
        if rotAxis == 0:
            renView.yawDirection = renView.YawRight
            renView.yaw()
            renView.setPitchRollYawIncrement(rotEnd)
            renView.yaw()
            renView.setPitchRollYawIncrement(oriInc)
            renView.yawDirection = oriDrt
        elif rotAxis == 1:
            renView.pitchDirection = renView.PitchDown
            renView.pitch()
            renView.setPitchRollYawIncrement(rotEnd)
            renView.pitch()
            renView.setPitchRollYawIncrement(oriInc)
            renView.pitchDirection = oriDrt
        else:
            renView.pitchDirection = renView.RollRight
            renView.roll()
            renView.setPitchRollYawIncrement(rotEnd)
            renView.roll()
            renView.setPitchRollYawIncrement(oriInc)
            renView.rollDirection = oriDrt

    @staticmethod
    def reset3Dview():
        viewNod = SCENE.GetNodeByID('vtkMRMLViewNode1')
        wig3D = slicer.app.layoutManager().threeDWidget(0)
        view3D = wig3D.threeDView()
        view3D.resetFocalPoint()
        return slicer.app.layoutManager(),viewNod

    @staticmethod
    def camFoc(pos, drt = 'A', n = 1):
        '''\u955c\u5934\u805a\u7126
        parame: - position
                - distance
                - approch
                    - A: antior;
                    - R: right;
                    - S: up;
        return:
        '''
        dDic = dict(R = np.array([1,0,0]), A = np.array([0,1,0]), S = np.array([0, 0, 1]))
        camera = SCENE.GetNodeByID('vtkMRMLCameraNode1')
        if isinstance(pos, str):
            ndA(pos)
        camera.SetFocalPoint(*pos)
        vct = pos*dDic[drt]*n
        camera.SetPosition(vct)
        camera.SetViewUp([0, 0, 0])
        camera.ResetClippingRange()
        return

    @staticmethod
    def mod2Seg(mNam):
        modelNode = slicer.util.getNode(mNam)
        # Create segmentation
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes() # only needed for display

        # Import the model into the segmentation node
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segmentationNode)
        image = slicer.vtkOrientedImageData()
        segmentationNode.GetBinaryLabelmapRepresentation(segmentID, image)
        # print(image)
        #
        return slicer.util.array(image)


    @staticmethod
    def cam3DZoom(nod, factor = 1, drt = None):
        node =  Helper.namNode(nod)
        threedView = slicer.app.layoutManager().threeDWidget(0).threeDView()

        # Get volume center
        bounds=[0,0,0,0,0,0]
        node.GetRASBounds(bounds)
        nodCen = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
        # Shift camera to look at the new focal point
        renView = threedView.renderWindow()
        renderer = renView.GetRenderers().GetFirstRenderer()
        camera = renderer.GetActiveCamera()
        oldFoc = camera.GetFocalPoint()
        oldPos = camera.GetPosition()
        camOff = [nodCen[0]-oldFoc[0], nodCen[1]-oldFoc[1], nodCen[2]-oldFoc[2]]
        camera.SetFocalPoint(nodCen)
        camera.SetPosition(oldPos[0]+camOff[0], oldPos[1]+camOff[1], oldPos[2]+camOff[2])
        camera.Zoom(factor)
        if drt:
            Helper.camFoc(nodCen,drt)

    # + endofcell="--"
    # ### 三维转二维
    # -

    @staticmethod
    def getTPed(arr: np.ndarray,
                pm: PS,
                obb: OB,
                mNam: str='',
                n: int=3,
                order: bool=True
                )-> Union[bool, np.ndarray, None]:
        ''
        def __getKs(xyArr: np.ndarray)-> np.ndarray:
            '''getKs 计算曲率

            由2D曲线Arr求曲率

            Args:
                xyArr (np.ndarray): 2D曲线曲率

            Returns:
                - 各点曲率

            Note:
                - _description_
            '''
            xT = np.gradient(xyArr[:,0])
            xTT = np.gradient(xT)
            yT = np.gradient(xyArr[:,1])
            yTT = np.gradient(yT)
            return np.abs(xTT * yT - xT * yTT) / (xT * xT + yT * yT) ** 1.5

        def __removeSmall(valKs: np.ndarray,
                            ps: list
                            )-> np.ndarray:
            '''__removeSmall 去掉污点

            去掉污点

            Args:
                valKs (np.ndarray): 曲率表
                ps (list): 凹点表

            Returns:
                - 凹点表

            Note:
                - 用于递归去重
            '''
            sps = []
            vs = valKs.take(ps)
            vals = np.copy(vs)
            for j, ij in enumerate(ps):
                jj = (j+1) % len(ps)
                if abs(ps[jj]-ij)<3:
                    sps.append(np.where(vs==vals[[jj,j][int(vals[jj]<vals[j])]])[0][0])
                    vals[jj] = vals[[jj,j][int(vals[jj]<vals[j])]] # 大值传递
            return np.delete(ps,sps)

        def __getConcaves(valKs: np.ndarray,
                            arrSn: np.ndarray,
                            obb: OB,
                            )-> list:
            '''getConcaves 获取凹点

            获取凹点

            Args:
                valKs (np.ndarray): 曲率
                arr (np.ndarray): 3D排序Arr
                obb (OB): obb

            Returns:
                - 凹点ind

            Note:
                - _description_
            '''
            ps=[]
            tS = np.percentile(valKs, 66)
            posX = np.where(valKs > tS)[0]
            for i in posX:
                ii = (i+1)%len(arrSn)
                mp = (arrSn[i-1]+arrSn[ii])/2
                outB = Helper.modIn(mp,arrSn[i],obb,6)[0]
                if outB == 1 :
                    ps.append(i)
            ps = np.asarray(ps)
            print(len(ps))
            if len(ps)>2:
                ps = __removeSmall(valKs, ps)
                print(ps)
            return ps

        def __getRidofRib(arrSn: np.ndarray,
                            pm: np.ndarray,
                            mNam: str,
                            ps: list,
                            )-> np.ndarray:

            arr1 = arrSn[ps[0]:ps[1]]
            cp1 = np.mean(arr1,axis=0)
            arr2 = np.delete(arrSn, [range(ps[0]+1, ps[1]+1)], 0)
            cp2 = np.mean(arr2,axis=0)
            arrs = [[arr1, cp1], [arr2, cp2]]
            if len(arr1)<=len(arr2):
                arrs = [arrs[1],arrs[0]]
            tArr = arrs[1]
            if len(arrs[0][0])>len(arrs[1][0])*2:
                tArr = arrs[0]
            else:
                tArr = [arrs[1],arrs[0]][int(Helper.p2pDst(pm,arrs[0][1])<Helper.p2pDst(pm,arrs[1][1]))]
            pd = Helper.pds2Mod(tArr[0], mNam ,True, )
            return tArr[1], Helper.getArr(pd)

        arrSn = Helper.getArr(arr, order, segLen=n)
        lK = Helper.lenK(arrSn, segLen=n)[0]
        if lK is False: # [0::9]+order=1 似乎是最优解
            print (lK)
            return False
        arrSn = Helper.getArr(arrSn, 0, segLen=n)
        arr2D = np.delete(Helper.projPla(arrSn), 1, 1)

        valKs = __getKs(arr2D)
        ps = __getConcaves(valKs, arrSn, obb)
        if len(ps) == 2:
            return __getRidofRib(arrSn, pm, mNam, ps)
        elif len(ps) < 2:
            return
        else:
            print(ps)
            return False
    @staticmethod
    def lenK(arr: np.ndarray,
             order: bool=False,
             segLen: int=1,
             closed: bool=True,
             k = 3,
            )-> float:
        '''lenK 曲线比

        最长:平均段长

        Args:
            arr (np.ndarray): 曲线矩阵

        Returns:
            - 曲线比

        Note:
            - 用于判断繁杂不均的曲线
        '''
        arr = Helper.getArr(arr,order,segLen)
        if closed:
            arr1=np.asarray(arr[1:].tolist()+[arr[0].tolist()])
            dists = npl.norm(arr-arr1, axis=1)
        else:
            dists = npl.norm(arr[:-1]-arr[1:], axis=1)
        vKs = dists/np.mean(dists)
        for i, vk in enumerate(vKs):
            if vk > k:
                return False, i
        else:
            return True, None

    def rot2P1(pc: np.ndarray,
                norm: np.ndarray,
                p0: np.ndarray,
                p1: np.ndarray,
                n: int,
                )-> np.ndarray:
        '''rot2P1 旋转至p1

        旋转至p1

        Args:
            pc (np.ndarray): 圆心
            norm (np.ndarray): 法向量
            p0 (np.ndarray): p0
            p1 (np.ndarray): p1
            n (int): 边数

        Returns:
            - cir

        Note:
            - _description_
        '''
        # -造圆; E: get a circle 'cir'
        cir = Helper.p3Cir(pc, norm, p1, sn=n)
        # -旋至p0; E: rotate to p0
        def __rot2P0(cir: np.ndarray,
                    p0: np.ndarray,
                    )-> np.ndarray:
            '''rot2P0 旋转至p0

            旋转至p0

            Args:
                cir (np.ndarray): cir
                p0 (np.ndarray): p0

            Returns:
                - cir

            Note:
                - _description_
            '''
            # -旋至p0; E: rotate to p0
            cir = cir - pc #
            cir = cir.dot(Helper.getRotM(norm, Helper.p2pDst(p0, pc)))
            cir = cir + pc
            return cir
        cir = Helper.rot2P0(cir, p0)
        return cir

# 三维空间- 已知: 正多边形(pc为圆心,norm为法向量,n为边数, 半径为pc到 p1的距离)cir,
# - p1于同一平面, 不一定是cir的顶点,
# - 以pc为中心, 在cir所在的平面旋转cir,
# - 旋转至cir的第一个点p0和p1点重合,
# - 用Python + vtk,建一个函数rot2P1,
# - 参数为cir 和 p1, 返回cir的定点坐标数组
# - Step by Step…please
    @staticmethod
    def rot2P1(p1: PS,
               cir: VPD=None,
               mNam='',
               pc: PS=None,
               norm: PS=None,
               sn: int=None,):
        # 获取正多边形的圆心pc、法向量norm、边数n、半径r
        if cir is not None:
            cArr = Helper.getArr(cir)
            pc = np.mean(cArr, axis=0)
            norm = Helper.p3Nor(cArr[0], cArr[len(cArr)//3], cArr[len(cArr)//3*2])
            sn = len(cArr)
        else:
            cArr = Helper.p3Cir(pc, norm, p1=p1, sn=sn)
        r = Helper.p2pDst(p1, pc)
        # 获取正多边形的第一个点p0
        p0 = cArr[0]
        # 计算p1和p0的夹角α
        alpha = Helper.angle3P(pc, p0, p1)
        def __rotate_circle(center, angle):
            # Create a transform object
            transform = vtk.vtkTransform()
            # Set the center of rotation
            transform.Translate(center[0], center[1], 0)
            # Rotate the circle
            transform.RotateZ(angle)
            # Return the transform
            return transform
        # 旋转圆
        transform = __rotate_circle(pc, alpha)
        # 旋转圆上的点
        ctArr = [transform.TransformPoint(p) for p in cArr]
        Helper.psMod(ctArr, mNam)
        return ctArr

#%%
#%%
END = 0



