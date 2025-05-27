"""pppUtil.py: Everything ppp-related for PPP (Optimized Version).

__author__ = "Jumbo Jing"
"""
#%%
import os
import sys
import time
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from functools import wraps
from numpy.linalg import norm
from itertools import combinations as ittCom
from typing import Callable, Any
import itertools as itt
import scipy.ndimage as ndIm
from scipy.ndimage import center_of_mass as ndCp, \
    binary_dilation as dila, \
    binary_erosion as erod, \
    label as scLb, \
    sum_labels as lbSum
from scipy.spatial import cKDTree as kdT
from scipy.optimize import minimize as sOpt
from scipy.spatial.transform import Rotation as Rt
from queue import Queue
from collections import defaultdict as dfDic
import slicer
from sitkUtils import PullVolumeFromSlicer, PushVolumeToSlicer
import SimpleITK as sitk

# 设置numpy打印选项
np.set_printoptions(precision=4, suppress=True)

# 常量定义
SCEN = slicer.mrmlScene
SNOD = SCEN.AddNewNodeByClass
SVOL = "vtkMRMLScalarVolumeNode"
LVOL = "vtkMRMLLabelMapVolumeNode"

# 简写函数
puSk = PullVolumeFromSlicer
skPu = PushVolumeToSlicer

# 标签字典
LBLVS = {
    1: 'C1',  2: 'C2',  3: 'C3',    4: 'C4',   5: 'C5',   6: 'C6',   7: 'C7',
    8: 'T1',  9: 'T2',  10: 'T3',  11: 'T4',  12: 'T5',  13: 'T6',  14: 'T7',
    15: 'T8', 16: 'T9',  17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',  21: 'L2',
    22: 'L3', 23: 'L4',  24: 'L5',  25: 'S1',  26: 'Sacrum', 27: 'Cocc',
    28: 'Cord', 29: 'L6', 50: 'Vbs'}

TLDIC = {14: 'T7',
         15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12',
         20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'S1',
         26: 'Sc', 27: 'vBs'}

TLs_ = ['T7', 'T8', 'T9', 'T10', 'T11', 'T12',
        'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'Sc', 'vBs']

# 基本向量
OP = np.zeros(3)
EPS = 1e-6
NX = np.array([1., 0, 0])
NY = np.array([0., 1, 0])
NZ = np.array([0., 0, 1])
XYZ = np.array([NX, NY, NZ])

# 类型定义
from typing import Sequence, Tuple, Optional, Union, Literal
P = Sequence[Tuple[float, float, float]]
PS = Sequence[P]
Opt = Optional
Seq = Sequence
Lit = Literal

# VTK类型
VPD = vtk.vtkPolyData
MD = slicer.vtkMRMLModelNode
MKS = slicer.vtkMRMLMarkupsNode
VOL = slicer.vtkMRMLVolumeNode
NOD = (MD, MKS, VOL)

print('import')

def isLs(ls): 
    return isinstance(ls, (list, tuple, np.ndarray))

def sNam(nam, suf):
    if nam == '':
        return nam
    else:
        return nam+'_'+suf

def ndA(*x):
    """将输入转换为NumPy数组"""
    if len(x) == 1:
        arr = x[0]
    else:
        arr = x

    if arr is None:
        return np.array([])

    if isinstance(arr, list):
        arr = [item for item in arr if item is not None]

    try:
        return np.asanyarray(arr)
    except ValueError:
        if isinstance(arr, (list, tuple)):
            return np.array(arr, dtype=object)
        raise

def getNods(nods):
    """多节点-->节点列表"""
    def list1_(ls):
        return [l for l in list(ls)
                if isinstance(l, list)
                for l in l]
    nods = slicer.util.getNodes(nods, useLists=True)
    return list1_(nods.values())

def nodsDisp(nods='*', disp=False, cls=None):
    """隐藏/显示节点"""
    if cls is None:
        cls = (MD, MKS)
    for nod in getNods(nods):
        if isinstance(nod, cls):
            nod.CreateDefaultDisplayNodes()
            nod.GetDisplayNode().SetVisibility(disp)

def dspNod(nod, mNam='', cls=None, color=None, opacity=None):
    """显示节点"""
    if mNam != '':
        nod = getNod(nod, mNam=mNam)
        nod.CreateDefaultDisplayNodes()
        dpNod = nod.GetDisplayNode()
        return dpNod

def isNdas(*arrs):
    """判断是否为ndarray"""
    ndAs = []
    for arr in arrs:
        ndAs += [getArr(arr),]
    if len(ndAs) == 1:
        return ndAs[0]
    return ndAs

def nx3ps_(ps, n=3, flat=True):
    """转换为Nx3形式的点集"""
    ps = getArr(ps)
    shp = ps.shape
    if shp == (n,):
        return ps
    if shp[-1] != n:
        pns = ps.reshape(-1, n)
    else:
        pns = ps
    if pns.size == n:
        return pns.reshape(n)
    if flat:
        if pns.ndim > 2:
            return pns.reshape(np.prod(shp[:-1]), n)
    return pns

def rgN_(x1=10, stp=1, x0=0): 
    return np.arange(x0, x1, stp)[:, None]

def psDst(p_ps=None, nor=True):
    """计算点距离"""
    p_ps = ndA(p_ps)
    if p_ps.shape == (3,):
        p_ps = ndA(p_ps)
        dts = norm(p_ps)
        if dts < EPS:
            pass
        if nor:
            dts = (p_ps/(dts+EPS), dts)
    else:
        dts = norm(p_ps, axis=1)
        if (dts < EPS).any():
            pass
        if nor:
            dts = (p_ps/(dts+EPS)[:, None], dts)
    return dts

def uNor(n): 
    """单位化向量"""
    return psDst(n, 1)[0]

def p2pXyz(nors=NX, sort=True, pp=None, mNam=''):
    """点对点生成v, w, u坐标系"""
    if pp is not None:
        pp = ndA(pp)
        shp = pp.shape
        if shp == (2, 3):
            p0 = pp[0]
            nors = psDst(pp[1]-pp[0])[0]
    else:
        p0 = pp
    nors = ndA(nors)
    if nors.ndim == 2:
        nor, zNor = nors
    else:
        nor = nors
        zNor = OP
        zId = np.argmin(np.abs(nor))
        zNor[zId] = 1
    yNor = np.cross(nor, zNor)
    yNor = uNor(yNor)
    zNor = np.cross(nor, yNor)
    zNor = uNor(zNor)
    if sort is False:
        return yNor, zNor
    vwu = ndA([nor, yNor, zNor])
    vwu = ndA([v if v[np.argmax(np.abs(v))] >= 0 else -v for v in vwu])
    vwu = vwu[np.argsort(np.argmax(vwu, axis=1))]
    return vwu

def lnPs(l=np.arange(0, 10), p=OP, n=ndA([NX, NY, NZ]), flat=False):
    """生成线上点集"""
    if not isinstance(l*1., float):
        l = ndA(l)[:, None]
    if p.ndim == 2:
        p = p[:, None]
    if n.ndim == 2:
        n = n[:, None]
        if not isinstance(l*1., float):
            nl = len(l)*len(n)
            gps = np.reshape(l*n, (nl, 3)) + p
    else:
        gps = l*n + p
    if flat:
        return nx3ps_(gps)
    return gps

def lpn_(l, p, n, flat=False):
    """线性插值点"""
    gps = p+n*(l if isinstance(l*1., float) else l[:, None])
    if flat:
        return nx3ps_(gps)
    return gps

def lnNod(p0, p1, mNam='', dia=1.):
    """创建线节点"""
    lnod = SNOD("vtkMRMLMarkupsLineNode", mNam)
    lnod.AddControlPoint(vtk.vtkVector3d(p0))
    lnod.AddControlPoint(vtk.vtkVector3d(p1))
    lnod.CreateDefaultDisplayNodes()
    dspNod = lnod.GetDisplayNode()
    dspNod.SetLineDiameter(dia)
    return dspNod

def p2pLn(p0: P, p1: Opt[P] = None, nor: Opt[P] = None, plus: float = 0.,
          flat: bool = False, mNam: str = '', dia: float = 1., **kw):
    """点对点生成线"""
    p0 = ndA(p0)*1.
    if p1 is not None:
        p1 = ndA(p1)*1.
        nor, dst = psDst(p1-p0)
        dst += plus*1.
    else:
        nor = nor/norm(nor)
        dst = 1. if plus==0. else plus*1.
    pt = dst*nor + p0
    if flat and pt.ndim == 3:
        shp0, shp1, _ = pt.shape
        pt = pt.reshape(shp0*shp1, 3)
    if mNam != "":
        lNod = SCEN.AddNewNodeByClass("vtkMRMLMarkupsLineNode", mNam)
        lNod.AddControlPoint(vtk.vtkVector3d(p0))
        lNod.AddControlPoint(vtk.vtkVector3d(pt))
        lNod.SetNthControlPointVisibility(1, 0)
        dspNod = lNod.GetDisplayNode()
        dspNod.SetCurveLineSizeMode(1)
        dspNod.SetLineDiameter(dia)
        dspNod.UseGlyphScaleOff()
        dspNod.SetGlyphType(6)
        dspNod.SetGlyphSize(dia)
        if mNam[-1] == '_':
            p3Cone(pt, nor, rad=dia*1.4, high=dia*4, mNam=sNam(mNam, "pt"))
    return pt, nor, dst, lambda l=dst, p=p0, n=nor: p+l*n

def p3Cone(bP, drt=None, mNam='', rad=1, high=3, seg=6, hP=None, rP=None, *kw):
    """创建圆锥"""
    if drt is None:
        rad = norm(rP-bP)
        vec = rP-bP
        high = norm(vec)
        drt = vec/high
    cone = vtk.vtkConeSource()
    cone.SetResolution(seg)
    cone.SetCenter(bP)
    cone.SetRadius(rad)
    cone.SetHeight(high)
    cone.SetDirection(drt)
    cone.Update()
    conePD = cone.GetOutput()
    return getNod(conePD, mNam=mNam)

def findPs(pds, p, mTyp: Union['min', 'max'] = 'min'):
    """找到最近/最远点"""
    ps_, p = getArr(pds), ndA(p)
    if ps_.ndim > 2:
        ps = nx3ps_(ps_)
    else:
        ps = ps_
    dsts = norm(ps-p, axis=1)
    if mTyp == 'min':
        mId = np.argmin(dsts)
    elif mTyp == 'max':
        mId = np.argmax(dsts)
    if ps_.ndim > 2:
        mIds = np.unravel_index(mId, ps_.shape[:-1])
        return ps[mId], dsts[mId], mIds
    if p.ndim > 1:
        pss = nx3ps_(p)
        return [findPs(ps, p_) for p_ in pss]
    else:
        return ps[mId], dsts[mId], mId

def psRoll_(ps, p0): 
    """滚动点集"""
    return np.roll(ps, -findPs(ps, p0)[-1], 0)

def psFitPla(ps, sd=-1, mNam=''):
    """拟合平面法向量"""
    ps = getArr(ps)
    cp = np.mean(ps, axis=0)
    mNor = np.linalg.svd((ps-cp).T)[0][:, sd]
    mNor = uNor(mNor)
    return mNor

def pdCln(Pd):
    """清理PolyData"""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(Pd)
    cleaner.Update()
    return cleaner.GetOutput()

def dotPlnX(pds, pln, eqX=0, rtnPjx=False, isIn=False):
    """平面裁剪点集"""
    pds = getArr(pds)
    ps = nx3ps_(pds)
    
    if isinstance(pln, (tuple, list, np.ndarray)):
        if len(pln) == 2:
            op, nor = ndA(pln)
        else:
            nor = ndA(pln)
            op = ps.mean(0)
    else:
        pln = getNod(pln)
        op, nor = ndA(pln.GetOrigin(), pln.GetNormal())
    
    pjs = (ps - op) @ nor
    
    if eqX is not None:
        if eqX == 0:
            return ps[np.argmin(abs(pjs))]
        elif abs(eqX) == 1:
            lb = (pjs * eqX) > 0
            return ps[lb]
        else:
            lb = pjs == eqX
            return ps[lb]
    else:
        ids = pjs > 0
        if isIn:
            return ids
    
    if rtnPjx:
        return pjs, lambda ids=ids: list(ps[ids])
    return ps[ids], ps[~ids]

def ps_pn(pds, pn, typ='min'):
    """在点集中找到距离平面最近或最远的点"""
    pds = nx3ps_(pds)
    op, nor = ndA(pn)
    pjs = (pds-op) @ nor
    
    if typ == 'min':
        id_ = np.argmin(abs(pjs))
        dst = abs(pjs[id_])
    elif typ == 'max':
        id_ = np.argmax(pjs)
        dst = abs(pjs[id_])
    elif typ is None:
        id_ = range(len(pjs))
        dst = np.sort(pjs)
    else:
        raise ValueError("typ参数必须是'min','max'或None。")        
    if 0 <= id_ < len(pds):
        return pds[id_], dst, id_
    else:
        raise IndexError(f"索引{id_}超出给定点集范围。")

def rePln_(pns, refP=None):
    """重新定向法向量"""
    plns = ndA(pns)
    if refP is None: 
        return plns 
    if plns.ndim == 2:
        op, nor = plns
        vec = refP - op
        return op, nor * np.sign(vec @ nor)
    else:
        ops = plns[:,0]
        nors = plns[:,1]
        vecs = refP - ops
        dots = np.einsum('ij,ij->i', vecs, nors)
        return ops, nors * np.sign(dots)[:,None]

def vtkPln(pln, mPd: vtk.vtkPolyData = None, mNam: str = '', 
           refP: PS = None, cPlns: bool = False, **kw):
    """创建VTK平面"""
    if not isinstance(pln, vtk.vtkPlane):
        op, nor = rePln_(pln, refP)
        pln = vtk.vtkPlane()
        pln.SetOrigin(tuple(op))
        pln.SetNormal(tuple(nor))
    if mNam != '' and mPd is None:
        SPln(nor, op, mNam)
    if cPlns:
        cPln = vtk.vtkPlaneCollection()
        cPln.AddItem(pln)
        if mPd is not None:
            return vtkCplnCrop(cPln, mPd, mNam=mNam, **kw)
        return cPln
    if mPd is not None:
        mPd = vtkPlnCrop(mPd, pln, mNam=mNam, **kw)
        return mPd
    return pln

def SPln(nor: PS, cp: PS, mNam: str = "") -> any:
    """创建Slicer平面"""
    pln = SNOD('vtkMRMLMarkupsPlaneNode', mNam)
    pln.SetCenter(cp)
    pln.SetNormal(nor)
    return pln

def vtkPlnCrop(mPd, fun, refP=None, inPd=False, mNam='', **kw):
    """平面裁剪"""
    pd = getPd(mPd)
    if isLs(fun):
        fun = addPlns(fun, refP)
    clp = vtk.vtkClipPolyData()
    clp.SetInputData(pd)
    clp.SetClipFunction(fun)
    clp.GenerateClippedOutputOn()
    clp.Update()
    pd = clp.GetOutput()
    pd = cnnEx(pd, mNam, **kw)
    pd0 = clp.GetClippedOutput()
    if inPd:
        pd0 = cnnEx(pd0, sNam(mNam,'0'), **kw)
        return pd0, pd
    return pd

def vtkCplnCrop(pln, mPd, mNam='', refP=None, **kw):
    """闭合面裁剪"""
    mPd = getPd(mPd)
    if isLs(pln):
        pln = vtkPlns(pln, cPlns=True, refP=refP, **kw)
    clip = vtk.vtkClipClosedSurface()
    clip.SetInputData(mPd)
    clip.SetClippingPlanes(pln)
    clip.Update()
    pd = clip.GetOutput()
    pd = cnnEx(pd, mNam, **kw)
    return pd

def addPlns(funs, refP=None):
    """添加多个平面"""
    funs = ndA(funs)
    clipFun = vtk.vtkImplicitBoolean()
    clipFun.SetOperationTypeToUnion()
    if funs.ndim == 2:
        clipFun.AddFunction(vtkPln(funs, refP=refP))
    elif funs.ndim == 3:
        for fun in funs:
            clipFun.AddFunction(vtkPln(fun, refP=refP))  
    else:
        raise TypeError("Unsupported type for funs: {}".format(type(funs)))
    return clipFun

def vtkPlns(pns: Any, mPd=None, mNam='', pdLs=False, cPlns=False, refP=None, **kw):
    """生成VTK平面集合"""
    pns = ndA(pns)
    if cPlns:
        plns = vtk.vtkPlaneCollection()
        for pn in pns:
            plns.AddItem(vtkPln(pn, refP=refP, cPlns=False))
        if mPd is not None:
            return vtkCplnCrop(plns, mPd, mNam=mNam, **kw)
    else:
        plns = addPlns(pns, refP)
        if mPd is not None:
            return vtkPlnCrop(mPd, plns, mNam=mNam, **kw)        
    return plns

def vtkCut(mPd, pln, mNam='', pad=3, lmd=False, **kw):
    """VTK切割"""
    def vtkCutter__(mPd, pln):
        cutter = vtk.vtkCutter()
        cutter.SetInputData(mPd)
        cutter.SetCutFunction(pln)
        cutter.Update()
        pd = cutter.GetOutput()
        return cnnEx(pd, mNam, **kw)
    
    mPd = getPd(mPd)
    if isinstance(pln, tuple):
        op, nor = pln
        pln = vtkPln((op, nor), mNam=mNam)
    elif isinstance(pln, str):
        sPn = getNod(mPd)
        pln = vtkPln((sPn.GetOrigin(), sPn.GetNormal()))
    else:
        pln = vtkPln((pdCp(mPd), pln))

    pd = vtkCutter__(mPd, pln)
    if lmd:
        return pd, lambda nor=nor, op=op: vtkCutter__(mPd, vtkPln((op, nor)))
    return pd

def cnnEx(mPd, mNam='', *, sp=None, exTyp: Lit['All', 'Lg', None] = None, pdn=False):
    """连通区提取"""
    pd = getPd(mPd)
    if exTyp is None:
        return getNod(pd, mNam)
    if sp is not None:
        return spCnnex(pd, mNam, sp=sp)
    else:
        cnn = vtk.vtkPolyDataConnectivityFilter()
        cnn.SetInputData(pd)
        if exTyp == 'Lg':
            cnn.SetExtractionModeToLargestRegion()
        else:
            cnn.SetExtractionModeToAllRegions()
        cnn.Update()
        pd_out = pdCln(cnn.GetOutput())
        if pdn:
            pd_out = vtkPdNor(pd_out)
        return getNod(pd_out, mNam)

def vtkPdNor(pd):
    """计算PolyData法向量"""
    pd = getPd(pd)
    norFt = vtk.vtkPolyDataNormals()
    norFt.SetInputData(pd)
    norFt.SetFlipNormals(0)
    norFt.AutoOrientNormalsOn()
    norFt.Update()
    return norFt.GetOutput()

def spCnnex(pd, sp, mNam='', pdn=False):
    """根据种子点提取连通区域"""
    pd = getPd(pd)
    sp = ndA(sp)
    nods = []
    nod = None
    cnn = vtk.vtkPolyDataConnectivityFilter()
    cnn.SetInputData(pd)

    def cnnSp__(sp, mNam=mNam):
        cnn.SetExtractionModeToClosestPointRegion()
        cnn.SetClosestPoint(sp)
        cnn.Update()
        pd_out = vtk.vtkPolyData()
        pd_out.DeepCopy(cnn.GetOutput())
        pd_out = pdCln(pd_out)
        if pdn:
            pd_out = vtkPdNor(pd_out)
        return getNod(pd_out, mNam)

    if sp.ndim == 1:
        return cnnSp__(sp)
    elif sp.ndim == 2:
        loc = vtk.vtkStaticPointLocator()
        loc.SetDataSet(pd)
        loc.BuildLocator()
        regC = {}
        for i, p in enumerate(sp):
            idx = loc.FindClosestPoint(p)
            cSp = pd.GetPoint(idx)
            key = tuple(cSp)
            if key not in regC:
                regC[key] = cnnSp__(p, sNam(mNam, str(i)))
            else:
                nod = regC[key]
            nods.append(nod)
        return nods

def pds2Mod(pds, mNam: str = '', psRad=9., refPd=None, **kw):
    """点集转模型"""
    arr = nx3ps_(pds)
    if refPd is not None:
        if isinstance(refPd, dict):
            pdC = pd2Dic(getPd(pds))
            pdC = pdC | refPd
        else:
            pdC = pd2Dic(refPd) 
            pdC['points'] = arr
        return dic2Pd(pdC, mNam)
    
    ps = vtk.vtkPoints()
    pg = vtk.vtkPolygon()
    pgId = pg.GetPointIds()
    cell = vtk.vtkCellArray()
    for i, p in enumerate(arr):
        ps.InsertNextPoint(*p.tolist())
        pgId.InsertNextId(i)
    getNodcell.InsertNextCell(pg)
    mpd = vtk.vtkPolyData()
    mpd.SetPoints(ps)
    mpd.SetPolys(cell)
    mpd = cnnEx(mpd, mNam, **kw)
    if psRad > 0 and mNam != '':
        mpDp = dspNod(mpd, mNam)
        mpDp.SetRepresentation(slicer.vtkMRMLModelDisplayNode.PointsRepresentation)
        mpDp.SetPointSize(psRad)
        mpDp.SetColor(.9, .3, .6)
    return mpd

def ls2dic_(ns, ls, i0_=False): 
    """列表转字典"""
    return dict(zip(list(ns),
                   list(ls[abs(len(ns)-len(ls)):]
                       if i0_ else
                       list(ls[:len(ns)]))))

def getPd(nod, mNam=''):
    """获取vtkPolyData对象"""
    if isinstance(nod, vtk.vtkPolyData):
        return nod
    elif isinstance(nod, str):
        pd = getNod(nod)
        if pd is None:
            raise ValueError(f"找不到模型: {nod}")
        return pd.GetPolyData()
    elif isinstance(nod, np.ndarray):
        return arr2pd(nod, mNam)
    elif isinstance(nod, (list, tuple)):
        return arr2pd(ndA(nod), mNam)
    else:
        try:
            return nod.GetPolyData()
        except:
            raise TypeError(f"无法将类型 {type(nod)} 转换为vtkPolyData")

def arr2pd(arr, mNam='', **kw):
    """将数组转换为vtkPolyData"""
    arr = nx3ps_(arr)
    ps = vtk.vtkPoints()
    pg = vtk.vtkPolygon()
    pgId = pg.GetPointIds()
    cell = vtk.vtkCellArray()
    for i, p in enumerate(arr):
        ps.InsertNextPoint(*p.tolist())
        pgId.InsertNextId(i)
    cell.InsertNextCell(pg)
    mpd = vtk.vtkPolyData()
    mpd.SetPoints(ps)
    mpd.SetPolys(cell)
    mpd = cnnEx(mpd, mNam, **kw)
    return mpd

def clonePd(pd0, mNam=''):
    """克隆PolyData"""
    vpd = vtk.vtkPolyData()
    vpd.DeepCopy(getPd(pd0))
    return getNod(vpd, mNam)

def getNod(data, mNam=''):
    """获取节点"""
    if isinstance(data, str):
        nod = slicer.util.getNode(data)
        if mNam != '':
            nod.SetName(mNam)
    else:
        nod = data
    if mNam != '':
        if isinstance(nod, VPD):
            mod = SNOD('vtkMRMLModelNode', mNam)
            mod.SetAndObservePolyData(data)
            mod.CreateDefaultDisplayNodes()
            return mod
        elif isinstance(nod, slicer.vtkMRMLVolumeNode):
            slicer.util.setSliceViewerLayers(nod, fit=True) 
    return nod

def getArr(nod, dpcopy=True) -> np.ndarray:
    """获取数组"""
    if isinstance(nod, vtk.vtkPolyData):
        pd = nod.GetPoints().GetData()
        arr = vtk_to_numpy(pd)
    elif isinstance(nod, str):
        arr = slicer.util.arrayFromVolume(nod)
    elif isLs(nod):
        arr = ndA(nod)
    else:
        arr = slicer.util.arrayFromVolume(nod.GetID())
    if dpcopy:
        arr = arr.copy()
    return arr

def pdBbx(pd, mNam='', pad=1):
    """获取PolyData包围盒"""
    if isinstance(pd, np.ndarray):
        pd = nx3ps_(pd)
        xyxy = ndA(np.min(pd, axis=0)-pad, np.max(pd, axis=0)+pad)    
        siz = xyxy[1]-xyxy[0]
        vm = np.prod(siz)
    else:
        pd = getPd(pd)
        x0, x1, y0, y1, z0, z1 = pd.GetBounds()
        xyxy = ndA(ndA(x0, y0, z0)+pad, ndA(x1, y1, z1)-pad)
        siz = ndA([x1-x0, y1-y0, z1-z0])
        x, y, z = [max(1, i) for i in siz]
        vm = x*y*z
    rNod = None
    if mNam != '':
        rNod = addRoi(siz, cp=np.mean(xyxy, axis=0), mNam=mNam)
    return xyxy, siz, vm, rNod

def getObt(mNod):
    """获取OBB树"""
    if isinstance(mNod, vtk.vtkOBBTree):
        return mNod
    else:
        pd = getPd(mNod)
        obT = vtk.vtkOBBTree()
        obT.SetDataSet(pd)
        obT.BuildLocator()
        return obT

def obBx(pData, mNam: str = "", nors=None, pad: float = 1, grid: float = 0, **kw):
    """生成定向包围盒"""
    pArr_ = getArr(pData)
    cp = np.mean(pArr_, axis=0)
    pArr = pArr_ - cp
    if nors is not None:
        nors = ndA(nors)
        if nors.ndim == 2:
            drts = nors
        elif nors.shape == (3,):
            dt0, dt1 = p2pXyz(nors, sort=False)
            drts = np.array([dt0, dt1, nors])
    else:
        cov = np.cov(pArr.T)
        cov += EPS*np.eye(3)
        evs, ets = np.linalg.eigh(cov)
        order = evs.argsort()[::-1]
        drts = ets[:, order].T
    
    psR = np.dot(pArr, drts.T)
    mnps = np.min(psR, axis=0)
    mxps = np.max(psR, axis=0)
    dsts = mxps - mnps
    ids = np.where(dsts > 1.)[0]
    dsts = dsts[ids]; drts = drts[ids]
    cn = list(itt.product(*zip(mnps[ids], mxps[ids])))
    cns = np.dot(ndA(cn), drts) + cp
    cns = addPad(cns, pad)
    dsts += pad*2.
    cp = np.mean(cns, axis=0)
    if len(dsts) == 2:
        rNod = vtkGridPln(cns[:-1], stp=grid, mNam=mNam)
    else:
        rNod = addRoi(dsts, drts, cp, f"{mNam}_roi", True)
    return drts, dsts, cns, cp, rNod

def addPad(cns, pad=3):
    """添加填充到包围盒角点"""
    g23 = [1.732, 1.414]
    cns = ndA(cns)
    cN = len(cns)
    cp = np.mean(cns, axis=0)
    cns -= cp
    dsts = psDst(cns, 0)
    pad *= g23[cN == 4]
    cns *= (1 + pad / dsts)[:, None]
    return cp + cns

def obGps(obPd=None, pn=None, siz=(10, 10), stp=1., flat=False, pad=2, mNam=''):
    """生成网格点阵"""
    if obPd is None:
        ns = NZ
        ds = siz
        p0 = OP
    else:
        ns, ds, cns = obBx(obPd, nor=pn, pad=pad, mNam=sNam(mNam, 'box'))[:3]
        p0 = cns[0]
    
    gps = lpn_(np.arange(0, ds[0]+1, stp), p0, ns[0], flat)
    if len(ds) > 1:
        for i in range(1, len(ds)):
            gps = lpn_(np.arange(0, ds[i]+1, stp),
                       gps[(slice(None),)*i + (None,)],
                       ns[i], flat)
    if mNam != '':
        pds2Mod(gps, mNam)
    return gps

def addRoi(dim=[10, 20, 40], drts=None, cp=None, mNam: str = "", mtx: bool = False):
    """创建ROI节点"""
    if drts is None:
        drts = ndA([NX, NY, NZ])
    roi = SNOD("vtkMRMLMarkupsROINode", mNam)
    roi.SetSize(*dim)
    if cp is None:
        cp = 0.5*np.dot(dim, drts)
    b2Rt = np.row_stack(
        (np.column_stack((drts[0], drts[1], drts[2], cp)), (0, 0, 0, 1)))
    b2RtMtx = slicer.util.vtkMatrixFromArray(b2Rt)
    roi.SetAndObserveObjectToNodeMatrix(b2RtMtx)
    dNod = dspNod(roi, mNam)
    dNod.SetColor(0, 0, 1)
    dNod.SetOpacity(0.2)
    if mtx:
        return roi, b2RtMtx
    else:
        return roi

def pdCp(pdata, mNam=''):
    """计算PolyData质心"""
    pd = getPd(pdata)
    cpFt = vtk.vtkCenterOfMass()
    cpFt.SetInputData(pd)
    cpFt.SetUseScalarsAsWeights(False)
    cpFt.Update()
    cp = ndA(cpFt.GetCenter())
    if mNam != '':
        addFid(cp, mNam)
    return cp

def getI2rMat(vol, isArr=True):
    """获取IJK到RAS变换矩阵"""
    vol = getNod(vol)
    mat = vtk.vtkMatrix4x4()
    vol.GetIJKToRASMatrix(mat)
    if isArr:
        return slicer.util.arrayFromVTKMatrix(mat)
    return mat

def getR2iMat(vol, arr=True):
    """获取RAS到IJK变换矩阵"""
    vol = getNod(vol)
    mat = vtk.vtkMatrix4x4()
    vol.GetRASToIJKMatrix(mat)
    if arr:
        return slicer.util.arrayFromVTKMatrix(mat)
    return mat

def ras2vks(ps, reVol=None, lb=1, pvks=True, mNam=''):
    """将RAS坐标系中的点集转换为体素坐标系"""
    if reVol is None:
        reVol = SCEN.GetFirstNodeByClass(LVOL)
    else:
        reVol = getNod(reVol)

    mat = getR2iMat(reVol)
    vArr = getArr(reVol)

    ps = getArr(ps)
    pShp = ps.shape
    if len(pShp) > 2:
        ps = nx3ps_(ps)

    ps1 = np.ones((len(ps), 1))
    ps4 = np.hstack((ps, ps1))

    ijk = (ps4 @ mat.T)[:, :3]
    ijk = ijk.astype(int)

    ijk = np.clip(ijk, a_min=0, a_max=ndA(vArr.shape)[::-1] - 1)

    z = ijk[:, 0]
    y = ijk[:, 1]
    x = ijk[:, 2]

    if pvks:
        varr = vArr.copy()
        varr = vArr[x, y, z]
        if lb != 0:
            varr = np.where(varr != 0, lb, 0)
        if len(pShp) > 2:
            varr = varr.reshape(pShp[:-1])
        return varr, ijk

    mArr = np.zeros_like(vArr)
    mArr[x, y, z] = lb
    if lb == 0:
        mArr[x, y, z] = 1
        mArr *= vArr
    if mNam != '':
        vol = volClone(reVol, mNam)
        slicer.util.updateVolumeFromArray(vol, mArr)
    return mArr

def cropVol(vol, roi=None, mNam='', cArr=None, delV=True):
    """裁剪体素"""
    vNod = getNod(vol)
    if roi is None:
        rNod = pdBbx(lVol2mpd(vNod, exTyp="All"), mNam)[-1]
    else:
        rNod = getNod(roi)
    cropLg = slicer.modules.cropvolume.logic()
    cropMd = slicer.vtkMRMLCropVolumeParametersNode()
    cropMd.SetROINodeID(rNod.GetID())
    cropMd.SetInputVolumeNodeID(vNod.GetID())
    cropMd.SetVoxelBased(True)
    cropLg.FitROIToInputVolume(cropMd)
    cropLg.Apply(cropMd)
    cropVol = SCEN.GetNodeByID(cropMd.GetOutputVolumeNodeID())
    if mNam != '':
        cropVol.SetName(mNam)
    if cArr is not None:
        slicer.util.updateVolumeFromArray(cropVol, cArr[:, ::-1])
    if delV:
        SCEN.RemoveNode(vNod)
    if roi is None:
        return rNod, volData(cropVol, mNam, exTyp="All")
    return volData(cropVol, mNam, exTyp="All")

volClone = lambda vol, nam='': slicer.modules.volumes.logic().CloneVolumeGeneric(SCEN, vol, nam)

def arr2vol(vol: Union[VOL, str]=None, arr=0, mNam='', rtnVd=False, pad=1) -> VOL:
    """数组转体素"""
    if vol is None:
        vol = SCEN.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
    else:    
        vol = getNod(vol)
    cVol = volClone(vol, mNam)
    vArr = getArr(cVol)
    if not isinstance(arr, np.ndarray):
        arr = np.ones_like(vArr) * arr
        arr = np.pad(arr, pad, mode='constant')
    else:
        arr = np.pad(arr, pad, mode='constant')
    slicer.util.updateVolumeFromArray(cVol, arr.astype(vArr.dtype))
    return volData(cVol) if rtnVd else cVol

def lVol2mpd(lVol, mNam='', **kw):
    """标签体素转模型"""
    vol = getNod(lVol)
    assert isinstance(vol, slicer.vtkMRMLLabelMapVolumeNode), f'{type(vol)=}必须是标签体素'
    seg = SNOD('vtkMRMLSegmentationNode')
    segLg = slicer.modules.segmentations.logic()
    segLg.ImportLabelmapToSegmentationNode(vol, seg)
    segs = seg.GetSegmentation()
    segn = segs.GetNumberOfSegments()
    getId = segs.GetNthSegmentID

    def getSeg_(ii=0, mNam=mNam, **kw):
        segId = getId(ii)
        pd = vtk.vtkPolyData()
        segLg.GetSegmentClosedSurfaceRepresentation(seg, segId, pd, 1)
        return cnnEx(pd, mNam=mNam, pdn=True, **kw)
    
    if segn == 1:
        pdc = getSeg_(0, mNam, **kw)
    else:
        pdc = {}
        for i in range(segn):
            id_ = getId(i)
            lb = segs.GetSegment(id_).GetLabelValue()
            lv = TLDIC[lb]
            mNam_ = mNam+str(lv) if mNam != '' else ''
            pdc[lv] = getSeg_(i, mNam_, **kw)
    slicer.mrmlScene.RemoveNode(seg)
    return pdc

def readIsoCT(ctF, mNam='', isLb=True, cstU8=True):
    """读取CT并初始化"""
    spc = (1, 1, 1)
    if os.path.exists(ctF):
        img = sitk.ReadImage(ctF)
    else:
        img = puSk(ctF)
    if cstU8 == True:
        img = sitk.Cast(img, sitk.sitkUInt8)
    spcOr = img.GetSpacing()
    reSpl = sitk.ResampleImageFilter()
    if spcOr != spc:
        sizOr = img.GetSize()
        siz = [int(round(osz * osp / sp)) for osz, osp, sp in zip(sizOr, spcOr, spc)]
        itpltor = [sitk.sitkLinear, sitk.sitkNearestNeighbor][isLb]
        reSpl.SetSize(siz)
        reSpl.SetOutputSpacing(spc)
        reSpl.SetInterpolator(itpltor)
        reSpl.SetOutputDirection(img.GetDirection())
        reSpl.SetOutputOrigin(img.GetOrigin())
        reSpl.SetDefaultPixelValue(0)
        reSpl.SetTransform(sitk.Transform())
        img = reSpl.Execute(img)
        img = sitk.DICOMOrient(img, 'RAS')
    vol = skPu(img, None, mNam, SVOL if isLb == False else LVOL)
    vol.SetOrigin(OP)
    vol.SetSpacing(spc)
    return volData(vol, mNam)

def volData_(vol, mNam='', **kw):
    """体素数据处理类"""
    class imInfo:
        def __init__(self, vol):
            self.vol = getNod(vol, mNam)
            self.nam = self.vol.GetName()

        @property
        def update(self):
            if not hasattr(self, '_update'):
                self._update = slicer.util.updateVolumeFromArray(self.vol, self.arr)
            return self._update

        @property
        def arr(self):
            if not hasattr(self, '_arr'):
                self._arr = getArr(self.nam)
            return self._arr

        @property
        def imData(self):
            if not hasattr(self, '_imData'):
                self._imData = self.vol.GetImageData()
            return self._imData

        @property
        def op(self):
            if not hasattr(self, '_op'):
                self._op = self.vol.GetOrigin()
            return self._op
        
        @property
        def spc(self):
            if not hasattr(self, '_spc'):
                self._spc = self.vol.GetSpacing()
            return self._spc

        @property
        def mat(self):
            if not hasattr(self, '_mat'):
                self._mat = getI2rMat(self.vol)
            return self._mat

        @property
        def ps(self):
            if not hasattr(self, '_ps'):
                self._ps = vks2Ras(self.vol, lbs=True)
            return self._ps

        @property
        def mod(self):
            if not hasattr(self, '_mod'):
                self._mod = lVol2mpd(self.vol, mNam, **kw)
            return self._mod

        @property
        def pd(self):
            if not hasattr(self, '_pd'):
                self._pd = lVol2mpd(self.vol, **kw)
            return self._pd

        @property
        def lbs(self):
            if not hasattr(self, '_lbs'):
                self._lbs = np.unique(self.arr)
                self._lbs = self._lbs[self._lbs != 0].astype(np.int8)
            return self._lbs
    
    imProp = imInfo(vol)
    assert np.max(imProp.arr) != 0, f'the volume data of {imProp.vol.GetName()} is empty'
    return imProp

def vks2Ras(vmData, vks=None, lbs=False):
    """体素坐标转RAS坐标"""
    def vks2Ps__(vks, vMat):
        vks = ndA(np.where(vks != 0)).T[:, ::-1]
        arr1 = np.ones((*vks.shape[:-1], 1))
        ijk = np.hstack((vks, arr1))
        ps = (ijk @ vMat.T)[:, :3]
        return ps
    
    if not isinstance(vmData, np.ndarray):
        vMat = getI2rMat(vmData)
        if vks is None:
            vks = getArr(vmData)
    else:
        vMat = vmData
    if lbs is False:
        return vks2Ps__(vks, vMat)
    else:
        lbs = np.unique(vks)
        lbs = lbs[lbs != 0]
        lbRas = {}
        for lb in lbs:
            lv = TLDIC[lb]
            lbRas[lv] = vks2Ps__(vks == lb, vMat)
    return lbRas

def addFid(p=OP, mNam='', dia=3):
    """添加标记点"""
    if mNam != '':
        fid = SNOD("vtkMRMLMarkupsFiducialNode")
        fid.AddControlPoint(p)
        fid.SetName(mNam)
        dpNod = dspNod(fid, mNam)
        dpNod.UseGlyphScaleOn()
        if dia > 0:
            dpNod.UseGlyphScaleOff()
            dpNod.SetGlyphSize(dia)
        return fid

def ps2cFids(ps, mNam='', lbNam=None, closed=False, lDia=0):
    """点集转换为曲线"""
    if isinstance(ps, dict):
        ps = list(ps.values())
    else:
        ps = nx3ps_(ps)
    cls = ["vtkMRMLMarkupsCurveNode", "vtkMRMLMarkupsClosedCurveNode"][closed*1]
    cuvNod = slicer.mrmlScene.AddNewNodeByClass(cls)
    slicer.util.updateMarkupsControlPointsFromArray(cuvNod, ndA(ps))
    cuvNod.SetName(mNam)

    dpNod = dspNod(cuvNod, mNam)
    dpNod.SetLineThickness(lDia)
    dpNod.UseGlyphScaleOn()
    if lDia == 0:
        dpNod.UseGlyphScaleOff()
        dpNod.SetGlyphSize(1.)
    if lbNam is not None:
        tuple(cuvNod.SetNthControlPointLabel(i, label)
              for i, label in enumerate(lbNam))
        dpNod.SetPointLabelsVisibility(True)
        dpNod.SetSliceProjection(True)
    return cuvNod

def pdPj(pds, pn=None, mNam='', **kw):
    """点投影平面"""
    pd = getPd(pds)
    if isinstance(pn, tuple):
        pp, nor = ndA(pn)
    else:
        nor = ndA(pn)
        pp = pdCp(pd)    
    nor = ndA(nor)
    pjPla = vtk.vtkProjectPointsToPlane()
    pjPla.SetInputData(pd)
    pjPla.SetProjectionTypeToSpecifiedPlane()
    pjPla.SetOrigin(*pp)
    pjPla.SetNormal(nor)
    pjPla.Update()
    pjPd = pjPla.GetOutput()
    return cnnEx(pjPd, mNam=mNam, **kw)

def psPj(pds, pn, mNam='', flat=True, **kw):
    """点集投影到平面"""
    ps = getArr(pds)[:]
    pn = ndA(pn)
    if len(pn) == 2:
        pp, nor = pn
    else:
        pp = np.mean(ps, axis=0)
        nor = pn
    n = uNor(nor)
    pjPs = ps - np.outer((ps-pp)@n, n)
    if mNam != '':
        pjPd = pds2Mod(pjPs, mNam=mNam, **kw)
        pjPs = getArr(pjPd)
    return pjPs if flat else pjPs.reshape(ps.shape)

def oriMat(norX, drts=False):
    """计算方向矩阵"""
    norZ = np.cross(ndA([1., 1., 1.]), norX, axis=0)
    norZ = uNor(norZ)
    norY = np.cross(norX, norZ, axis=0)
    norY = uNor(norY)
    if drts:
        return norX, norY, norZ
    else:
        mat = vtk.vtkMatrix4x4()
        mat.Identity()
        for i in range(0, 3):
            mat.SetElement(i, 0, norX[i])
            mat.SetElement(i, 1, norY[i])
            mat.SetElement(i, 2, norZ[i]*-1)
        return mat

def kdOlbs_(ps, r=.4, qs=None, rtnLen=True):
    """KD树球查询"""
    ps = getArr(ps)
    qs = ps if qs is None else getArr(qs)
    psT = kdT(ps)
    lbs = psT.query_ball_point(qs, r, return_length=rtnLen)
    return lbs, psT, lambda qs=qs, r=r, ps=ps, ln=rtnLen: kdT(ps).query_ball_point(qs, r, return_length=ln)

def mxNorPs_(pjPs, nor=[1, 0, 0], cp=None, mnP=False, rjPs=None, tp=None):
    """在某个方向找极值点"""
    ps = getArr(pjPs)
    if cp is None:
        cp = ps.mean(0)
    if tp is not None:
        nor *= np.sign(np.dot(tp-cp, nor))
    pjs = np.dot(ps-cp, ndA(nor))
    if mnP:        
        return ps[np.argmin(abs(pjs))]
    else:
        iMn = np.argmin(pjs)
    iMx = np.argmax(pjs.ravel())
    nxPs = ndA(ps[iMn], ps[iMx])
    if rjPs is not None:
        return nxPs, ndA([rjPs[iMn], rjPs[iMx]])
    return nxPs

def rayCast(p1, p2=None, mPd='', nor=None, plus=0, oneP: bool = True, inOut=False):
    """射线投射"""
    pt = p2pLn(p1, p2, nor=nor, plus=plus)[0]
    obT = getObt(mPd)
    scPs = vtk.vtkPoints()
    code = obT.IntersectWithLine(p1, pt, scPs, None)
    if inOut:
        return code
    if code == 0:
        print(f"No intersection points found")
        return ndA(None)
    else:
        sPs = scPs.GetData()
        num = sPs.GetNumberOfTuples()
        secPs = []
        for i in range(num):
            secPs.append(sPs.GetTuple3(i))
        if oneP == True:
            return ndA(secPs)[0]
        else:
            return ndA(secPs)

def log_(text="Done", tim=None, sec=.5):
    """记录日志"""
    if tim is None:
        print(text, end='>')
        return
    tim_ = time.time()
    timx = tim_-tim[0]
    tTxt = f"{int(timx//60)}\'. "
    timy = tim_-tim[1]
    tTxt += f"费时:{int(timy//60)}\'{(timy%60):.1f}\""
    txt = f'{text} T: {tTxt}'
    time.sleep(sec)
    slicer.app.processEvents()
    zoom()
    print(txt)
    return tim_

def zoom():
    """缩放视图"""
    slNods = slicer.util.getNodes('vtkMRMLSliceNode*')
    sl3d = slicer.app.layoutManager().threeDWidget(0).threeDView()
    for slNod in list(slNods.values()):
        slWgt = slicer.app.layoutManager().sliceWidget(slNod.GetLayoutName())
        slWgt.sliceLogic().FitSliceToAll()
        sl3d.resetFocalPoint()
        sl3d.resetCamera()

def c2s_(c, arr=False): 
    """字典值转列表或数组"""
    return ndA(list(c.values())) if arr else list(c.values())

def dic2Pd(dic, mNam=''):
    """字典转换为VTK PolyData"""
    pd = vtk.vtkPolyData()

    if 'points' in dic:
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(dic['points']))
        pd.SetPoints(pts)

    if 'cells' in dic:
        cells = dic['cells']
        if cells.ndim == 1:
            cells = cells[:, None]

        ca = vtk.vtkCellArray()
        for cell in cells:
            ca.InsertNextCell(len(cell))
            for pid in cell:
                ca.InsertCellPoint(int(pid))

        if cells.shape[1] == 1:
            pd.SetVerts(ca)
        elif cells.shape[1] == 2:
            pd.SetLines(ca)
        elif cells.shape[1] == 3:
            pd.SetPolys(ca)
        else:
            pd.SetPolys(ca)

    if 'cell_data' in dic:
        for k, v in dic['cell_data'].items():
            arr = numpy_to_vtk(v)
            arr.SetName(k)
            pd.GetCellData().AddArray(arr)

    if 'point_data' in dic:
        for k, v in dic['point_data'].items():
            arr = numpy_to_vtk(v)
            arr.SetName(k)
            pd.GetPointData().AddArray(arr)

    return getNod(pd, mNam)

def pd2Dic(pds):
    """将VTK PolyData转换为字典格式"""
    pd = getPd(pds)

    points = vtk_to_numpy(pd.GetPoints().GetData())
    cells = np.array([[pd.GetCell(i).GetPointId(j)
                      for j in range(pd.GetCell(i).GetNumberOfPoints())]
                     for i in range(pd.GetNumberOfCells())])

    def get_data(data_obj):
        return {data_obj.GetArray(i).GetName():
                vtk_to_numpy(data_obj.GetArray(i))
                for i in range(data_obj.GetNumberOfArrays())}

    cell_data = get_data(pd.GetCellData())
    point_data = get_data(pd.GetPointData())

    return {
        'points': points,
        'cells': cells,
        'cell_data': cell_data,
        'point_data': point_data
    }

def lsDic(dic, lDic, upDate=False) -> dict:
    """列表字典操作"""
    if not lDic:
        lDic = dfDic(list)
    if upDate:
        for k, v in dic.items():
            if k in lDic.keys():
                lDic[k] = []
    for k, v in dic.items():
        data = [(k, v)]
        for (key, value) in data:
            lDic[key].append(value)
    return lDic

def pdAndPds(pds, mNam=''):
    """合并多个PolyData"""
    assert len(pds) > 1, 'pds 至少两个'
    pd0, pd1 = getPd(pds[-2]), getPd(pds[-1])
    vAdd = vtk.vtkAppendPolyData()
    vAdd.AddInputData(pd0)
    vAdd.AddInputData(pd1)
    vAdd.Update()
    scPd = vAdd.GetOutput()
    if len(pds) > 2:
        for pd in pds[:-2]:
            scPd = pdAndPds((scPd, getPd(pd)))
    return getNod(scPd, mNam)

def vtkGridPln(ps=OP, nor=NZ, stp=1.0, siz=(10., 10.), push=0.0, mNam=''):
    """创建网格平面"""
    ps = ndA(ps)
    if len(ps) == 3:
        op, xp, yp = ps
        n = uNor(np.cross(xp-op, yp-op))
        if stp!=0:
            siz = norm(ps[1:] - op, axis=1)
    else:
        siz = ndA(siz)
        n = uNor(nor)
        u, v = p2pXyz(n, False)
        op = ps[:]
        xp, yp = op + .5*siz*ndA(u, v)
    
    pln = vtk.vtkPlaneSource()
    pln.SetOrigin(op)
    pln.SetPoint1(xp)
    pln.SetPoint2(yp)
    if stp!=0:
        gSiz = (siz/stp).astype(int)
        pln.SetXResolution(gSiz[0])
        pln.SetYResolution(gSiz[1])
    pln.Update()
    grid = pln.GetOutput()
    if stp!=0: 
        gArr = getArr(grid).reshape(gSiz[0]+1, gSiz[1]+1, 3)
    else:
        gArr = getArr(grid)
    if push!=0: 
        grid = vtkPush(grid, n, push)
    if mNam != '':
        _=getNod(grid, mNam=mNam)
    return gArr, grid

def vtkPush(pd, nor=NZ, dst=1., mNam=''):
    """VTK挤出"""
    pd = getPd(pd)
    push = vtk.vtkLinearExtrusionFilter()
    push.SetInputData(pd)
    push.SetVector(*nor)
    push.SetScaleFactor(dst)
    push.Update()
    return getNod(push.GetOutput(), mNam)

def p3Angle(p0, p1, p2, rtn='deg'):
    """三点求角度（顶点在p1）"""
    rst = {}
    v1, v2 = ndA(p1, p2) - ndA(p0)    
    cos_ = np.dot(v1, v2) / (norm(v1) * norm(v2))
    rst['cos'] = cos_
    rst['acr'] = np.arccos(np.clip(cos_, -1.0, 1.0))
    rst['deg'] = np.degrees(rst['acr'])
    return lambda x=rtn: rst[x]

def rdrgsRot(v, k, a):
    """罗德里格斯旋转公式"""
    cos_ = np.cos(a)
    sin_ = np.sin(a)
    return v*cos_ + np.cross(k,v)*sin_ + k*k@v*(1-cos_)

def thrGrid(p0, p1, rad=10., tLg=None, sn=12, mNam=''):
    """在圆柱体表面生成螺旋网格点"""
    p0, p1 = ndA(p0, p1)
    nor, lg, px = p2pLn(p0, p1)[1:]
    cArr, cir_ = vCir30(nor, rad)
    if tLg is None:
        tLg = rad*.3
    stmLg = lg-rad
    thrPs = px(rgN_(tLg, tLg/sn), cArr)
    stmPs = list(px(rgN_(stmLg, tLg)[:, None], thrPs)+p0)
    stmPs += [list(cArr+p0),]
    tiPs = [cir_(rad*.7)+px(stmLg+rad*.3),]
    tiPs += [cir_(rad*.4)+px(stmLg+rad*.6),]
    tiPs += [cir_(rad*.1)+p1]
    thrPs = ndA(stmPs+tiPs)
    if mNam:
        ps2cFids(nx3ps_(thrPs), mNam)
    return thrPs

SIN = np.array([0.0, 0.5, 0.866, 1.0, 0.866, 0.5, 
                -0.0, -0.5, -0.866, -1.0, -0.866, -0.5])
COS = np.array([1.0, 0.866, 0.5, 0.0, -0.5, -0.866,
                -1.0, -0.866, -0.5, -0.0, 0.5, 0.866])

def vCir30(nor=NZ, rad=1., cp=OP, mNam=''):
    """生成30度间隔圆"""
    cp, nor = ndA(cp, nor)
    v, w = p2pXyz(nor, False)
    bCir_ = np.outer(COS, v) + np.outer(SIN, w)
    cir = bCir_ * rad + cp
    if mNam!='':
        ps2cFids(cir, mNam, None, 1, 1.)
    return cir, lambda r, p=cp: bCir_*r+p

def psMic(pds, inGps=None, nor=None, stp=1.0, mNam='', mxIt=20):
    """计算点集的最大内切圆"""
    ps = getArr(pds)
    psT = kdT(ps)
    
    if nor is None:
        nor = psFitPla(ps)
    else:
        nor = ndA(nor)
        if nor.shape == (2,3):
            nor, drt = nor
            
    if inGps is None:
        gCp = ps.mean(0)
        mnDt = findPs(ps, gCp)[1]
        gps = obGps(ps, nor, flat=True)
        inGps = gps[kdOlbs_(gps, mnDt, gCp, False)[0]]
    else:
        inGps = getArr(inGps)
        if pds is None:
            gCp = inGps.mean(0)
            mnDt = findPs(ps, gCp)[1]

    cir_ = vCir30(nor)[-1]

    def mnMx(gps):
        """找到最优点和实际半径"""
        dts = psT.query(gps, k=1)[0]
        mxId = np.argmax(dts)
        cp = gps[mxId]
        rad = psT.query(cp[None], k=1)[0][0]
        return cp, rad

    cp, rad = mnMx(inGps)
    best_cp, best_rad = cp, rad
    i = 0

    while (stp >= EPS and i < mxIt):
        ps_ = cir_(stp, cp)
        cp_, rad_ = mnMx(ps_)

        rOpt = (rad_ - rad) / rad
        dOpt = norm(cp-cp_) / (stp * 2.0)

        if (dOpt <= 1.0 and rOpt > -0.05 and rOpt/dOpt > -0.1):
            cp, rad = cp_, rad_
            if rad > best_rad:
                best_cp, best_rad = cp, rad

        stp *= 0.7
        i += 1

    cp, rad = best_cp, best_rad

    actual_rad = psT.query(cp[None], k=1)[0][0]
    if abs(actual_rad - rad) > EPS:
        print(f"警告: 实际半径({actual_rad:.4f})与计算半径({rad:.4f})不匹配")
        rad = actual_rad
    arr = vCir30(nor, rad, cp, mNam)[0]
    return cp, rad, arr

def lnXpln(pn, p0, p1=None):
    """计算平面和直线的交点"""
    p, n = ndA(pn)
    if p1 is not None:
        p0, p1 = ndA(p0, p1)
        v = uNor(p1 - p0)
    else:
        p0, v = ndA(p0)
    d = np.dot(v, n)
    if abs(d) <= 1e-8:
        raise ValueError('plnXln: The line is parallel to the plane.')
    def dt(p=p): return np.dot(p - p0, n) / d
    return p0 + dt() * v, dt

def erod_(msk, gps, its=3, sp=0, r=1/3):
    """分离弱连接区域并保留主体"""
    msk = ndA(msk)
    s = np.array([[0, r, 0], [r, 1, r], [0, r, 0]])
    edMsk = erod(msk, s, its)
    lbs, num = scLb(edMsk)
    
    if sp is not None:
        if np.all(sp == 0):
            msk_ = (lbs == delBdMsk_(lbs))
        else:
            try:
                spId = findPs(gps, sp)[-1]
                spLb = lbs[spId]
                msk_ = (lbs == spLb)
            except:
                print("警告: 种子点处理失败，使用最大连通区域")
                msk_ = (lbs == delBdMsk_(lbs))
    else:
        msk_ = (lbs == delBdMsk_(lbs))
    
    ctPs = gps[msk_^dila_(msk_, 1)]
    ctPs = psLbs(ctPs)
    return gps[msk_], ctPs

def delBdMsk_(lbs_, mxCnt=10):
    """判断区域是否接触边界并按点数筛选"""
    if not isinstance(lbs_, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if lbs_.ndim != 2 or lbs_.size == 0:
        return np.array([], dtype=int)

    bdMsk = np.zeros_like(lbs_, dtype=bool)
    bdMsk[[0, -1], :] = bdMsk[:, [0, -1]] = True
    
    msk = ~np.isin(lbs_, np.unique(lbs_[bdMsk]))
    
    cnts = np.bincount(lbs_.ravel() * msk.ravel())
    mxLb = np.array([np.argmax(cnts[1:])+1])
    
    return mxLb.astype(int)

def dila_(msk, delCt=3, knlTyp='enhanced', r=.3, lb=None):
    """增强型膨胀操作"""
    if knlTyp == 'full':
        s = np.ones((3,3)) * r
    elif knlTyp == 'cross':
        s = np.array([[0, r, 0], [r, r, r], [0, r, 0]])
    elif knlTyp == 'enhanced':   
        s = np.array([[0, r, r, r, 0], [r, r, r, r, r], [r, r, r, r, r],
                      [r, r, r, r, r], [0, r, r, r, 0]])
    
    for _ in range(delCt):
        msk = dila(msk, s)
        msk |= dila(msk, np.array([[0, r, 0], [r, r, r], [0, r, 0]]))
    
    return msk

def psLbs(ps, num=1, rad=1.0, mnSps=5, ax=2, mNam=''):
    """聚类点云分群函数"""
    from sklearn.cluster import DBSCAN
    ps = getArr(ps)
    if rad is None:
        from sklearn.neighbors import NearestNeighbors
        def kDst_():
            nbs = NearestNeighbors(n_neighbors=mnSps)
            nbs.fit(ps)
            dsts, _ = nbs.kneighbors(ps)
            return dsts[:, -1]
        rad = np.percentile(kDst_(), 95)
        print(f"自动计算邻域半径: {rad:.2f} mm")
    
    clt = DBSCAN(rad, min_samples=mnSps).fit(ps)
    cLbs_ = clt.labels_
    lbs_ = np.unique(cLbs_[cLbs_ >= 0])
    num_ = len(lbs_)
    
    if num is not None:
        if num_ < num:
            print(f"警告：可能过分割，建议减小rad（当前{rad}）或增大mnSps（当前{mnSps}）")
        elif np.sum(clt.labels_ == -1) > len(ps)*0.3:
            print(f"警告：噪声点超过30%，建议增大rad（当前{rad}）或减小mnSps（当前{mnSps}）")
        
        if num == 1:
            cnts = np.bincount(cLbs_[cLbs_ >= 0])
            mxLb = np.argmax(cnts)
            cPs = ps[cLbs_ == mxLb]
            if mNam != '': 
                pds2Mod(cPs, mNam=mNam)
            return cPs
            
        if num_ > num:
            sizes = [(lb, np.sum(cLbs_ == lb)) for lb in lbs_]
            stLbs = sorted(sizes, key=lambda x: x[1])[-num:]
            lbs_ = np.array([lb for lb, _ in stLbs])
            
        clts = []
        meds = []
        
        for lb in lbs_:
            clt = ps[cLbs_ == lb]
            clts.append(clt)
            meds.append(np.median(clt[:, ax]))
        
        stClts = [clts[i] for i in np.argsort(meds)]
    else:
        stClts = [ps[cLbs_ == lb] for lb in lbs_]
        
    if mNam:
        for i, cPs in enumerate(stClts):
            pds2Mod(cPs, mNam=f"{mNam}_{i}")
    
    return stClts

class CtPj:
    """计算投影轮廓和内部网格点集的类"""
    def __init__(self, pds, pjNor=None, cJp=None, sp=None, clean=False, eSp=None,
                mNam='', rad=1/3, thr=None):
        self.pds = getArr(pds)
        self.pd = getPd(pds)
        if self.pds is None or len(self.pds) == 0:
            raise ValueError("输入点集为空")
            
        self.cJp = ndA(cJp if cJp is not None else self.pds.mean(0))
        self.sp = sp
        self.mNam = mNam
        self.r = rad
        self.thr = thr if thr is not None else self.r * 0.1
        self.eSp = eSp
        
        if pjNor is not None:
            pjNor = ndA(pjNor)
            if pjNor.ndim==2: 
                _, self.pjNor, self.sDrt = self.ras = pjNor
            else: 
                self.pjNor = pjNor; self.ras = None
            self.pjPs_ = psPj(self.pds, (self.cJp, self.pjNor), 
                            mNam=sNam(mNam, 'pjPs_'), exTyp=None)
        else:
            self.pjPs_ = self.pds
            self.pjNor = psFitPla(self.pjPs_)
        
        if pjNor.ndim==2:
            self.sDrt = pjNor[2]
        
        if clean:
            self.pjPs = cleanPj(cnnEx(self.pjPs_), mNam=sNam(mNam, 'pjPs'))
        else:
            self.pjPs = getArr(cnnEx(self.pjPs_, exTyp='Lg'))
        
        self.gps = obGps(self.pjPs, 
                    self.ras if self.ras is not None else self.pjNor, 
                    stp=1/3, mNam=sNam(mNam, 'gps'))    
        _, pjT, kdO_ = kdOlbs_(self.pjPs, 1.)
        inds = kdO_(self.gps)
        gMsk = np.where(inds > 0, 0, 1)
        
        try:            
            bMsk = np.zeros_like(gMsk, dtype=bool)
            bMsk[[0, -1], :] = gMsk[[0, -1], :] == 1
            bMsk[:, [0, -1]] = gMsk[:, [0, -1]] == 1
            stt = np.ones((3, 3), dtype=bool)
            bds = dila(bMsk, stt, -1, gMsk==1)
            self.msk = gMsk.copy()
            self.msk[bds] = 0
            if self.sp is not None:
                spId = findPs(self.gps, self.sp)[-1]
                lbs, num = scLb(self.msk)
                if num > 0:
                    sLb = lbs[spId]
                    self.msk = lbs==sLb
            
            self.inGps = self.gps[self.msk>0]
            pds2Mod(self.inGps, sNam(mNam, 'inGps'))
            ctId = pjT.query(self.inGps)[1]
            self.ctPs = self.pjPs[ctId]
            pds2Mod(self.ctPs, sNam(mNam, 'ctPs'))
        except Exception as e:
            print(f"CtPj初始化失败: {str(e)}")
            self.inGps = np.array([])
            self.ctPs = np.array([])
            self.pjPs_ = self.pjPs = np.array([])
            raise        
    
    def mic(self):
        """计算最大内切圆"""
        if not hasattr(self, 'inGps') or len(self.inGps) == 0:
            self.inGps = np.array([])
            
        try:
            pds2Mod(self.ctPs, mNam=sNam(self.mNam, 'ctPs'))
            pds2Mod(self.inGps, mNam=sNam(self.mNam, 'iPs_'))
            cp, rad, mic = psMic(self.ctPs, self.inGps, self.pjNor,
                                mNam=sNam(self.mNam, 'ctMic'))
            return cp, rad, mic, self.ctPs
        except Exception as e:
            print(f"计算最大内切圆失败: {str(e)}")
            raise

    def edPs(self):
        """计算椭圆弧"""
        try:
            inGps, self.bdPs = ctBd_ed(self.msk, self.gps, 9)
            if len(inGps) == 0:
                raise ValueError("未找到有效内部点")
                
            pds2Mod(inGps, mNam=sNam(self.mNam, 'iPs'))
            
            rjPs_ = self.pds[kdT(self.pjPs_).query(inGps)[1]]
            pds2Mod(rjPs_, mNam=sNam(self.mNam, 'rjPs_'))
            
            eIps, eSps = psLbs(rjPs_, 2, 2)
            if len(eSps) == 0 or len(eIps) == 0:
                raise ValueError("分终板计算失败")
            
            cp_ = findPs(eSps, self.cJp)[0]
            cp_ = cp_-self.pjNor*2
            eSps = dotPlnX(eSps, (cp_, self.pjNor), 1.)
            pds2Mod(eSps, mNam=sNam(self.mNam, 'eSps'))
            
            sDrt = psFitPla(eSps)
            if not any(sDrt):
                raise ValueError("椎体方向计算失败")
                
            vbIps = psPj(inGps, (self.cJp, sDrt))
            self.iPs = psPj(self.ctPs, (self.cJp, sDrt))
            
            pds2Mod(vbIps, mNam=sNam(self.mNam, 'vbIps'))
            pds2Mod(self.iPs, mNam=sNam(self.mNam, 'Ips'))
            
            return eSps, eIps, sDrt, vbIps, self.iPs
            
        except Exception as e:
            print(f"椭圆弧计算失败: {str(e)}")
            return np.array([]), np.array([]), np.array([0,0,1]), np.array([]), np.array([])

    def sCt(self):
        """计算椎体截面"""
        try:
            pds2Mod(self.ctPs, mNam=sNam(self.mNam, 'ctPs'))
            
            rjPs = self.pds[kdT(self.pjPs_).query(self.ctPs)[1]]
            
            pjs = (rjPs-self.cJp) @ self.pjNor
            rjPs = rjPs[pjs>0]
            ctPs = self.ctPs[pjs>0]
            pds2Mod(rjPs, mNam=sNam(self.mNam, 'rjPs'))
            return self.inGps, ctPs, rjPs
        except Exception as e:
            print(f"计算椎体截面失败: {str(e)}")
            return np.array([]), np.array([]), np.array([])

def ctBd_ed(lbs, gps, delCt=3):
    """计算轮廓边界点并删除指定层数"""
    msk = (lbs == 1).copy()
    
    bdy = np.zeros_like(msk, dtype=bool)
    bdy[1:] |= (msk[1:] ^ msk[:-1])
    bdy[:-1] |= (msk[:-1] ^ msk[1:])
    bdy[:, 1:] |= (msk[:, 1:] ^ msk[:, :-1])
    bdy[:, :-1] |= (msk[:, :-1] ^ msk[:, 1:])
    
    bdy &= (lbs == 0)
    ids = np.argwhere(bdy)
    
    xC, yC = {}, {}
    for x, y in ids:
        xC.setdefault(x, []).append(y)
        yC.setdefault(y, []).append(x)
    
    for y, xs in yC.items():
        msk[:, y] = 0
        msk[min(xs):max(xs)+1, y] = 1
    
    for x, ys in xC.items():
        msk[x, :] = 0
        msk[x, min(ys):max(ys)+1] = 1

    if delCt > 0:
        return erod_(msk, gps, delCt)
    
    iPs = gps[msk]
    ctPs = gps[msk^dila_(msk, 1)]
    
    return iPs, ctPs

def cleanPj(pjPd, pn=None, r=.3, thr=2, mNam='vtCt'):
    """清理投影点"""
    if pn is not None:
        pjPd = getNod(pjPd)
        ps = psPj(pjPd, pn)
    ps = getArr(pjPd)
    ctPs = kdCt_(ps)
    ctPs = psLbs(ctPs, mNam=mNam)
    return ctPs

def kdCt_(ctPs, r=.3, thr=2, cp=None, rad=1., mNam=''):
    """去除平面点集轮廓内的孤立岛点集"""
    ctPs = getArr(ctPs)
    if cp is None:
        cp = ctPs.mean(0)
    lbs_, _, kdx_ = kdOlbs_(ctPs, r)
    ps_ = ctPs[lbs_>thr]
    lens = kdx_(cp, rad)    
    if lens>0:
        lbs=kdx_(ps_, r=rad)
        ps = ps_[lbs>lens]
    else:
        ps = ps_
    if mNam != '':
        pds2Mod(ps, mNam=mNam)
    return ps

def readVtk(filePath):
    """读取VTK文件"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filePath)
    reader.Update()
    polyData = reader.GetOutput()
    return polyData

def pdTf(mPd, p0=OP, go=OP, nor=None, oMat=None, goX=0., goY=0., goZ=0.,
         rotY=0., rotZ=0., rotX=0., sca=(1., 1., 1.), cyl=False, delMpd=False, mNam=''):
    """PolyData变换"""
    pd = getPd(mPd)
    if p0 is None:
        p0 = pdCp(pd)
    if (go != OP).any():
        nor = psDst(go-p0)[0]
        if cyl:
            p0 = (p0+go)/2
        rotZ = -90
    if oMat is None and nor is not None:
        oMat = oriMat(nor)
    tf = vtk.vtkTransform()
    tf.Translate(p0)
    if nor is not None:
        tf.Concatenate(oMat)
        tf.RotateZ(rotZ)
    else:
        tf.RotateX(rotX)
        tf.RotateY(rotY)
        tf.RotateZ(rotZ)
        tf.Scale(sca)
        tf.Translate(goX, goY, goZ)

    tfPd = vtk.vtkTransformPolyDataFilter()
    tfPd.SetTransform(tf)
    tfPd.SetInputData(pd)
    tfPd.Update()
    pd = tfPd.GetOutput()
    if delMpd is True and isinstance(mPd, MD):
        SCEN.RemoveNode(mPd)
    return getNod(pd, mNam)

print('funEnd')
#%%