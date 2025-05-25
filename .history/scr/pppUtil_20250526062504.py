"""pppUtil.py: Everything ppp-related for PPP.

__author__ = "Jumbo Jing"
"""
#%%
import os
import sys
# import handcalcs.render

qprc_path = os.path.abspath(os.path.join('.'))
if (qprc_path not in sys.path):
    sys.path.append(qprc_path)
from Helper import *
# import skLogic as sk
# from pppUtil import *
from scipy.optimize import minimize as sOpt
from scipy.spatial.transform import Rotation as Rt
from queue import Queue
import time
from functools import wraps
from numpy.linalg import norm
# import TotalSegmentator as ttSeg  # 导入
from itertools import combinations as ittCom
from typing import Callable, Any
import itertools as itt
# from skimage.measure import label, find_contours
# from skimage.measure import regionprops as lbData
import scipy.ndimage as ndIm
from scipy.ndimage import center_of_mass as ndCp, \
    binary_dilation as dila, \
    binary_erosion as erod, \
    label as scLb, \
    sum_labels as lbSum
from scipy.spatial import cKDTree as kdT

# from skimage import measure as skMs
# from viztracer import VizTracer
np.set_printoptions(precision=4, suppress=True)

SCEN = slicer.mrmlScene
SNOD = SCEN.AddNewNodeByClass
SVOL = "vtkMRMLScalarVolumeNode"
LVOL = "vtkMRMLLabelMapVolumeNode"
# ndA = lambda *x: np.asanyarray(
#     x[0] if len(x) == 1 else x)
npl = np.linalg
import slicer
from sitkUtils import PullVolumeFromSlicer, PushVolumeToSlicer
# 从Slicer中拉取体数据的简写
puSk = PullVolumeFromSlicer
# 从Slicer中推送体数据到SimpleITK的简写
skPu = PushVolumeToSlicer
sk2Sc_ = lambda vol, nam='', isLb=0, tNod=None: \
    puSk(vol, tNod, nam, SVOL if isLb == 0 else LVOL)
LBLVS = {
    1: 'C1',  2: 'C2',  3: 'C3',    4: 'C4',   5: 'C5',   6: 'C6',   7: 'C7',
    8: 'T1',  9: 'T2',  10: 'T3',  11: 'T4',  12: 'T5',  13: 'T6',  14: 'T7',
    15: 'T8', 16: 'T9',  17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',  21: 'L2',
    22: 'L3', 23: 'L4',  24: 'L5',  25: 'S1',  26: 'Sacrum', 27: 'Cocc',
    28: 'Cord', 29: 'L6', 50: 'Vbs'}
TLDIC = {14: 'T7',
         15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12',
         20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'S1',
         26: 'Sc', 27: 'vBs'}  # 标签🔠
# TLDIC.values()
TLs_ = ['T7',
        'T8', 'T9', 'T10', 'T11', 'T12',
        'L1', 'L2', 'L3', 'L4', 'L5', 'S1',
        'Sc', 'vBs']  # 签🗒️(含L6)
lbNam = TLs_[:-3]

P = Seq[Tuple[float, float, float]]
PS = Seq[P]
OP = np.zeros(3)
EPS = 1e-6

RL = R, L = 'RL'
AP = A, P = 'AP'
SI = S, I = 'SI'
# COLORS = [ndA([0., 1/lb, 0.]) if lb != 0 else OP for lb in range(0, 256)]
def tsLb2vers(lb): return 51-int(lb)
print('import')


def isLs(ls): return isinstance(ls, (list, tuple, np.ndarray))


MD = slicer.vtkMRMLModelNode
MKS = slicer.vtkMRMLMarkupsNode
VOL = slicer.vtkMRMLVolumeNode
NOD = (MD, MKS, VOL)


def sNam(nam, suf):
    if nam == '':
        return nam
    # elif nam[0] == '_':
    #     return suf+nam
    else:
        return nam+'_'+suf


def ndA(*x):
    """将输入转换为NumPy数组
    🔱 参数:
        *x: 输入数据
    🎁 返回:
        NumPy数组
    """
    if len(x) == 1:
        arr = x[0]
    else:
        arr = x

    # 处理None值
    if arr is None:
        return np.array([])

    # 如果是列表且包含None,先过滤None
    if isinstance(arr, list):
        arr = [item for item in arr if item is not None]

    try:
        return np.asanyarray(arr)
    except ValueError:
        # 处理不规则形状
        if isinstance(arr, (list, tuple)):
            # 返回对象数组
            return np.array(arr, dtype=object)
        raise


NX = ndA([1., 0, 0])
NY = ndA([0., 1, 0])
NZ = ndA([0., 0, 1])
XYZ = ndA(NX, NY, NZ)
# tag getNods: 多节点-->节点列表


def getNods(nods):
    def list1_(ls):
        return [l for l in list(ls)
                if isinstance(l, list)
                for l in l]
    nods = ut.getNodes(nods, useLists=True)
    return list1_(nods.values())
# len(getNods('*'))

# tag hideNods 隐藏节点

def nodsDisp(nods='*', disp=False, cls=None):
    if cls is None:
        cls = (MD, MKS)
    for nod in getNods(nods):
        if isinstance(nod, cls):
            nod.CreateDefaultDisplayNodes()
            nod.GetDisplayNode().SetVisibility(disp)
# tag dspNod 显示节点


def dspNod(nod, mNam='', cls=None, color=None, opacity=None):
    if mNam != '':
        # mNam = 'mod'
        nod = getNod(nod, mNam=mNam)
        nod.CreateDefaultDisplayNodes()
        dpNod = nod.GetDisplayNode()
        # if
        return dpNod

# tag pdMDisp 模型显示


def pdMDisp(polydata, mNam='', **kw):
    """polydata to model
    """
    if mNam == '':
        return polydata
    if isinstance(polydata, VPD):
        model = SCENE.AddNewNodeByClass(MOD, mNam)
        model.SetAndObservePolyData(polydata)
        dpNod = dspNod(model, mNam)
        dpNod.SetColor(0, 0, 1)
        dpNod.SetOpacity(0.2)
        return model
    elif isinstance(polydata, vtk.vtkImageData):
        vol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        vol.SetIJKToRASMatrix(slicer.vtkMRMLSliceNode.GetXYToRAS())
        vol.SetAndObserveImageData(polydata)
        vol.CreateDefaultDisplayNodes()
        vol.CreateDefaultStorageNode()
        return vol

# tag isNdas 判断是否为ndarray


def isNdas(*arrs):
    ndAs = []
    for arr in arrs:
        ndAs += [getArr(arr),]
    if len(ndAs) == 1:
        return ndAs[0]
    return ndAs


def pPjPln_(p, n, pn): return p - np.dot(p-pn, n)


def psPjPln(nodArr, nor,
            pn=None,
            mNam=''):
    ps = getArr(nodArr)[:]
    if type(nor) == tuple:
        pn, nor = nor
    if pn is None:
        pn = np.mean(ps,
                     axis=0)
    pn, v = ndA(pn), ndA(nor)
    v = psDst(v)[0].astype(ps.dtype)
    d = (ps - pn) @ v
    pss = ps - np.outer(d, v)
    if mNam != '':
        pds2Mod(pss, mNam=mNam)
    return pss

# tag pD_pdPjPln 点投影平面


def pdPj(pds, pn=None, mNam='', **kw):
    """
    🎁: pjPd, pn , pjCut(if reS)
    """
    pd = getPd(pds)
    if isinstance(pn, tuple):
        pp, nor = ndA(pn)
    else:
        nor = ndA(pn)
        pp = pdCp(pd)    
    nor = ndA(nor)
    # print(f'{pn=}')
    pjPla = vtk.vtkProjectPointsToPlane()
    pjPla . SetInputData(pd)
    pjPla . SetProjectionTypeToSpecifiedPlane()
    pjPla . SetOrigin(*pp)
    pjPla . SetNormal(nor)
    pjPla . Update()
    pjPd = pjPla.GetOutput()
    return cnnEx(pjPd, mNam=mNam, **kw)

# tag p2pXyz 点对点生成v, w, u
def p2pXyz(
        nors=NX,
        # zNor=None,
        sort=True,
        pp=None,
        mNam=''):
    ''' 点对点生成v, w, u
      🎁: dDic: (rDrt: , aDrt: , sDrt: )
      vwu   : v, w, u
    '''
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
    if mNam != '':
        if p0 is None:
            p0 = OP
        vtkPln((p0, vwu[2]), mNam=mNam)
    return vwu


def isUv_(v): return np.isclose(norm(ndA(v)), 1)


def nx3ps_(ps, n=3, flat=True):
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


def rgN_(x1=10, stp=1, x0=0): return \
    np.arange(x0, x1, stp)[:, None]


def lnPs(l=np.arange(0, 10),
         p=OP,
         n=ndA([NX, NY, NZ]),
         flat=False):
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
    gps = p+n*(
        l if isinstance(l*1., float) else
        l[:, None])
    if flat:
        return nx3ps_(gps)
    return gps


def psDst(p_ps=None, nor=True):
    p_ps = ndA(p_ps)
    if p_ps.shape == (3,):
        p_ps = ndA(p_ps)
        dts = norm(p_ps)
        if dts < EPS:
            # print('距离过小, 是一个点吧?')
            pass
        if nor:
            dts = (p_ps/(dts+EPS), dts)
    else:
        dts = norm(p_ps, axis=1)  # , keepdims=True)
        if (dts < EPS).any():
            # print(f'第{np.where(dts<EPS)[0]}个点距离为0')
            pass
        if nor:
            dts = (p_ps/(dts+EPS)[:, None], dts)
    return dts


def uNor(n): return psDst(n, 1)[0]


def lnNod(p0, p1, mNam='', dia=1.):
    lnod = SNOD("vtkMRMLMarkupsLineNode", mNam)
    lnod.AddControlPoint(vtk.vtkVector3d(p0))
    lnod.AddControlPoint(vtk.vtkVector3d(p1))
    lnod.CreateDefaultDisplayNodes()
    dspNod = lnod.GetDisplayNode()
    dspNod.SetLineDiameter(dia)
    return dspNod


def psLn(ps0, ps1=None, nors=None,
         plus=0, dia=1., mNam='', **kw):
    ps0 = ndA(ps0)
    if ps1 is not None:
        ps1 = ndA(ps1)
        nors, dts = psDst(ps1-ps0)
        dts += plus
        pt = lpn_(dts, ps0, nors)
    elif nors is not None:
        nors = psDst(nors, 0)
        dts = plus
        pt = lpn_(dts, ps0, nors)
    if mNam != '':
        if pt.shape == (3,):
            lnNod(ps0, pt, mNam=mNam, dia=dia)
        else:
            for i in range(pt.shape[0]):
                p0 = ps0 if ps0.shape == (3,) else ps0[i]
                lnNod(p0, pt[i],
                      mNam=f'{mNam}_{i}',
                      dia=dia, **kw)
    return pt, nors, dts, \
        lambda l=0, p=ps0, n=nors: lpn_(l, p, n)
# psLn(ps0, ps1)


def rgN_(x1=10., stp=1., x0=0.): return \
    np.arange(x0, x1, stp)[:, None]


def sortNors(drts, points):
    # 一次性计算所有方向与所有点的点积
    dots = np.abs(np.dot(points, drts.T))  # shape: (n_points, n_directions)
    # 计算每个方向的标准差
    spreads = np.std(dots, axis=0)  # shape: (n_directions,)
    # 获取排序索引
    order = np.argsort(spreads)
    return drts[order]

# gps = lnGps_(cns)
def getInGps(pjPs, gps=None, nor = None, mNam=''):
    '''获取体数据中的点集
    
    🧮 函数: 获取体数据中的点集'''
    pjPs = nx3ps_(pjPs)
    if nor is None:
        nor = psFitPla(pjPs)
    if gps is None:
        gps = obGps(pjPs, nor, stp=1/3)    
    _, pjT, kdO_ = kdOlbs_(pjPs, 1.)
    inds = kdO_(gps)
    gMsk = np.where(inds > 0, 0, 1)
    bMsk = np.zeros_like(gMsk, dtype=bool)
    bMsk[[0, -1], :] = gMsk[[0, -1], :] == 1
    bMsk[:, [0, -1]] = gMsk[:, [0, -1]] == 1
    # stt = np.array([[0,1,0],
    #                 [1,1,1],
    #                 [0,1,0]], dtype=bool)
    stt = np.ones((3, 3), dtype=bool)
    bds = dila(bMsk, stt, -1, gMsk==1)
    msk = gMsk.copy()
    msk[bds] = 0
    # 使用形态学开运算去除孤立噪点
    # strel = np.array([  [0,1,0],
    #                     [1,1,1],
    #                     [0,1,0]], dtype=bool)
    # msk = ndIm.binary_opening(msk, structure=strel)
    inGps = gps[msk>0]
    ctId = pjT.query(inGps)[1]
    ctPs = pjPs[ctId]
    if mNam!='':
        pds2Mod(inGps, sNam(mNam, 'inGps'))
        pds2Mod(ctPs, sNam(mNam, 'ctPs'))
    return msk, inGps, ctPs
# tag p2pLn 点对点生成线


def p2pLn(p0: P,
          p1: Opt[P] = None,
          nor: Opt[P] = None,
          plus: float = 0.,
          flat: bool = False,
          mNam: str = '',
          dia: float = 1.,
          **kw
          ):
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
        lNod = SCEN.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", mNam)
        lNod.AddControlPoint(vtk.vtkVector3d(p0))
        lNod.AddControlPoint(vtk.vtkVector3d(pt))
        lNod.SetNthControlPointVisibility(1, 0)
        dspNod = lNod.GetDisplayNode()
        dspNod.SetCurveLineSizeMode(1)
        dspNod.SetLineDiameter(dia)
        # dspNod.SetColor(0, 0, 1)
        # dspNod.SetOpacity(0.2)
        # dspNod.SetVisibility(1)
        # dspNod.SetTextScale(0)
        dspNod.UseGlyphScaleOff()
        dspNod.SetGlyphType(6)
        dspNod.SetGlyphSize(dia)
        # Helper.markDisp(mNam, gType=3, lDia=dia)
        if mNam[-1] == '_':
            p3Cone(pt, nor, 
                rad=dia*1.4, high=dia*4, 
                mNam=sNam(mNam, "pt"))
    return pt, nor, dst, \
        lambda l=dst, p=p0, n=nor: p+l*n

def p3Cone(bP, drt=None, mNam='',
           rad=1, high=3, seg=6, 
           hP=None,rP=None, *kw):
        # 设置bP为椎底坐标, rP为椎底圆边任一点, hP椎顶坐标
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

# tag addPad: 添加pad
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


def psFitPla(ps, sd=-1, mNam=''):
    """
    🔱 参数:
        ps: 点集
        mNam: 模型名称
        sd: 主方向. -1: 最大方向, 0: 最小方向(法向量), 
        1: 次大方向
    🎁 返回:
        平面法向量
    """
    ps = getArr(ps)
    cp = np.mean(ps, axis=0)
    mNor = npl.svd((ps-cp).T)[0][:, sd]
    mNor = uNor(mNor)
    if mNam != '':
        if sd!=0:
            vtkPln((cp, mNor), mNam=mNam)
        else:
            px = p2pLn(cp, nor=mNor, plus=50)[-1]
            p2pLn(px(50), px(-50), mNam=mNam)
    return mNor


# tag vtkCln 清理pD


def pdCln(Pd):
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(Pd)
    cleaner.Update()
    return cleaner.GetOutput()


def vtkCyl(
        pNor,
        rad: float,
        mPd=None,
        inPd=False,
        mNam='',
        sp=None,
        exTyp='Lg',
        **kw):
    cp, nor = ndA(pNor)
    vCyl = vtk.vtkCylinder()
    vCyl.SetRadius(rad)
    vCyl.SetCenter(cp)
    vCyl.SetAxis(nor)
    if mPd is not None:
        return vtkPlnCrop(mPd, vCyl,
                          inPd=inPd,
                          mNam=mNam,
                          sp=sp,
                          exTyp=exTyp,
                          **kw)
    return vCyl


def bbxArr(ps, bbx, scale_factor=1e6): return \
    ps[np.all((np.round(ps * scale_factor) >= np.round(bbx[0] * scale_factor))
              & (np.round(ps * scale_factor) <= np.round(bbx[1] * scale_factor)),
              axis=1)]


def ppsArr(ps, pps, ax=1): return \
    ps[np.all((ps[:, ax] >= pps[0][ax])
              & (ps[:, ax] <= pps[1][ax]),
              axis=ax)]

def dotPn(ps, pn):
    ps, pn = ndA(ps, pn)
    if isinstance(pn, tuple):
        return (ps-pn[0]) @ pn[1]
    else:
        return ps @ pn

def ps_pn(pds, pn, typ='min'):
    """
    在点集中找到距离平面最近或最远的点

    参数:
        pds (array-like): 要评估的点集
        pn (tuple): 包含平面原点和法向量的元组
        typ (str): 指定查找'min'(最近点)还是'max'(最远点)

    返回:
        tuple: 包含所选点、其到平面的距离及其在输入数组中的索引的元组
    """
    pds = nx3ps_(pds)  # 确保输入点集是二维数组
    op, nor = ndA(pn)  # 解包平面参数
    pjs = (pds-op) @ nor  # 计算每个点到平面的投影距离
    
    if typ == 'min':
        id_ = np.argmin(abs(pjs))  # 找到距离最小点的索引
        dst = abs(pjs[id_])
    elif typ == 'max':
        id_ = np.argmax(pjs)  # 找到距离最大点的索引
        dst = abs(pjs[id_])
    elif typ is None:
        id_ = range(len(pjs))
        dst = np.sort(pjs)
    else:
        raise ValueError("typ参数必须是'min','max'或None。")        
    if 0 <= id_ < len(pds):  # 检查索引是否有效
        return pds[id_], dst, id_  # 返回点坐标、距离和索引
    else:
        raise IndexError(f"索引{id_}超出给定点集范围。")


# tag dotPlnX 平面裁剪点集
def dotPlnX(pds, pln, 
            eqX = 0, rtnPjx=False, isIn=False
            ):
    '''dotPlnX 平面裁剪点集
    🧮 函数: 根据平面将点集分为两部分或获取特定距离的点
    🔱 参数:
        pds: 点集
        pln: 平面(点,法向量)或仅法向量
        eqX: 等值线距离,0表示平面上的点,1表示正半区,None表示分割点集
        rtnPjx: 是否返回投影距离和点集选择器
        isIn: 是否仅返回布尔索引
    🎁 返回:
        根据参数返回不同结果:
        - 当eqX=0: 返回距离平面最近的点
        - 当eqX=±1: 返回正/负半区点集
        - 当eqX为其他值: 返回距离平面为eqX的点
        - 当eqX=None且rtnPjx=True: 返回(投影距离,点集选择器)
        - 当eqX=None且rtnPjx=False: 返回(正半区点集,负半区点集)
    '''
    # 确保输入为numpy数组
    pds = getArr(pds)
    ps = nx3ps_(pds)
    
    # 解析平面参数
    if isinstance(pln, (tuple, list, np.ndarray)):
        if len(pln) == 2:
            op, nor = ndA(pln)
        else:
            nor = ndA(pln)
            op = ps.mean(0)  # 使用点集中心作为平面原点
    else:
        # 处理vtk平面对象
        pln = getNod(pln)
        op, nor = ndA(pln.GetOrigin(),
                      pln.GetNormal())
    
    # 计算每个点到平面的投影距离
    pjs = (ps - op) @ nor
    
    # 根据eqX参数处理不同情况
    if eqX is not None:
        if eqX == 0:
            # 返回距离平面最近的点
            return ps[np.argmin(abs(pjs))]
        elif abs(eqX) == 1:
            # 返回正半区或负半区点集
            lb = (pjs * eqX) > 0
            return ps[lb]
        else:
            # 返回距离平面为eqX的点
            lb = pjs == eqX
            return ps[lb]
    else:
        # 分割点集
        ids = pjs > 0
        if isIn:
            return ids
    
    # 返回投影距离和点集选择器或分割的点集
    if rtnPjx:
        return pjs, lambda ids=ids: list(ps[ids])
    return ps[ids], ps[~ids]

def dotCut(pds, pln=None, dst=0, 
            thr=(.5, -.5), 
            cp = None, mNam=''):
    '''点集裁切
    🧮 函数: 点集裁切
    🔱 参数:
        pds: 点集
        pln: 裁切平面
        dst: 裁切距离
        thr: 裁切阈值
        cp: 裁切中心
        mNam: 模型名
    🎁 返回:
        裁切后的点集
    🔰 说明: 
    '''
    if isinstance(pds, tuple):
        pjs, pjx = pds
    else:
        if len(pln) > 0 and isinstance( pln[0], 
                                        (tuple, list)):
            # 多平面连续裁切
            cPs = getArr(pds)
            if isinstance(dst, (tuple, list)):
                dst = np.array(dst)
            else:
                dst = np.array([dst,])
            for pl, dt in zip(pln, dst):
                op, nor = ndA(pl)
                if cp is not None: op, nor = rePln_(pl, cp)
                    # # 调整法向使其朝向cp点
                    # dt = np.dot(ndA(cp)-op, nor)
                    # nor = nor if dt >= 0 else -nor
                cPs = dotCut(cPs, (op, nor), dst=dt)
            if mNam != '':
                return pds2Mod(cPs, mNam)
            return cPs
        pjs, pjx = dotPlnX(pds, pln, None, rtnPjx=True)
    def dotJ_():
        if isinstance(thr, (tuple, list)):
            return pjx((pjs<=thr[0])\
                    & (pjs>=thr[1]))
        else:
            return psPj(pjx(pjs<=thr), pln)
    if dst == None: # Cut
        cPs = dotJ_()
    elif dst == 0:  # 单向Crop
        cPs = pjx()
        assert len(cPs) > 0, '无点集'
        tps = dotJ_()
        if len(tps) > 0:
            cPs = np.vstack((cPs, tps))
    elif dst==1:   # 双向Crop
        cPs = pjx(); cPs_= pjx(pjs<0)
        assert len(cPs) > 0, '无点集'
        tps = dotJ_()
        if len(tps) > 0:
            cPs = [ np.vstack((cPs , tps)), 
                    np.vstack((cPs_, tps))]
    else:          # 距离裁切
        cPs = pjx((pjs>0) & (pjs<dst))
        assert len(cPs) > 0, '无点集'
    if mNam != '':
        return pds2Mod(cPs, mNam)
    return cPs

# tag dotCut 点集裁切类
class DotCut:
    '''点集裁切类
    🧮 类: 用于点集裁切操作,支持单平面和多平面裁切
    '''
    def __init__(self, pds, pln=None, 
                 cp=None, dst=0, 
                 thr=(.5, -.5)):
        """初始化点集裁切对象
        🔱 参数:
            pds: 点集或(投影距离,点集选择器)元组
            pln: 裁切平面(点,法向量)或多平面列表
            cp: 裁切中心,用于重定向法向量
            dst: 裁切距离或多平面距离列表
            thr: 裁切阈值(上限,下限),用于确定交线区域
        """
        # 处理已经计算过投影的情况
        if isinstance(pds, tuple) and len(pds) == 2 and callable(pds[1]):
            self.pjs, self.pjx = pds
            self.multi_cut = False
        else:
            # 检查是否为多平面裁切
            if pln is not None and isinstance(pln, (list, tuple)) and len(pln) > 0 and isinstance(pln[0], (tuple, list)):
                self.multi_cut = True
                self.pds = getArr(pds)
                self.pln = pln
                # 确保dst与平面数量匹配
                if isinstance(dst, (list, tuple)):
                    self.dst = np.array(dst)
                else:
                    self.dst = np.array([dst] * len(pln))
            else:
                # 单平面裁切
                self.multi_cut = False
                if pln is None:
                    raise ValueError("裁切平面不能为None")
                self.pjs, self.pjx = dotPlnX(pds, pln, None, rtnPjx=True)
        
        # 存储其他参数
        self.cp = cp
        self.thr = ndA(thr)
        self.dst = dst
        self.pln = pln

    @property 
    def ctPs(self):
        """获取裁切线(平面与点集的交线)"""
        return self.dot_j()

    @property
    def crop(self):
        """获取正向裁切点集(平面正半区及交线)"""
        if self.multi_cut:
            return self.cut(mode='crop')
            
        cPs = self.pjx()
        tps = self.dot_j()
        if len(tps) > 0:
            cPs = np.vstack((cPs, tps))
        return cPs

    @property
    def _crop(self):
        """获取负向裁切点集(平面负半区及交线)"""
        if self.multi_cut:
            return self.cut(mode='_crop')
            
        cPs = self.pjx(self.pjs < 0) 
        tps = self.dot_j()
        if len(tps) > 0:
            cPs = np.vstack((cPs, tps))
        return cPs

    @property
    def crops(self):
        """获取双向裁切点集[正向点集,负向点集]"""
        return [self.crop, self._crop]

    def dot_j(self):
        """获取裁切交线点集
        返回: 平面与点集的交线点集
        """
        if self.multi_cut:
            return self.cut(mode='cut')
            
        # 选择阈值范围内的点并投影到平面上
        mask = (self.pjs <= self.thr[0]) & (self.pjs >= self.thr[1])
        points = self.pjx(mask)
        if len(points) == 0:
            return np.array([])
        return psPj(points, self.pln)

    def cut(self, mNam='', mode='cut'):
        """执行裁切操作
        🔱 参数:
            mNam: 模型名称,非空时将结果转换为模型
            mode: 切取模式
                - cut: 仅取交线
                - crop: 正向裁切(平面正半区及交线)
                - _crop: 负向裁切(平面负半区及交线)
                - crops: 双向裁切[正向点集,负向点集]
                - dist: 距离裁切(平面正半区指定距离内的点)
        🎁 返回:
            根据mode返回裁切后的点集或模型
        """
        try:
            # 处理多平面连续裁切
            if self.multi_cut:
                if len(self.pln) == 0:
                    raise ValueError("多平面裁切需要至少一个平面")
                    
                # 确保平面数量与距离数量匹配
                if len(self.dst) != len(self.pln):
                    self.dst = np.array([self.dst[0]] * len(self.pln))
                
                # 依次应用每个平面进行裁切
                cPs = self.pds
                for i, (pl, dt) in enumerate(zip(self.pln, self.dst)):
                    # 解析平面参数
                    if isinstance(pl, (tuple, list)) and len(pl) == 2:
                        op, nor = ndA(pl)
                    else:
                        raise ValueError(f"无效的平面参数格式: {pl}")
                    
                    # 根据裁切中心重定向法向量
                    if self.cp is not None:
                        op, nor = rePln_(pl, self.cp)
                    
                    # 应用单平面裁切
                    cut_result = DotCut(cPs, (op, nor), dst=dt).cut(mode=mode)
                    
                    # 检查结果
                    if cut_result is None or (isinstance(cut_result, np.ndarray) and len(cut_result) == 0):
                        print(f"警告: 第{i+1}个平面裁切后结果为空")
                        if i > 0:  # 如果不是第一个平面,返回上一步结果
                            break
                        return None
                    
                    cPs = cut_result
                
                # 返回最终结果
                return pds2Mod(cPs, mNam) if mNam else cPs

            # 单平面裁切: 根据mode选择切取点集
            cut_modes = {
                'cut': lambda: self.dot_j(),
                'crop': lambda: self.crop,
                '_crop': lambda: self._crop,
                'crops': lambda: self.crops,
                'dist': lambda: self._distance_cut()
            }

            if mode not in cut_modes:
                raise ValueError(f'无效的切取模式: {mode}')
                
            # 执行裁切
            cPs = cut_modes[mode]()
            
            # 确保有结果点
            if not isinstance(cPs, (list, ValueError)) and len(cPs) == 0:
                print(f"警告: {mode}模式裁切结果为空点集")
                return np.array([]) if not mNam else None
                
            # 返回结果
            return pds2Mod(cPs, mNam) if mNam else cPs

        except Exception as e:
            print(f'裁切失败: {str(e)}')
            import traceback
            traceback.print_exc()
            return None
            
    def _distance_cut(self):
        """距离裁切(平面正半区指定距离内的点)"""
        if self.dst is None:
            raise ValueError('距离裁切需要指定dst!')
        return self.pjx((self.pjs > 0) & (self.pjs < self.dst))


def vtkPlnCrop(mPd, fun, refP=None,
               *, inPd=False, sp=None,
               exTyp='Lg', mNam='', 
               **kw):
    pd = getPd(mPd)
    if isLs(fun):
        fun = addPlns(fun, refP)
    clp = vtk.vtkClipPolyData()
    clp.SetInputData(pd)
    # clp.SetValue(0.0)
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
# tag reNor_ 重新定向

def rePln_(pns, refP=None):
    """重新定向法向量,支持矢量化处理多个方向"""
    
    plns = ndA(pns)
    if refP is None: return plns 
    if plns.ndim == 2:  # 单个平面情况
        op, nor = plns
        vec = refP - op
        return op, nor * np.sign(vec @ nor)
    else:  # 多个平面情况
        # 正确提取所有原点和法向量
        ops = plns[:,0]  # 形状 (n,3)
        nors = plns[:,1]  # 形状 (n,3)
        vecs = refP - ops  # 广播计算
        # 计算批量点积 (n,3) @ (n,3) -> (n,)
        dots = np.einsum('ij,ij->i', vecs, nors)
        return ops, nors * np.sign(dots)[:,None]


def addPlns(funs, refP=None):
    clipFun = vtk.vtkImplicitBoolean() # 定义一个合集隐式布尔函数
    clipFun.SetOperationTypeToUnion()
    for fun in funs:
        if isLs(fun):
            clipFun.AddFunction(vtkPln(fun, refP=refP))
        elif isinstance(fun, vtk.vtkImplicitFunction):
            clipFun.AddFunction(fun)
        elif isinstance(fun, vtk.vtkImplicitBoolean):
            clipFun.AddFunction(fun.GetFunction())
        else:
            raise TypeError("Unsupported function type")
    return clipFun

# # tag reNor_ 重新定向


# def reNor_(op, nor, refP=None):
#     if refP is not None:
#         dt = np.dot(ndA(refP)-op, nor)
#         return (op, nor) if dt >= 0 else (op, -nor)
#     return (op, nor)


# tag vtkCplnCrop 闭裁(close surface)

def vtkCplnCrop(pln,
                mPd, 
                mNam='',
                refP=None,
                **kw):
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

# tag vtkPlns 生成Vk平面s

def vtkPlns(  # 🧮 平面集
        pns: Any,  # 🔱 平面|点集
        mPd=None,
        mNam='',
        pdLs=False,
        cPlns=False,
        refP=None,
        **kw):  # 莫名
    '''vtkPlns 生成Vk平面s
        '''
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
    return plns  # 🎁 平面


def vtkPs(pds):
    ps = getArr(pds)
    vPs = vtk.vtkPoints()
    vPs.SetNumberOfPoints(ps.shape[0])
    vPs.SetData(numpy_to_vtk(ps))
    return vPs


def vtkNors(nors):
    vNors = vtk.vtkDoubleArray()
    vNors.SetNumberOfComponents(3)
    nors_double = nors.astype(np.float64)
    vNors.SetNumberOfTuples(nors_double.shape[0])
    for iv, vec in enumerate(nors_double):
        for ic, comp in enumerate(vec):
            vNors.SetComponent(iv, ic, comp)
    return vNors

# tag vtkPln 平面


def vtkPln(
        pln,
        mPd: vtk.vtkPolyData = None,
        mNam: str = '',
        refP: PS = None,
        cPlns: bool = False,
        **kw
):
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

# tag SPln slicer平面


def SPln(
    nor: PS,
    cp: PS,
    mNam: str = ""
) -> any:
    pln = SNOD('vtkMRMLMarkupsPlaneNode', mNam)
    pln.SetCenter(cp)
    pln.SetNormal(nor)
    return pln


def p3Box(p3s, mNam = "pln"):
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
    plane.SetOrigin(p3s[0])
    plane.SetPoint1(p3s[1])
    plane.SetPoint2(p3s[2])
    plane.Update()
    polydata = plane.GetOutput()
    return getNod(polydata, mNam)

def vtkAquash(pd, plns, refP=None, mNam=''):
    """vtkPolyData压扁
    参数:
        pds: vtkPolyData对象
        plns: 切割平面列表
        refP: 参考点
        mNam: 模型名称  
    返回:
        pd: 压扁后的vtkPolyData对象
    """
    pd = getPd(pd); plns=ndA(plns)
    if refP is None:
        refP = pdCp(pd)
    ops, nors = rePln_(plns, refP)
    pd0s = []
    for op, nor in zip(ops, nors):
        cut = vtkCut(pd, (op, nor), )
        oGps, obx = obBx(cut, grid=1)[-1]
        pBx = vtkPush(obx)
        bxDic = pd2Dic(pBx)
        xPd = pdsBool_(pd, 'x', pBx)
        bxT = kdT(nx3ps_(xPd))
        pnPs = xPd[bxT.query(nx3ps_(oGps))[1]]
        bxDic['points'] = pnPs
        pd0s.append(dic2Pd(bxDic))
    pd1 = vtkPlns(plns, pd,refP=refP, cPlns=True)
    pds = pdAndPds(pd0s+[pd1,], mNam=mNam)
    return pds

def vtkGridPln(ps = OP, nor = NZ, 
            stp = 1.0, 
            siz = (10., 10.),
            push = 0.0,
            mNam = ''):
    """
    创建一个平面网格，并返回其vtkPolyData对象。
    
    参数:
        op (tuple): 平面原点坐标 (x, y, z)
        nor (tuple): 平面法向量 (nx, ny, nz)
        stp (float): 网格步长
        pad (float): 网格填充量
        mNam (str): 网格名称
        size (float or tuple): 网格大小。如果是float，表示正方形边长；
                              如果是tuple (size_x, size_y)，表示矩形尺寸
        
    返回:
        vtk.vtkPolyData: 平面网格的vtkPolyData对象
    """
    ps = ndA(ps)
    if len(ps) == 3:
        op, xp, yp = ps
        n = uNor(np.cross(xp-op, yp-op))
        if stp!=0:
            siz = norm(ps[1:] - op, axis=1)
    else:
        siz = ndA(siz)
        # 归一化法向量
        n = uNor(nor)
        u, v = p2pXyz(n, False)  # 计算局部坐标系的基向量
        op = ps[:]
        xp, yp = op + .5*siz*ndA(u, v)   # 局部 X 方向端点
    
    # 创建平面数据源
    pln = vtk.vtkPlaneSource()
    pln.SetOrigin(op)                # 中心点
    pln.SetPoint1(xp)                # 局部 X 方向端点
    pln.SetPoint2(yp)                # 局部 Y 方向端点
    if stp!=0:
        gSiz = (siz/stp).astype(int)
        pln.SetXResolution(gSiz[0])  # X 方向分辨率（单元格数）
        pln.SetYResolution(gSiz[1])  # Y 方向分辨率（单元格数）
    pln.Update()  # 更新数据源以生成网格
    # 获取生成的网格数据
    grid = pln.GetOutput()
    if stp!=0: 
        gArr = getArr(grid).reshape(gSiz[0]+1, gSiz[1]+1, 3)
    else:
        gArr = getArr(grid)
    if push!=0: grid = vtkPush(grid, n, push)
    if mNam != '':
        _=getNod(grid, mNam=mNam) # 可选，为网格数据添加名称
    return gArr, grid
def vtkPush(pd, nor=NZ, dst=1., mNam=''):
    pd = getPd(pd)
    push = vtk.vtkLinearExtrusionFilter()
    push.SetInputData(pd)
    push.SetVector(*nor)  # 设置挤出方向为Z轴正方向
    push.SetScaleFactor(dst)
    push.Update()
    return getNod(push.GetOutput(), mNam)
def pdsBool_(pd0, opt, pd1, rtnPs=True, mNam="modBool"):
        '''
        Param:Add:A+B;Sub:A-B;Ins:A&B
        '''
        import vtkSlicerCombineModelsModuleLogicPython as vtkbool
        bool = vtkbool.vtkPolyDataBooleanFilter()
        odic = {'+': bool.SetOperModeToUnion(),
                '-': bool.SetOperModeToDifference(),
                'x': bool.SetOperModeToIntersection()}
        pd0 = getNod(pd0)
        pd1 = getNod(pd1)
        try:
            odic[opt]
        except KeyError:
            raise ValueError(f"Invalid operation: {opt}. Use '+', '-', or 'x'.")
        bool.SetInputData(0, pd0) if isinstance(pd0, vtk.vtkPolyData) else \
                bool.SetInputConnection(0, pd0.GetPolyDataConnection())
        bool.SetInputData(1, pd1) if isinstance(pd1, vtk.vtkPolyData) else \
                bool.SetInputConnection(1, pd1.GetPolyDataConnection())
        bool.Update()
        pd = bool.GetOutput()
        if rtnPs:
            ps = pd.GetPoints().GetData()
            return vtk_to_numpy(ps)
        return getNod(pd, mNam)

# tag pD_obxCrop 裁剪obbBox


def obxCrop(pd, obPd=None, 
            nors=None, drts=None, 
            rfCp=None, mNam='', 
            pad=1):
    '''obxCrop obbBox裁剪pd
    📒: 1. 若drts为None,则从obPd中提取obbBox;
        2. bx 二维则 4 平面, 三维则 6 平面
    '''
    cpd = getPd(pd)

    if nors is not None:
        if obPd is None:
            pvs = ndA
            nors, _, cns, cp = obBx(cpd, mNam, nors, pad)[:4]
        else:
            nors, _, cns, cp = obBx(obPd, mNam, nors, pad)[:4]
    else:
        if isinstance(obPd, tuple):
            nors, cns = ndA(obPd)
            cp = cns.mean(axis=0)
        else:
            nors, _, cns, cp = obBx(obPd, mNam, nors, pad)[:4]
    p0 = cns[0]
    p1 = findPs(cns[1:], p0, mTyp='max')[0]
    if drts is None:
        drts = nors
    ln = len(drts)
    ps = [p0]*ln + [p1]*ln
    vs = list(drts) + list(-drts)
    if rfCp is None:
        rfCp = cp
    pvs = [(p, v) for p, v in zip(ps, vs)]
    cpd = vtkPlns(pvs, cpd, refP=rfCp)
    return cnnEx(cpd, mNam=mNam)


# tag vtkCut
def vtkCut(mPd, pln, mNam='', pad=3,
           lmd=False, **kw):
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
        return pd, lambda nor=nor, op=op: \
            vtkCutter__(mPd, vtkPln((op, nor)))
    return pd


def vtkPdNor(pd):
    pd = getPd(pd)
    norFt = vtk.vtkPolyDataNormals()
    norFt.SetInputData(pd)
    norFt.SetFlipNormals(0)
    norFt.AutoOrientNormalsOn()
    norFt.Update()
    return norFt.GetOutput()

# tag cnnEx 连通区提取


def spCnnex(pd, sp, mNam='', pdn=False):
    """根据给定的点集 sp，从 pd 中提取与每个点最近的连通区域"""
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
        # 单个点的情况
        return cnnSp__(sp)
    elif sp.ndim == 2:
        # 多个点的情况
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


def cnnEx(mPd, mNam='',
          *,
          sp=None,
          exTyp: Lit['All', 'Lg', None] = None,
          pdn=False,
          ):
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

# tag vtkCnnEx 连通区提取


def cnnsEx(cnn, reverse=True):
    output = cnn.GetOutput()
    num_regions = output.GetNumberOfRegions()
    rgns = []
    ps = output.GetPoints()

    for i in range(num_regions):
        ids = vtk.vtkIdList()
        cnn.GetRegionPointIds(i, ids)

        region_points = [ids.GetId(j) for j in range(ids.GetNumberOfIds())]
        region_coords = [ps.GetPoint(point_id) for point_id in region_points]

        rgns.append({
            'coords': region_coords,
            'size': len(region_coords)
        })

    # 按大小排序
    sorted_regions = sorted(rgns, key=lambda r: r['size'], reverse=reverse)

    return sorted_regions

# 使用示例
# cnn.SetExtractionModeToAllRegions()
# cnn.Update()
# sorted_regions = extract_size_sorted_regions(cnn, reverse=True)

# 现在 sorted_regions 是一个按大小降序排序的区域列表
# tag vtkPush 拉伸

# tag pD_ps2mod 点集转模型


def pds2Mod(pds,
            mNam: str = '',
            psRad=9.,
            refPd = None,
            **kw
            ):
    ''' 点集转形数
    📒: 若psRad >0,则显示点集; 
                =0则显示模型
    🔱 参数:
        pds: 点集数据
        mNam: 模型名称
        psRad: 点大小，>0显示点集，=0显示模型
        refPd: 参考PolyData对象或字典，用于提供除点集外的其他属性
        **kw: 其他参数
    '''
    arr = nx3ps_(pds)
    if refPd is not None:
        if isinstance(refPd, dict):
            # 如果refPd已经是字典，直接使用
            pdC = pd2Dic(getPd(pds))
            pdC = pdC | refPd
        else:
            pdC = pd2Dic(refPd) 
            # 否则将refPd转换为字典
            pdC['points'] = arr
        return dic2Pd(pdC, mNam)
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
    if psRad > 0 and mNam != '':
        mpDp = dspNod(mpd, mNam)
        mpDp.SetRepresentation(
            slicer.vtkMRMLModelDisplayNode.PointsRepresentation)
        mpDp.SetPointSize(psRad)
        mpDp.SetColor(.9, .3, .6)
    return mpd

# tag ls2Dic 🗒️转🔠
# seq2dic


def ls2Dic(lbs, ls): return \
    dict(zip(lbs, ls))

# tag findPs 找点


def findPs(pds,
           p,
           mTyp: Union['min', 'max'] = 'min',
           ):
    '''
      🎁
        ps[mI]: 目标点  
        dst: 最小距离
        mI: 最小距离的索引
    '''
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


def psRoll_(ps, p0): return \
    np.roll(ps, -findPs(ps, p0)[-1], 0)


def pNorCir(cp=OP,
            pln=NZ,
            p0=None,
            rad=None,
            sn:   int = 60,
            rp=None,
            mNam: str = "",
            rtnSc=False,
            **kw
            ) -> np.ndarray:
    """三点圆
        - parame:
            cPs: 圆心点坐标
            pcn: 法向量或三点坐标
            p0: 圆上起始点
            rad: 半径
            sn: 圆上点的数量
            rol: 是否旋转到起始点
            mNam: 模型名称
        - return: 圆上点的坐标数组
        - Note: 默认p0为圆心, pcn为法向量, 或者半径为p,pc距离
    """
    cp = ndA(cp)  # 🔢 转换圆心为numpy数组
    pln = ndA(pln)  # 🔢 转换法向量为numpy数组
    if rad is None and p0 is not None:  # 如果没有给定半径
        rad = psDst(cp - p0, 0)  # 使用起始点到圆心的距离作为半径
    if not np.allclose(norm(pln), 1.0, atol=EPS):
        pln = p3Nor_(cp, pln, p0)
    # 使用VTK创建规则多边形
    cir = vtk.vtkRegularPolygonSource()
    cir.SetNumberOfSides(sn)  # 设置边数
    cir.SetRadius(rad)  # 设置半径
    cir.SetGeneratePolygon(0)  # 是否生成面片
    cir.SetNormal(pln)  # 设置法向量
    cir.SetCenter(cp)  # 设置圆心
    cir.Update()

    pd = cir.GetOutput()  # 获取输出
    arr = getArr(pd)  # 转换为numpy数组

    if rp is not None:  # 🚦 如果给定起始点
        rP = findPs(arr, rp)[0]
        arr = psRoll_(arr, rP)  # 将数组旋转到起始点位置

    if mNam != "":  # 🚦 如果给定模型名称
        # p2pLn(cp, arr[0],
        #       mNam=mNam + 'rad')  # 显示半径
        getNod(pd, mNam)  # 显示模型
    if rtnSc:
        return arr, cir
    return arr


def getI2rMat(vol, isArr=True):
    vol = getNod(vol)
    mat = vtk.vtkMatrix4x4()
    vol.GetIJKToRASMatrix(mat)
    if isArr:
        return ut.arrayFromVTKMatrix(mat)
    return mat


def ras2Ijk_(vMat, ps):
    if type(vMat) is not np.ndarray:
        mat = getR2iMat(vMat)
    else:
        mat = vMat
    ps = getArr(ps)
    mat = ndA(vMat)
    # 确保ras是numpy数组
    # if ps.ndim == 1:
    #   ps = ndA([ps])
    if ps.ndim == 2:  # 处理二维情况 (n, 3)
        pp = np.ones((ps.shape[0], 1))
        ras1 = np.hstack((ps, pp))
        ijk = (ras1 @ mat.T)[:, :3]
    elif ps.ndim == 3:  # 处理三维情况 (n, m, 3)
        pp = np.ones((*ps.shape[:-1], 1))  # 增加一个维度 (n, m, 4)
        ras1 = np.concatenate((ps, pp), axis=-1)  # 增加一个维度 (n, m, 4)
        ijk = ras1 @ np.swapaxes(mat, -1, -2)  # 交换最后两个维度 (n, m, 4)
        ijk = ijk[..., :3]  # 去掉最后一个维度 (n, m)
    else:
        raise ValueError(
            f"Unsupported dimension for 'ras'. Expected (n, 3) or (n, m, 3), got {ps.shape}")
    return ijk.astype(int)


# tag ras2vks: ras点集转像素
def ras2vks(
        ps,
        reVol=None,
        lb=1,
        pvks=True,
        mNam=''):
    """
    🧮 将RAS坐标系中的点集转换为体素坐标系
    🔱 ps: 点集, 可以是(n,3)或(n,m,3)形状的数组
    🔱 reVol: 参考体素, 默认为场景中第一个体素
    🔱 lb: 标签值, 默认为1
    🔱 pvks: 是否返回点集对应的体素值, 默认为True
    🔱 mNam: 模型名称, 默认为空
    🎁 返回体素数组或体素值
    """
    # 获取参考体素
    if reVol is None:
        reVol = SCEN.GetFirstNodeByClass(LVOL)
    else:
        reVol = getNod(reVol)

    # 获取RAS到IJK的变换矩阵和体素数组
    mat = getR2iMat(reVol)
    vArr = getArr(reVol)

    # 处理输入点集
    ps = getArr(ps)
    pShp = ps.shape
    if len(pShp) > 2:
        ps = nx3ps_(ps)  # 转换为(n,3)形状

    # 添加齐次坐标
    ps1 = np.ones((len(ps), 1))  # (len, 1)
    ps4 = np.hstack((ps, ps1))

    # 坐标变换
    ijk = (ps4 @ mat.T)[:, :3]
    ijk = ijk.astype(int)

    # 裁剪坐标范围
    ijk = np.clip(ijk,
                  a_min=0,
                  a_max=ndA(vArr.shape)[::-1] - 1)

    # 分离坐标分量
    z = ijk[:, 0]
    y = ijk[:, 1]
    x = ijk[:, 2]

    # 返回点集对应的体素值
    if pvks:
        varr = vArr.copy()
        varr = vArr[x, y, z]
        if lb != 0:
            varr = np.where(varr != 0, lb, 0)
        if len(pShp) > 2:
            varr = varr.reshape(pShp[:-1])
        return varr, ijk

    # 返回标记后的体素数组
    mArr = np.zeros_like(vArr)
    mArr[x, y, z] = lb
    if lb == 0:
        mArr[x, y, z] = 1
        mArr *= vArr
    if mNam != '':
        vol = volClone(reVol, mNam)
        ut.updateVolumeFromArray(vol, mArr)
    return mArr

# tag cropVol 裁剪体素 ✅


def cropVol(
        vol,
        roi=None,
        mNam='',
        cArr=None,
        delV=True):
    '''
    🎁: 🚦roi: vd; 🚥: rNod, vd
    '''
    vNod = getNod(vol)
    if roi is None:
        rNod = pdBbx(lVol2mpd(vNod, exTyp="All"), mNam)[-1]
    else:
        rNod = getNod(roi)
    cropLg = slicer.modules.cropvolume.logic()
    cropMd = slicer.vtkMRMLCropVolumeParametersNode()
    # SCEN.AddNode(cropMd)
    cropMd.SetROINodeID(rNod.GetID())
    cropMd.SetInputVolumeNodeID(vNod.GetID())
    cropMd.SetVoxelBased(True)
    cropLg.FitROIToInputVolume(cropMd)
    cropLg.Apply(cropMd)
    cropVol = SCEN.GetNodeByID(
        cropMd.GetOutputVolumeNodeID())
    if mNam != '':
        cropVol.SetName(mNam)

    # cropVol.SetName(vNod.GetName()+'Crop')
    if cArr is not None:
        ut.updateVolumeFromArray(cropVol, cArr[:, ::-1])
    if delV:
        SCEN.RemoveNode(vNod)
    if roi is None:
        return rNod, volData(cropVol, mNam, exTyp="All")
    return volData(cropVol, mNam, exTyp="All")


def getPd(nod, mNam=''):
    """获取vtkPolyData对象
    🧮 函数: 将输入转换为vtkPolyData
    🔱 参数:
        nod: 输入对象(可以是字符串、数组、vtkPolyData等)
        mNam: 模型名称
    🎁 返回:
        vtkPolyData对象
    """
    # 类型检查和转换
    if isinstance(nod, vtk.vtkPolyData):
        # if mNam != '':
        return nod

    elif isinstance(nod, str):
        # 如果是字符串，尝试获取模型节点
        pd = getNod(nod)
        if pd is None:
            raise ValueError(f"找不到模型: {nod}")
        return pd.GetPolyData()

    elif isinstance(nod, np.ndarray):
        # 如果是numpy数组，转换为vtkPolyData
        return arr2pd(nod, mNam)

    elif isinstance(nod, (list, tuple)):
        # 如果是列表或元组，转换为numpy数组再处理
        return arr2pd(ndA(nod), mNam)

    else:
        # 其他类型，尝试获取PolyData属性
        try:
            return nod.GetPolyData()
        except:
            raise TypeError(f"无法将类型 {type(nod)} 转换为vtkPolyData")


def arr2pd(arr, mNam='', **kw):
    """将数组转换为vtkPolyData
    🧮 函数: 将numpy数组转换为vtkPolyData
    🔱 参数:
        arr: numpy数组
        mNam: 模型名称
    🎁 返回:
        vtkPolyData对象
    """
    # 确保是Nx3形式
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
# tag clonePd 克隆pd


def clonePd(pd0, mNam=''):
    vpd = vtk.vtkPolyData()
    vpd.DeepCopy(getPd(pd0))
    return getNod(vpd, mNam)

# tag getNod 获取nod ✅
def getNod(data, mNam=''):
    if isinstance(data, str):
        nod = ut.getNode(data)
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
            ut.setSliceViewerLayers(nod, fit=True) 
    return nod

# tag getArr 获取🔢 ✅
def getArr(
    nod,
    dpcopy=True,
) -> np.ndarray:
    """
    获取array
    """
    if isinstance(nod, vtk.vtkPolyData):
        pd = nod.GetPoints().GetData()
        arr = vtk_to_numpy(pd)
    elif isinstance(nod, str):
        arr = ut.array(nod)
    elif isLs(nod):
        arr = ndA(nod)
    else:
        arr = ut.array(nod.GetID())
    if dpcopy:
        arr = arr.copy()
    return arr

# tag pD_bbx pD包围盒 ✅


def pdBbx(  pd,
            mNam='',
            pad = 1,
            ):
    '''
    🎁: xyxy, size, vm, rNod
    '''
    if isinstance(pd, np.ndarray):
        # if vPd:
        pd = nx3ps_(pd)
        xyxy = ndA( np.min(pd, axis=0)-pad,
                    np.max(pd, axis=0)+pad)    
        siz = xyxy[1]-xyxy[0]
        vm = np.prod(siz)
        # else:
        #     dim = arr.ndim
        #     out = []
        #     for ax in ittCom(range(dim),
        #                     dim - 1):
        #         non0 = np.any(pd, axis=ax)
        #         xxyy = ndA(np.where(non0)[0][[0, -1]])
        #         out     . extend([xxyy],)
        #     xyxy = np.vstack(ndA(out)).T
        #     siz = xyxy[1]-xyxy[0]
        #     # xyxy = vks2pD(xyxy)
        #     vm = 1
        #     for v in siz:
        #         vm *= v
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

# tag pD_getObb: 获取obbT ✅
def getObt(mNod):
    if isinstance(mNod, vtk.vtkOBBTree):
        return mNod
    else:
        pd = getPd(mNod)
        obT = vtk.vtkOBBTree()
        obT.SetDataSet(pd)
        obT.BuildLocator()
        return obT

# tag pD_obbBox 生成obbBox
def obBx(  # 🧮obbBox
    pData,  # 🔱点集
    mNam: str = "",  # 莫名
    nors=None,  # 法向量
    pad: float = 1,  # 添加pad
    grid: float = 0,  # 网格大小
    **kw: any
):
    '''obbBox 生成obbBox
    🎁:
        - drts: xyz🧭
        - dsts: xyz距离
        - cns: 顶点
        - cp: 中心点
        - rNod: node+Mtx
    '''
    pArr_ = getArr(pData)  # 点🔢
    cp = np.mean(pArr_, axis=0)  # 中心点
    pArr = pArr_ - cp  # 点🔢去中心
    if nors is not None:  # 🚦法向量存在
        nors = ndA(nors)  # ~~~~~🔢
        if nors.ndim == 2:
            # 直接检查nors的正交性
            # dotNors = np.dot(nors, nors.T)
            # np.fill_diagonal(dotNors, 0)  # 将对角线元素置0
            # if np.any(np.abs(dotNors) > EPS):  # 检查非对角元素是否接近0
            #     raise ValueError(f"{dotNors=} nors 必须是正交矩阵")
            drts = nors
        elif nors.shape == (3,):
            dt0, dt1 = p2pXyz(nors, sort=False)
            drts = np.array([dt0, dt1, nors])
    else:  # 🚥
        cov = np.cov(pArr.T)  # 协方差矩阵
        cov += EPS*np.eye(3)
        evs, ets = npl.eigh(cov)  # 特征值,特征向量
        order = evs.argsort()[::-1]  # 特征值按照从大到小排序
        drts = ets[:, order].T  # 特征向量
    psR = np.dot(pArr, drts.T)  # 点🔢旋转
    mnps = np.min(psR, axis=0)  # 最小点
    mxps = np.max(psR, axis=0)  # 最大点
    dsts = mxps - mnps  # 距离s
    ids = np.where(dsts > 1.)[0]
    dsts = dsts[ids]; drts = drts[ids]
    cn = list(itt.product(*zip(mnps[ids], mxps[ids])))  # 角点集
    cns = np.dot(ndA(cn), drts) + cp  # ~~~~~旋转回中心
    cns = addPad(cns, pad)  # ~~~~~添加pad
    dsts += pad*2.
    cp = np.mean(cns, axis=0)  # ~~~~~中心点
    if len(dsts) == 2:
        rNod = vtkGridPln(cns[:-1], stp=grid, mNam=mNam)
    else:
        rNod = addRoi(dsts, drts, cp,
                    f"{mNam}_roi", True)  # 生成ROI
    return drts, dsts, cns, cp, rNod  # 🎁🧭s, 📏s, 角点集, 中心点, ROI节点

def obGps(obPd=None, pn=None, 
          siz=(10, 10), stp=1., 
          flat=False, pad=2, 
          mNam=''):
    """生成网格点阵
    🧮 函数: 根据包围盒生成网格点阵
    🔱 参数:
        obPd: 物体或包围盒参数元组(法向量,尺寸,角点)
        nors: 可选的法向量
        stp: 网格步长,默认1
        flat: 是否展平输出,默认False
        pad: 包围盒填充大小,默认1
        mNam: 模型名称,默认空字符串
    🎁 返回:
        gps: 网格点坐标数组
    """
    if obPd is None:
        # 当obPd为None时，使用默认值
        ns = NZ
        ds = siz
        p0 = OP
    else:
        # 如果输入不是元组,计算包围盒参数
        ns, ds, cns = obBx(obPd, 
                            nor=pn, 
                            pad=pad, 
                            mNam=sNam(mNam, 'box'))[:3]
        p0 = cns[0]
    # else:
    #     obPd = ndA(obPd)
    #     # 如果是元组,直接解包
    #     if obPd.shape == (3, ):
    #         cns = obPd[:3]
    #         p0 = cns[0]
    #         ns, ds = psDst(cns[1:] - p0)
    #     elif obPd.shape == (2,):
    #         ns = NZ
    #         ds = (obPd[0], obPd[1])
    #         p0 = OP
            
    #     else:
    #         ns, ds = [
    #             [ls[0] for ls in obPd],
    #             [ls[1] for ls in obPd],
    #             ] # ((p,n,l)...)
    #         p0 = pn
    # 生成第一维度的网格点
    gps = lpn_(np.arange(0, ds[0]+1, stp), 
                p0, ns[0], flat)
    if len(ds) > 1:
        # 如果维度>1,循环生成其他维度的网格点
        for i in range(1, len(ds)):
            gps = lpn_(np.arange(0, ds[i]+1, stp),
                       gps[(slice(None),)*i + (None,)],
                       ns[i], flat)
    if mNam != '':
        # 如果提供了模型名称,转换为模型显示
        pds2Mod(gps, mNam)
    # if nors is None:
    #     return gps, ns
    return gps

# tag roi 生成ROI


def addRoi(dim=[10, 20, 40],
           drts=None,
           cp=None,
           mNam: str = "",
           mtx: bool = False
           ):
    if drts is None:
        drts = ndA([NX, NY, NZ])
    roi = SNOD("vtkMRMLMarkupsROINode", mNam)
    roi.SetSize(*dim)
    if cp is None:
        cp = 0.5*np.dot(dim, drts)
    # roi.SetCenter(cp)
    b2Rt = np.row_stack(
        (np.column_stack((drts[0],
                          drts[1],
                          drts[2],
                          cp)), (0, 0, 0, 1)))
    b2RtMtx = ut.vtkMatrixFromArray(b2Rt)
    roi.SetAndObserveObjectToNodeMatrix(b2RtMtx)
#   Helper.markDisp(roi, **kw)
    dNod = dspNod(roi, mNam)
    dNod.SetColor(0, 0, 1)
    dNod.SetOpacity(0.2)
    if mtx:
        return roi, b2RtMtx
    else:
        return roi

# tag pD_pdCp pD质心


def pdCp(pdata, mNam=''):
    pd = getPd(pdata)
    cpFt = vtk.vtkCenterOfMass()
    cpFt . SetInputData(pd)
    cpFt . SetUseScalarsAsWeights(False)  # 不使用标量作为权重
    cpFt . Update()
    cp = ndA(cpFt.GetCenter())
    if mNam != '':
        addFid(cp, mNam)
    return cp

# tag r2iMat 卷转矩阵


def getR2iMat(vol, arr=True):
    vol = getNod(vol)
    mat = vtk.vtkMatrix4x4()
    vol.GetRASToIJKMatrix(mat)
    if arr:
        return ut.arrayFromVTKMatrix(mat)
    return mat

# tag pD_pds2Vd: ras点集转像素


def pds2Vd(
        pds,
        refVol=None,
        lb=1,
        vks=True,
        mNam=''):
    ps = getArr(pds)
    ps = nx3ps_(ps)
    if refVol is None:
        refVol = SCEN.GetFirstNodeByClass("vtkMRMLLabelMapVolmeNodeNode")
    else:
        refVol = getNod(refVol)
    mat = getR2iMat(refVol)
    vArr = getArr(refVol)
    if lb != 0:
        vArr[vArr > 0] = lb
    ps1 = np.ones((len(ps), 1))  # (len, 1)
    ps4 = np.hstack((ps, ps1))
    ijk = (ps4 @ mat.T)[:, :3]
    ijk = ijk.astype(int)
    # print(f'{ijk.shape=}')
    ijk = np.clip(ijk,
                  a_min=0,
                  a_max=ndA(vArr.shape)[::-1] - 1)
    if vks:
        vArr = vArr[ijk[:, 2],
                    ijk[:, 1],
                    ijk[:, 0]]
        if mNam != '':
            vol = volClone(refVol, mNam)
            ut.updateVolumeFromArray(vol, vArr)
            return volData(vol, mNam)
        return vArr
    return ijk


def vks2Ras(vmData, vks=None, lbs=False):
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

# tag volData 卷数据


def volData(vol, mNam='', **kw):
    ''' 卷数据
      🎁
        vol: 卷数据
        arr: 卷像素集
        ps : 坐标点集
        lbs: 标签集
        id : ID
        imData: vtkImageData
        mod : 💃
        pd  : vtkPolyData  
    '''

    class imInfo:
        def __init__(self, vol):
            self.vol = getNod(vol, mNam)
            self.nam = self.vol.GetName()

        @property
        def update(self):
            if not hasattr(self, '_update'):
                self._update = ut.updateVolumeFromArray(self.vol, self.arr)
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
    assert np.max(
        imProp.arr) != 0, f'the volNodlume data of {imProp.vol.GetName()} is empty'
    return imProp


def vks2pD(vdata, vMat=None, op=None, getLbs=True, mNam=''):
    """ 将体素坐标转换为物理坐标
    """
    def vks2Ps__(vArr, vMat):
        if isinstance(vArr, np.ndarray) and vArr.ndim > 2:
            indices = np.atleast_1d(np.nonzero(vArr))
            vArr = np.stack(indices, axis=-1)[:, ::-1]  # 反转索引顺序
        else:
            vArr = np.atleast_1d(np.array(np.where(vArr != 0))).T[:, ::-1]
        if len(vArr) == 0:  # 处理空数组情况
            return np.zeros((0, 3))
        arr1 = np.ones((vArr.shape[0], 1))
        ijk = np.hstack((vArr, arr1))
        ps = (ijk @ vMat.T)[:, :3]
        return ps
    vdata = getNod(vdata)
    if op is None:
        op = vdata.GetOrigin()
    op = ndA(op)
    vArr = getArr(vdata)
    if vMat is None:
        vMat = getI2rMat(vdata)
    if vMat.shape != (4, 4):
        raise ValueError(f"变换矩阵维度应为 (4,4)，当前为 {vMat.shape}")
    if getLbs:
        lbs = np.unique(vArr)
        lbs = lbs[lbs != 0]
        lbRas = {}
        for lb in lbs:
            ps = vks2Ps__(vArr == lb, vMat)
            lbRas[lb] = ps+op
            if mNam != '':
                pds2Mod(ps, mNam+str(lb))
        return lbRas
    else:
        ps = vks2Ps__(vArr, vMat)
        if mNam != '':
            pds2Mod(ps, mNam)
        return ps+op

# op, nor = inSecVol(vtArr_, vbArr_, mNam='aaa')


def lsDic(dic, lDic, upDate=False) -> dict:
    # ls2Arr = lambda x: ndA([ndA(i[0]) for i in x])
    if not lDic:
        lDic = dfDic(list)
    if upDate:  # 更新则清空lDic中和dic相同key的值后加入新数据
        for k, v in dic.items():
            if k in lDic.keys():
                lDic[k] = []
    for k, v in dic.items():
        data = [(k, v)]
        for (key, value) in data:
            lDic[key].append(value)
    return lDic


def readVtk(filePath):
    # 创建一个vtkPolyDataReader对象
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filePath)
    reader.Update()  # 更新读取器以读取数据

    # 获取读取的polyData
    polyData = reader.GetOutput()

    return polyData

def pdAndPds(pds, mNam=''):
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
# vd = volData('vt', lbs=True)
# ps17 = self.vtVd.ps[self.lv]


volClone = lambda vol, nam='': \
    slicer.modules.volumes.logic().CloneVolumeGeneric(
        SCEN, vol, nam)
# tag rayCastTkd 射线kdT


def rayCastT(pds, pps, nor=None, dst=0, r=1, mTyp='min'):
    """
    Perform a ray casting operation to find the intersection points.

    🔱 参数:
    pds: 点集数据
    pps: 投影点集
    nor: 法向量 (默认值为 None)
    dst: 距离 (默认值为 0)
    r: 半径 (默认值为 1)
    mTyp: 查找类型 (默认值为 'min')

    🎁 返回:
    pt: 最近的交点
    dt: 距离
    """
    # 将点集数据转换为数组
    ps = getArr(pds)

    # 创建kd树
    pjT = kdT(ps)

    # 计算投影点集
    gps = lGps(pps, nor, dst, r)

    # 查询kd树，找到距离在r范围内的点的索引
    ids = pjT.query(gps, distance_upper_bound=r)[1]

    # 过滤掉超出范围的索引
    ids = ids[ids < ps.shape[0]]

    # 获取有效的点
    pts = ps[ids]

    # 查找最近的点和距离
    dt, pt = findPs(pts, gps[0], mTyp=mTyp)[:-1]

    return pt, dt

# tag rayCast p2p射线


def rayCast(p1, p2=None, mPd='', nor=None, plus=0, oneP: bool = True, inOut=False):
    pt = p2pLn(p1, p2, nor=nor, plus=plus)[0]
    obT = getObt(mPd)
    scPs = vtk.vtkPoints()
    code = obT.IntersectWithLine(p1, pt, scPs, None)
    if inOut:
        return code  # 1: in; 0: none; -1: out
    if code == 0:
        Helper.addLog(f"No intersection points found")
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
            # Helper.addFids(ndA(psInsec),mNam)
            return ndA(secPs)

# tag psPj 点集投影到平面
def pPj_(p, pn): return p - np.outer((p-pn[0])@pn[1], pn[1])

def psPj(pds, pn,
         mNam='',
         flat = True,
         **kw):
    ps = getArr(pds)[:]
    pn = ndA(pn)
    if len(pn) == 2:
        pp, nor = pn
    else:
        pp = np.mean(ps, axis=0)
        nor = pn
    n = uNor(nor)
    pjPs = pPj_(ps, (pp, n))
    if mNam != '':
        pjPd = pds2Mod(pjPs, mNam=mNam, **kw)
        pjPs = getArr(pjPd)
    return pjPs if flat else pjPs.reshape(ps.shape)

def oriMat(norX, drts=False):
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

# tag kdballPs kd树球点集


def kdBalPs(kps, qps=None, rad=None, rtnLg=False, lb=None, mNam=''):
    """球形区域点集搜索"""
    try:
        # 获取或构建KD树
        if isinstance(kps, kdT):
            psT, ps = kps, kps.data
        else:
            ps = nx3ps_(getArr(kps))  # 点集
            if len(ps) == 0:  # 处理空数组情况
                return np.array([])
            psT = kdT(ps)     # KD树

        # 处理查询点
        if qps is None:
            qps = np.mean(ps, axis=0)  # 质心
        else:
            qps = getArr(qps)
            if len(qps) == 0:  # 处理空查询点情况
                return np.array([])

        # 计算搜索半径
        if rad is None:
            rad = findPs(ps, qps)[1]  # 到质心最远距离

        # 球形区域搜索
        inds = psT.query_ball_point(
            qps, r=rad,
            return_length=rtnLg)  # 搜索索引

        # 标签过滤
        if lb is not None:
            inds = inds == lb

        # 获取结果点集
        if len(inds) <= len(ps):
            qPs = ps[inds] 
        else:
            qPs = qps[inds]

        # 检查结果是否为空
        if len(qPs) == 0:
            return np.array([])

        # 输出模型
        if mNam and len(qPs) > 0:
            pds2Mod(qPs, mNam=mNam)  # 转换为模型

        return qPs  # 返回结果点集

    except Exception as e:
        print(f'kdBalPs error: {e}')
        return np.array([])  # 返回空数组


def p2pCyl_(p0, p1,
            rad: float = 1.,
            mNam: str = "",
            Seg: int = 12,
            cap=True,
            **kw
            ) -> any:
    """点对点棱柱"""
    p0 = ndA(p0)
    p1 = ndA(p1)
    drt, dst = psDst(p1-p0)
    cyl = vtk.vtkCylinderSource()
    cyl.SetHeight(dst)
    cyl.SetRadius(rad)
    cyl.SetResolution(Seg)
    cyl.SetCapping(cap)
    cyl.Update()
    cylPd = cyl.GetOutput()
    return pdTf(cylPd, p0, p1, cyl=True, mNam=mNam, **kw)


def pdTf(mPd, p0=OP, go=OP,
         nor=None, oMat=None,
         goX=0., goY=0., goZ=0.,
         rotY=0., rotZ=0., rotX=0.,
         sca=(1., 1., 1.),
         cyl=False,
         delMpd=False,
         mNam=''):
    pd = getPd(mPd)
    if p0 is None:
        p0 = pdCp(pd)
    if (go != OP).any():
        nor = psDst(go-p0)[0]
        if cyl:
            p0 = (p0+go)/2
        rotZ = -90
    if oMat is None and nor is not None:
        # print(f'{nor=}')
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
    return Helper.pdMDisp(pd, mNam)

# tag arr2vol 🔢转vd ✅


def arr2vol(  # 🧮: 🔢列更新vd
    vol: Union[VOL, str]=None,  # 🔱🗞
    arr=0,  # 🏷🔢
    mNam='',  # 莫名
    rtnVd=False,
    pad = 1
) -> VOL:
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
    ut.updateVolumeFromArray(cVol, arr.astype(vArr.dtype))
    # cVol = cnnExVol(cVol, arr, mNam=mNam)
    return volData(cVol) if rtnVd else cVol


# tag lVol2mpd 🏷🗞转mpd✅


def lVol2mpd(lVol, mNam='', **kw):
    vol = getNod(lVol)
    assert isinstance(
        vol, slicer.vtkMRMLLabelMapVolumeNode), \
        f'{type(vol)=}必须是🏷🗞'
    seg = SNOD('vtkMRMLSegmentationNode')
    segLg = slicer.modules.segmentations.logic()
    segLg.ImportLabelmapToSegmentationNode(vol, seg)
    segs = seg.GetSegmentation()
    segn = segs.GetNumberOfSegments()
    getId = segs.GetNthSegmentID

    def getSeg_(ii=0, mNam=mNam, **kw):
        segId = getId(ii)
        pd = vtk.vtkPolyData()
        segLg.GetSegmentClosedSurfaceRepresentation(
            seg, segId, pd, 1)
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


def ls2dic_(ns, ls, i0_=False): return \
    dict(zip(list(ns),
             list(ls[abs(len(ns)-len(ls)):]
                  if i0_ else
                  list(ls[:len(ns)]))))  # 🗒️转🔠{ns:ls}

# tag readIsoCT 读取CT并初始化


def readIsoCT(ctF,
              mNam='',
              isLb=True,
              cstU8 = True
              ):
    '''
    🎁: 🚦isLb==0: Vd
        🚥isLb==1: Vol
    '''
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
        siz = [int(round(osz * osp / sp))
               for osz, osp, sp in
               zip(sizOr, spcOr, spc)]
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
    vol = skPu(img, None, mNam,
               SVOL
               if isLb == False else
               LVOL)
    vol.SetOrigin(OP)
    vol.SetSpacing(spc)
    return volData(vol, mNam)


def addFid(p=OP, mNam='', dia=3):
    if mNam != '':
        fid = SNOD("vtkMRMLMarkupsFiducialNode")
        # slicer.util.updateMarkupsControlPointsFromArray(fid, ndA(p))
        fid.AddControlPoint(p)
        fid.SetName(mNam)
        dpNod = dspNod(fid, mNam)
        dpNod.UseGlyphScaleOn()
        if dia > 0:
            dpNod.UseGlyphScaleOff()
            dpNod.SetGlyphSize(dia)
        return fid

# tag ps2Clns 点集转换为曲线


def ps2cFids(ps, mNam='', lbNam=None, closed=False, lDia=0):
    
    if isinstance(ps, dict):
        # lbNam = list(ps.keys())
        ps = list(ps.values())
    else:
        ps = nx3ps_(ps)
    cls = ["vtkMRMLMarkupsCurveNode",
           "vtkMRMLMarkupsClosedCurveNode"][closed*1]
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


def p2pRot(p0, p1, p2, agl, rad=None,
           res=4, pn=None, mNam=''):
    """生成圆弧
    🧮 函数: 根据三点和角度生成圆弧

    🔱 参数:
        p0: 圆心点
        p1: 起始点 
        p2: 辅助点(用于确定旋转平面)
        agl: 旋转角度(度)
        rad: 圆弧半径,默认使用|p1-p0|
        res: 分辨率,默认4
        pn: 可选的法向量,默认由p0,p1,p2确定
        mNam: 输出模型名称

    🎁 返回:
        圆弧模型节点
    """
    # 转换为numpy数组并计算相对向量
    p0, p1, p2 = ndA(p0), ndA(p1), ndA(p2)
    # 计算法向量(如未指定)
    if pn is None:
        pn = p3Nor_(p0, p2, p1)
    # 计算起始向量和辅助向量
    v1 = p1 - p0  # 起始向量
    # 归一化向量
    vn1 = v1/norm(v1)
    vn2 = uNor(p2-p0)
    # 设置极向量(起始方向)
    if rad is not None:
        v1 = vn1 * rad
    # 根据三点确定旋转方向
    agl *= np.sign(np.dot(np.cross(vn1, vn2), pn))
    # 创建VTK圆弧源
    vtkArc = vtk.vtkArcSource()
    vtkArc.SetCenter(p0)
    vtkArc.UseNormalAndAngleOn()
    vtkArc.SetAngle(agl)
    vtkArc.SetPolarVector(v1)
    vtkArc.SetNormal(pn)
    vtkArc.SetResolution(res)
    vtkArc.Update()
    mpd = getNod(vtkArc.GetOutput(), mNam)
    return getArr(mpd)
# points= p2pRot(p0, p1, p2, 45, res=8, mNam='arc')


def p3Arc(p0, p1, p2, rtnAgl=False, mNam='',
          res=4, rad=None):
    """
    🧮 3点求出一个圆弧
    """
    p0, p1, p2 = ndA(p0, p1, p2)
    v1 = p1-p0
    nv1, n1 = psDst(v1)
    r = rad if rad is not None else n1
    p1_ = p0 + nv1 * r
    vtkArc = vtk.vtkArcSource()
    vtkArc.SetCenter(p0)
    vtkArc.SetPoint1(p1_)
    vtkArc.SetPoint2(p2)
    vtkArc.SetResolution(res)
    vtkArc.Update()
    pd = vtkArc.GetOutput()
    ps = getArr(pd)
    if mNam:
        ps2cFids(ps, mNam=mNam)
    if rtnAgl:
        arc = np.arccos(np.clip(np.dot(nv1, uNor(p2-p0)), -1.0, 1.0))
        return ps, np.degrees(arc)
    return ps
# points, agl = p3Arc(p0, p1, p2, res=8, mNam='arc0', rtnAgl=True)

def p3Angle(p0, p1, p2, rtn='deg'):
    """三点求角度（顶点在p1）
    
    参数：
        p0, p1, p2: 三维坐标点
        deg: 是否返回角度制（默认True），False返回弧度
        
    返回：
        顶点p1处的角度值
    """
    rst = {}
    v1, v2 = ndA(p1, p2) - ndA(p0)    
    cos_ = np.dot(v1, v2) / (norm(v1) * norm(v2))
    rst['cos'] = cos_
    rst['acr'] = np.arccos(np.clip(cos_, -1.0, 1.0))
    rst['deg'] = np.degrees(rst['acr'])
    return lambda x=rtn: rst[x]
def rdrgsRot(v, k, a):
    """罗德里格斯(rodrigues)旋转公式   
    """
    cos_ = np.cos(a)
    sin_ = np.sin(a)
    return v*cos_ + np.cross(k,v)*sin_ + k*k@v*(1-cos_)

def p3Nor_(p, p1, p2): return psDst(np.cross(p1-p, p2-p))[0]

def mxNorPs_(pjPs,  # 🧮 最大方向
             nor=[1, 0, 0], 
             cp=None, mnP=False,  
             rjPs=None,  tp=None# 极值点反投影
             ):
    ''' 在某个方向找极值点 '''
    ps = getArr(pjPs)
    if cp is None:
        cp = ps.mean(0)
    # ps_ = ps-cp
    # nor = ndA(nor) if nor is not None else uNor(tp-cp)
    if tp is not None:  # 🚦 tp存在,🚥 nor为cp<--tp方向*-1
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

# tag ctBd 轮廓点集


def eroDila(msk, mnCn=2, r=1/3):
    """ 分离弱连接区域并保留主体 """
    s = ndA([[0, r, 0],
             [r, 1, r],
             [0, r, 0]])

    return scLb(
        dila(
            scLb(
                erod(
                    msk, s, mnCn),
                s)[0] > 0,
            s, mnCn),
        s)[0]

# def eroDila(msk, mnCn=2, r=1/3):
#     """分离弱连接区域并保留主体
#     参数:
#         msk: 输入二值掩码
#         mnCn: 最小连通区域大小
#         r: 结构元素权重
    
#     返回:
#         处理后的标记数组
#     """
#     # 确保输入是numpy数组
#     msk = ndA(msk)
    
#     # 定义十字形结构元素
#     s = np.array([
#         [0, r, 0],
#         [r, 1, r],
#         [0, r, 0]
#     ])
    
#     # 1. 执行腐蚀操作
#     msk_eroded = erod(msk, s, mnCn)
    
#     # 2. 标记连通区域
#     lbs, num = scLb(msk_eroded)
    
#     # 3. 执行膨胀操作
#     msk_dilated = dila(lbs > 0, s, mnCn)
    
#     # 4. 再次标记连通区域
#     final_lbs, _ = scLb(msk_dilated)
    
#     return final_lbs

# tag ctBd 轮廓点集
def erod_(  msk, 
            gps,
            # ct = 1,
            its=3,
            sp=0,
            r=1/3):
    """分离弱连接区域并保留主体
    参数:
        msk: 输入二值掩码
        gps: 网格点坐标数组
        sp: 种子点坐标
        its: 迭代次数
        r: 结构元素权重
    返回:
        处理后的标记数组
    """
    # 确保输入是numpy数组
    msk = ndA(msk)
    # 定义十字形结构元素
    s = np.array([
        [0, r, 0],
        [r, 1, r],
        [0, r, 0]
    ])
    # 执行腐蚀操作
    edMsk = erod(msk, s, its)
    # 标记连通区域
    lbs, num = scLb(edMsk)
    # 处理种子点或最大连通区域
    if sp is not None:
        if np.all(sp == 0):
            msk_ = (lbs == delBdMsk_(lbs))
        else:  # 使用种子点确定区域
            try:
                spId = findPs(gps, sp)[-1]
                spLb = lbs[spId]
                msk_ = (lbs == spLb)
            except:
                print("警告: 种子点处理失败，使用最大连通区域")
                msk_ = (lbs == delBdMsk_(lbs))
    else:  # 使用最大连通区域
        msk_ = (lbs == delBdMsk_(lbs))
    ctPs = gps[msk_^dila_(msk_, 1)]
    ctPs = psLbs(ctPs)
    return gps[msk_], ctPs
# ips, msk_, inGps = erod_(msk, gps, its=9)

def ctBd(lbs, gps, mNam=''):
    """计算轮廓边界点
    🔱 lbs: 标签数组
        gps: 网格点坐标数组 
        mNam: 标记名称
    🎁 ctps: 轮廓边界点坐标数组
    """
    # 创建与输入标签数组相同形状的零数组作为边界标记
    bdy = np.zeros_like(lbs)  # , dtype=bool)

    # 获取上下左右相邻位置的标签值
    u, d = lbs[1:], lbs[:-1]  # 上下
    r, l = lbs[:, 1:], lbs[:, :-1]  # 左右

    # 检测标签值变化的位置,标记为边界
    bdy[1:] |= (u != d)  # 上下边界
    bdy[:-1] |= (d != u)  # 下上边界
    bdy[:, 1:] |= (r != l)  # 左右边界
    bdy[:, :-1] |= (l != r)  # 右左边界

    # 只保留标签为0的边界点
    bdy &= (lbs == 0)

    # 获取边界点的索引坐标
    ids = np.argwhere(bdy)

    # 根据索引获取边界点的实际坐标
    ctps = gps[ids[:, 0], ids[:, 1]]

    # 如果提供了标记名称,则创建标记点
    if mNam != '':
        ps2cFids(ctps, mNam=mNam)

    return ctps
def psLbs(ps, 
        num = 1,        # 聚类数量
        rad=1.0,        # 邻域半径/体素分辨率
        mnSps=5,        # 最小簇点数
        ax=2,           # 聚类轴
        mNam=''  # 模型名称
        ):
    """聚类点云分群函数"""
    from sklearn.cluster import DBSCAN
    ps = getArr(ps)
    if rad is None:
        def kDst_():
            from sklearn.neighbors import NearestNeighbors
            nbs = NearestNeighbors(n_neighbors=mnSps)
            nbs.fit(ps)
            dsts, _ = nbs.kneighbors(ps)
            return dsts[:, -1]
        # kDst = kDst_()
        # 默认取第95百分位的距离作为候选值
        rad = np.percentile(kDst_(), 95)
        print(f"自动计算邻域半径: {rad:.2f} mm")
    # 🔠密度聚类
    clt = DBSCAN(rad, min_samples=mnSps).fit(ps)
    cLbs_ = clt.labels_
    # 获取所有非噪声点的标签
    lbs_ = np.unique(cLbs_[cLbs_ >= 0])
    num_ = len(lbs_)
    
    if num is not None:
        if num_ < num:
            print(f"警告：可能过分割，建议减小rad（当前{rad}）或增大mnSps（当前{mnSps}）")
        elif np.sum(clt.labels_ == -1) > len(ps)*0.3:
            print(f"警告：噪声点超过30%，建议增大rad（当前{rad}）或减小mnSps（当前{mnSps}）")
        
        # 处理单个聚类的情况
        if num == 1:
            # 获取所有标签的计数
            cnts = np.bincount(cLbs_[cLbs_ >= 0])
            # 找出最大计数的标签
            mxLb = np.argmax(cnts)
            cPs = ps[cLbs_ == mxLb]
            if mNam != '': 
                pds2Mod(cPs, mNam=mNam)
            return cPs
            
        # 如果聚类数量超过需要的数量，只保留最大的num个聚类
        if num_ > num:
            # 计算每个聚类的大小
            sizes = [(lb, np.sum(cLbs_ == lb)) for lb in lbs_]
            # 获取最大的num个聚类
            stLbs = sorted(sizes, key=lambda x: x[1])[-num:]
            lbs_ = np.array([lb for lb, _ in stLbs])
            
        # 直接用numpy高级索引获取每个簇的点集并按指定轴的中位数排序
        clts = []
        meds = []
        
        # 只计算一次点集和中位数
        for lb in lbs_:
            clt = ps[cLbs_ == lb]
            clts.append(clt)
            meds.append(np.median(clt[:, ax]))
        
        # 使用numpy高效排序
        stClts = [clts[i] for i in np.argsort(meds)]
    else:
        # 使用numpy高效获取每个簇的点集
        stClts = [ps[cLbs_ == lb] for lb in lbs_]
        
    # 📊可视化结果
    if mNam:
        for i, cPs in enumerate(stClts):
            pds2Mod(cPs, mNam=f"{mNam}_{i}")
    
    return stClts

def epCut(ps, ePln, pjLn=True, mNam=''):
    ps = getArr(ps)
    if isinstance(ePln, tuple):
        ep, drt = ePln
    else:
        ePln = getNod(ePln)
        ep, drt = ePln.GetOrigin(), ePln.GetNormal()
    pjs, pjx = dotPlnX(ps, (ep, drt), None, 1)
    ctPs = list(pjx())
    
    assert len(ctPs) > 0, 'ePlnCut: ctPs is empty'
    if pjLn:
        pjs0 = pjx(pjs > 0)
        if len(pjx()) > 0 and len(pjs0)>1:            
            ctPs += list(getArr(psPj(pjs0, (ep, drt))))
    ctPs = ndA(ctPs)
    pds2Mod(ctPs, mNam=mNam)
    return ctPs
# 用于通过vbcut和pjps拟合提取椎体内轮廓, 已确定a方向

def kdQ_(kP, qP=None, r=2.0):
    """使用KD树进行近邻点查询
    
    参数:
        kP: 参考点集
        qP: 查询点集，默认为None时使用参考点集 
        r: 搜索半径，默认2.0
        
    返回:
        qPs: 匹配的点集
        query_func: 查询函数
    """
    # 转换输入为numpy数组
    kP = getArr(kP)
    qP_ = kP if qP is None else getArr(qP)
    
    # 构建KD树
    xT = kdT(nx3ps_(kP))
    
    # 执行半径范围查询
    ids = xT.query(qP_, distance_upper_bound=r)[1]
    
    # 过滤有效匹配
    isU = ids < len(kP)
    qPs = kP[ids[isU]] if qP is None else qP_[isU]
    
    # 返回匹配点集和查询函数
    return qPs, lambda q=qP_: xT.query(q, distance_upper_bound=r)[1]

def dila_(msk, delCt=3, knlTyp='enhanced', r=.3, lb=None):
    """增强型膨胀操作，特别处理弱连接
    参数：
        msk: 输入二值掩码
        delCt: 膨胀次数
        kernel_type: 核类型 ('enhanced'强化十字形, 'full'全连接)
        r: 结构元素权重
    """
    # 根据类型选择结构元素
    if knlTyp == 'full':
        s = np.ones((3,3)) * r  # 全连接膨胀
    elif knlTyp == 'cross':
        s = np.array([
                [0, r, 0],
                [r, r, r],
                [0, r, 0]])  # 十字形膨胀
    elif knlTyp == 'enhanced':   
        s = np.array([
                [0, r, r, r, 0],
                [r, r, r, r, r],
                [r, r, r, r, r],
                [r, r, r, r, r],
                [0, r, r, r, 0]]) # 增强型十字形
    
    # 执行分层膨胀增强连接
    for _ in range(delCt):
        msk = dila(msk, s)  # 执行单次膨胀
        # 添加中间处理增强水平/垂直连接
        msk |= dila(msk, 
                    np.array([
                        [0, r, 0],
                        [r, r, r],
                        [0, r, 0]]))
    
    return msk


def ctBd_ed(lbs, gps, delCt=3):
    """计算轮廓边界点并删除指定层数（最终优化版）

    参数：
        lbs: 标签数组
        gps: 网格点坐标数组
        delCt: 删除层数
    返回：
        ctps: 轮廓边界点坐标数组
        msk: 目标区域掩码
    """
    # 初始化目标区域掩码
    msk = (lbs == 1).copy()
    # assert delCt > 0, "delCt必须大于0"
    # 初始边界检测
    bdy = np.zeros_like(msk, dtype=bool)
    bdy[1:] |= (msk[1:] ^ msk[:-1])    # 垂直边界
    bdy[:-1] |= (msk[:-1] ^ msk[1:])
    bdy[:, 1:] |= (msk[:, 1:] ^ msk[:, :-1])  # 水平边界
    bdy[:, :-1] |= (msk[:, :-1] ^ msk[:, 1:])
    # 提取初始边界点
    bdy &= (lbs == 0)     # 目标区域边界
    ids = np.argwhere(bdy)
    # 按x和y分组，收集极值点
    xC, yC = {}, {}
    for x, y in ids:
        xC.setdefault(x, []).append(y)
        yC.setdefault(y, []).append(x)
    # 提取极值坐标
    # cts = set()
    # 提取每行的x极值
    for y, xs in yC.items():
        # cts.add((min(xs), y))
        # cts.add((max(xs), y))
        msk[:, y] = 0
        msk[min(xs):max(xs)+1, y] = 1
    # 提取每列的y极值
    for x, ys in xC.items():
        # cts.add((x, min(ys)))
        # cts.add((x, max(ys)))
        msk[x, :] = 0
        msk[x, min(ys):max(ys)+1] = 1
        # idxy.add(msk[:, y][min(xs):max(xs)+1],)

    # 对每列的y极值进行排序
    # 转换为排序后的numpy数组
    # ids_ = np.array(sorted(cts))
    # ctPs = gps[ids_[:, 0], ids_[:, 1]]
    # msk = ndA(idxy)
    # 使用预定义的腐蚀函数处理多层删除
    if delCt > 0:
        return erod_(msk, gps, delCt)
    # 获取腐蚀后的内部点集
    iPs = gps[msk]
    # 获取轮廓点集
    ctPs = gps[msk^dila_(msk, 1)]
    
    return iPs, ctPs

def kdOlbs_(ps, r=.4, qs=None, rtnLen=True):
    ps = getArr(ps)
    qs = ps if qs is None else getArr(qs)
    psT = kdT(ps)
    lbs = psT.query_ball_point(
        qs, r,
        return_length=rtnLen)
    # print(lbs)
    return lbs, psT, lambda qs=qs, r=r, ps=ps, ln=rtnLen: kdT(ps).query_ball_point(qs, r, return_length=ln)

def kdCt_(ctPs, r=.3, thr=2, cp = None, rad = 1., mNam=''):
    """
    去除平面点集轮廓内的孤立岛点集 (矢量化版本)
    """
    ctPs = getArr(ctPs)
    if cp is None:
        cp = ctPs.mean(0)
    lbs_, _, kdx_ = kdOlbs_(ctPs, r)
    ps_ = ctPs[lbs_>thr]
    # pds2Mod(ps_, mNam=mNam+'_0')
    lens = kdx_(cp, rad)    
    # print(lens)
    if lens>0:
        lbs=kdx_(ps_, r=rad)
        ps = ps_[lbs>lens]
    else:
        ps = ps_
    if mNam != '':
        pds2Mod(ps, mNam=mNam)
    return ps

def kdoCt_(ps, mNam='', thr=.333, r=1):
    """使用KD树过滤点云中的离群点
    
    参数:
        ps: 输入点云数组
        mNam: 输出模型名称
        thr: 过滤阈值,默认0.333
        r: 搜索半径,默认1
        
    返回:
        过滤后的点云数组
    """
    # 转换输入为numpy数组
    ps = getArr(ps)
    
    # 获取每个点的邻域点数
    lbs = kdOlbs_(ps, r)[0]
    
    # 计算过滤阈值
    msk = np.max(lbs)*thr
    
    # 过滤离群点
    ctPs = ps[lbs>msk]
    
    # 输出为模型(如果提供名称)
    pds2Mod(ctPs, mNam=mNam)
    
    return ctPs

def cleanPj(pjPd,
            pn=None, r=.3, thr=2, 
            # cp = None, rad = 5., 
            mNam='vtCt'):
    if pn is not None:
        pjPd = getNod(pjPd)
        ps = psPj(pjPd, pn)
    ps = getArr(pjPd)
    ctPs = kdCt_(ps)
    ctPs = psLbs(ctPs, mNam=mNam)
    return ctPs

def delBdMsk_(lbs_, mxCnt=10):
    """判断区域是否接触边界并按点数筛选"""
    # 输入验证 (保持原样)
    if not isinstance(lbs_, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if lbs_.ndim != 2 or lbs_.size == 0:
        return np.array([], dtype=int)

    # 优化1：合并边界掩码生成
    bdMsk = np.zeros_like(lbs_, dtype=bool)
    bdMsk[[0, -1], :] = bdMsk[:, [0, -1]] = True  # 同时设置上下左右边界
    
    # 优化2：一步完成边界标签过滤
    msk = ~np.isin(lbs_, np.unique(lbs_[bdMsk]))  # 排除边界标签
    
    # 优化3：使用更高效的标签统计方式
    cnts = np.bincount(lbs_.ravel() * msk.ravel())  # 合并掩码到统计中
    mxLb = np.array([np.argmax(cnts[1:])+1]) # Get index of max count (excluding 0)
    
    return mxLb.astype(int)
def getInGps(   pjPs, pjNor=None,
                stp=1/3,
                mNam=''):
    gps = obGps(pjPs, 
                pjNor, 
                stp=stp, 
                mNam=sNam(mNam, 'gps'))    
    _, pjT, kdO_ = kdOlbs_(pjPs, 1.)
    inds = kdO_(gps)
    gMsk = np.where(inds > 0, 0, 1)
    bMsk = np.zeros_like(gMsk, dtype=bool)
    bMsk[[0, -1], :] = gMsk[[0, -1], :] == 1
    bMsk[:, [0, -1]] = gMsk[:, [0, -1]] == 1
    # stt = np.array([[0,1,0],
    #                 [1,1,1],
    #                 [0,1,0]], dtype=bool)
    stt = np.ones((3, 3), dtype=bool)
    bds = dila(bMsk, stt, -1, gMsk==1)
    msk = gMsk.copy()
    msk[bds] = 0
    if sp is not None:
        spId = findPs(gps, sp)[-1]
        lbs, num = scLb(self.msk)
        if num > 0:  # 确保有连通区域
            sLb = lbs[spId]
            msk = lbs==sLb
    # 使用形态学开运算去除孤立噪点
    # strel = np.array([  [0,1,0],
    #                     [1,1,1],
    #                     [0,1,0]], dtype=bool)
    # msk = ndIm.binary_opening(msk, structure=strel)
    inGps = gps[msk>0]
    pds2Mod(inGps, sNam(mNam, 'inGps'))
    ctId = pjT.query(inGps)[1]
    ctPs = pjPs[ctId]
    pds2Mod(self.ctPs, sNam(mNam, 'ctPs'))
    return inGps, ctPs   

class CtPj:
    """计算投影轮廓和内部网格点集的类
    """
    def __init__(self, pds, pjNor=None, 
                cJp=None, sp=None, clean=False, eSp=None,
                mNam='', rad=1/3, thr=None):
        """初始化类
        🔱 参数:
            pds: 点集
            gps: 网格点集，如果为None则自动生成
            pjNor: 投影法向量 
            cJp: 投影中心点
            sp: 种子点 
            mNam: 模型名称
            r: 网格半径
            thr: 阈值
        """
        # 直接使用getArr处理点集
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
        # 计算投影点集
        if pjNor is not None:
            pjNor = ndA(pjNor)
            if pjNor.ndim==2: 
                _, self.pjNor, self.sDrt = self.ras = pjNor
            else: self.pjNor = pjNor; self.ras = None
            self.pjPs_ = psPj(self.pds, 
                            (self.cJp, self.pjNor), 
                            mNam=sNam(mNam, 'pjPs_'),
                            exTyp=None
                            )
        else:
            self.pjPs_ = self.pds
            self.pjNor = psFitPla(self.pjPs_)
        if pjNor.ndim==2:
            self.sDrt = pjNor[2]
        if clean:
            self.pjPs = cleanPj(cnnEx(self.pjPs_), 
                                mNam=sNam(mNam, 'pjPs'))
        else:
            self.pjPs = getArr(cnnEx(self.pjPs_, exTyp='Lg'))
        self.gps = obGps(self.pjPs, 
                    self.ras if self.ras is not None else self.pjNor, 
                    stp=1/3, 
                    mNam=sNam(mNam, 'gps'))    
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
                if num > 0:  # 确保有连通区域
                    sLb = lbs[spId]
                    self.msk = lbs==sLb
            # 使用形态学开运算去除孤立噪点
            # strel = np.array([  [0,1,0],
            #                     [1,1,1],
            #                     [0,1,0]], dtype=bool)
            # msk = ndIm.binary_opening(msk, structure=strel)
            self.inGps = self.gps[self.msk>0]
            pds2Mod(self.inGps, sNam(mNam, 'inGps'))
            ctId = pjT.query(self.inGps)[1]
            self.ctPs = self.pjPs[ctId]
            pds2Mod(self.ctPs, sNam(mNam, 'ctPs'))
        except Exception as e:
            print(f"CtPj初始化失败: {str(e)}")
            # 设置默认值以避免后续操作出错
            self.inGps = np.array([])
            self.ctPs = np.array([])
            self.pjPs_ = self.pjPs = np.array([])
            raise        
    def mic(self):
        """计算最大内切圆
        🎁 返回: (cp, rad, mic, ctPs)
            cp: 圆心
            rad: 半径 
            mic: 内切圆点集
            ctPs: 轮廓点集
        """
        if not hasattr(self, 'inGps') or len(self.inGps) == 0:
            self.inGps = np.array([])
            
        try:
            pds2Mod(self.ctPs, mNam=sNam(self.mNam, 'ctPs'))
            pds2Mod(self.inGps, mNam=sNam(self.mNam, 'iPs_'))
            if self.eSp is not None:
                mnDt = np.dot(self.ctPs-self.eSp, -self.sDrt).min()
                if mnDt < 0:
                    pd, sPs = vtkPln((self.eSp, -self.sDrt), 
                                self.pd, pd0=True,
                                mNam=sNam(self.mNam, 'eSpln'))
                    # ePjPs = pdPj(sPs, (self.eSp, self.sDrt))
                    # ePjPs = psPj(sPs, (self.eSp, self.sDrt))[:, None] + self.sDrt*rgN_(1,.3)
                    ePjPs = psPj(sPs, (self.eSp, self.sDrt))
                    self.pds = ndA(list(nx3ps_(pd))+list(nx3ps_(ePjPs)))
                    self.ctPs = dotCut(self.ctPs, (self.eSp, - self.sDrt), thr=0)
                    self.inGps = dotCut(self.inGps, (self.eSp, - self.sDrt))
                pds2Mod(self.pds, mNam=sNam(self.mNam, 'cPds'))
            pds2Mod(self.ctPs, mNam=sNam(self.mNam, 'ctPs'))
            pds2Mod(self.inGps, mNam=sNam(self.mNam, 'iPs'))
            cp, rad, mic = psMic(self.ctPs, self.inGps, self.pjNor,
                                mNam=sNam(self.mNam, 'ctMic'))
            return cp, rad, mic, self.ctPs
        except Exception as e:
            print(f"计算最大内切圆失败: {str(e)}")
            raise

    def edPs(self):
        """计算椭圆弧
        
        Returns:
            tuple: (eSps, eIps, sDrt, vbIps, iPs)
                eSps: 上椭圆弧点集
                eIps: 下椭圆弧点集 
                sDrt: 椎体方向向量
                vbIps: 椎体内部点集
                iPs: 投影轮廓点集
        """
        try:
            # 获取内部点和边界点
            inGps, self.bdPs = ctBd_ed(self.msk, self.gps, 9)
            if len(inGps) == 0:
                raise ValueError("未找到有效内部点")
                
            pds2Mod(inGps, mNam=sNam(self.mNam, 'iPs'))
            
            # # 计算最大内切圆
            # cp, rad, _ = psMic(self.bdPs, self.pjNor, inGps, 
            #                   mNam=sNam(self.mNam, 'ctMic'))
            # 获取内部点
            # cPs, cKdx = kdQ_(inGps, self.pjPs_, self.r)
            # pds2Mod(cPs, mNam=sNam(self.mNam, 'iPs_'))
            
            # # 计算反投影点集
            # rjPs_ = self.pds[cKdx()]
            # pj_T = kdT(self.pjPs_)
            rjPs_ = self.pds[kdT(self.pjPs_).query(inGps)[1]]
            
            pds2Mod(rjPs_, mNam=sNam(self.mNam, 'rjPs_'))
            # 分终板
            eIps, eSps = psLbs(rjPs_, 2, 2)
            if len(eSps) == 0 or len(eIps) == 0:
                raise ValueError("分终板计算失败")
            # Filter out boundary points that are too far from center plane
            # Keep points within 2mm of center plane
            cp_ = findPs(eSps, self.cJp)[0]
            cp_ = cp_-self.pjNor*2 # ; addFid(cp_, sNam(self.mNam, 'cp_'))
            eSps = dotPlnX(eSps, (cp_, self.pjNor), 1.)
            pds2Mod(eSps, mNam=sNam(self.mNam, 'eSps'))
            # 计算椎体方向
            sDrt = psFitPla(eSps)
            if not any(sDrt):
                raise ValueError("椎体方向计算失败")
                
            # 投影vbPs
            vbIps = psPj(inGps, (self.cJp, sDrt))
            self.iPs = psPj(self.ctPs, (self.cJp, sDrt))
            
            # 保存反投影点集到模型
            pds2Mod(vbIps, mNam=sNam(self.mNam, 'vbIps'))
            pds2Mod(self.iPs, mNam=sNam(self.mNam, 'Ips'))
            
            return eSps, eIps, sDrt, vbIps, self.iPs
            
        except Exception as e:
            print(f"椭圆弧计算失败: {str(e)}")
            return np.array([]), np.array([]), np.array([0,0,1]), np.array([]), np.array([])

    def sCt(self):
        """计算椎体截面
        
        Returns:
            tuple: (inGps, ctPs, rjPs) 
                inGps: 内部网格点集
                ctPs: 轮廓点集
                rjPs: 反投影点集
        """
        try:
            # 保存轮廓点到模型
            pds2Mod(self.ctPs, mNam=sNam(self.mNam, 'ctPs'))
            
            # 计算反投影点集
            rjPs = self.pds[kdT(self.pjPs_).query(self.ctPs)[1]]
            pds2Mod(rjPs, mNam=sNam(self.mNam, 'rjPs'))
            return self.inGps, self.ctPs, rjPs
        except Exception as e:
            print(f"计算椎体截面失败: {str(e)}")
            return np.array([]), np.array([]), np.array([])

 
def zoom():
    """Zoom 
    """
    slNods = ut.getNodes('vtkMRMLSliceNode*')
    sl3d = slicer.app.layoutManager().threeDWidget(0).threeDView()
    for slNod in list(slNods.values()):
        slWgt = slicer.app.layoutManager().sliceWidget(
            slNod.GetLayoutName())
        slWgt.sliceLogic().FitSliceToAll()
        sl3d.resetFocalPoint()
        sl3d.resetCamera()

# import logging


def log_(text="Done",
         tim=None,
         sec=.5
         #  end='\n\n',
         ):
    ''' log: 记录日志
    '''
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


def c2s_(c, arr=False): return ndA(list(c.values()))\
    if arr else list(c.values())


def dic2Pd(dic, mNam=''):
    """字典转换为VTK PolyData
    🧮 将包含points,cells等数据的字典转换为VTK PolyData对象

    🔱 dic: 包含以下键的字典:
        - points: (n,3)数组, 点坐标
        - cells: (m,k)数组, 每行是一个单元的点索引
        - cell_data: 字典, 单元数据
        - point_data: 字典, 点数据

    🎁 vtkPolyData对象
    """
    # 1. 创建vtkPolyData对象
    pd = vtk.vtkPolyData()

    # 2. 设置点集
    if 'points' in dic:
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(dic['points']))
        pd.SetPoints(pts)

    # 3. 设置单元
    if 'cells' in dic:
        cells = dic['cells']
        if cells.ndim == 1:
            cells = cells[:, None]  # 转为2D数组

        # 创建单元数组
        ca = vtk.vtkCellArray()
        for cell in cells:
            ca.InsertNextCell(len(cell))
            for pid in cell:
                ca.InsertCellPoint(int(pid))

        # 根据单元类型设置
        if cells.shape[1] == 1:  # 点
            pd.SetVerts(ca)
        elif cells.shape[1] == 2:  # 线
            pd.SetLines(ca)
        elif cells.shape[1] == 3:  # 三角形
            pd.SetPolys(ca)
        else:  # 其他多边形
            pd.SetPolys(ca)

    # 4. 设置单元数据
    if 'cell_data' in dic:
        for k, v in dic['cell_data'].items():
            arr = numpy_to_vtk(v)
            arr.SetName(k)
            pd.GetCellData().AddArray(arr)

    # 5. 设置点数据
    if 'point_data' in dic:
        for k, v in dic['point_data'].items():
            arr = numpy_to_vtk(v)
            arr.SetName(k)
            pd.GetPointData().AddArray(arr)

    return getNod(pd, mNam)


def pd2Dic(pds):
    """将VTK PolyData转换为字典格式

    🔱 参数:
        pds: polydata对象或名称
    🎁 返回:
        包含points,cells,cell_data,point_data的字典
    """
    # 获取polydata
    pd = getPd(pds)

    # 获取points和cells
    points = vtk_to_numpy(pd.GetPoints().GetData())
    cells = np.array([[pd.GetCell(i).GetPointId(j)
                      for j in range(pd.GetCell(i).GetNumberOfPoints())]
                     for i in range(pd.GetNumberOfCells())])

    # 获取cell data和point data
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

# tag pd2Vps: vPd对齐🗞, 目的: 减小误差
def pd2Vps(vd, pdc=None, thr=80., mNam=''):
    vol = vd.vol
    arr = vd.arr
    if thr is None:
        thr = arr.mean()
        arr_ = np.where(arr < thr, 0, arr)
        vd = arr2vol(vol, arr_, rtnVd=True)
    vps = vd.ps
    vpc_ = vps.copy()
    if isinstance(vpc_  , np.ndarray):
        vpc_ = {1: vpc_}
    if pdc is None:
        pdc = vd.pd
    aPdc = {}
    for k in vpc_.keys():
        vps = vpc_[k]
        vpT = kdT(vps)
        pd = pdc[k]
        pds = getArr(pd)
        _, inds = vpT.query(pds, k=1)
        vpd = vps[inds]
        apd = pd2Dic(pd)
        apd['points'] = vpd
        apd_ = dic2Pd(apd)
        aPdc[k] = cnnEx(apd_, mNam)
    return aPdc
# pdc = pd2Vps('vt')
SIN = np.array([0.0, 0.5, 0.866, 1.0, 0.866, 0.5, 
                -0.0, -0.5, -0.866, -1.0, -0.866, -0.5])
COS = np.array([1.0, 0.866, 0.5, 0.0, -0.5, -0.866,
                -1.0, -0.866, -0.5, -0.0, 0.5, 0.866])

def vCir30(nor=NZ, rad=1., cp=OP, mNam=''):
    cp, nor = ndA(cp, nor)
    # S30 = ndA([0.0, 0.5, 0.866, 1.0, 0.866, 0.5])
    # sin_ = np.concatenate([S30, -S30])
    # cos_ = np.roll(sin_, 3)
    v, w = p2pXyz(nor, False)
    bCir_ = np.outer(COS, v) + np.outer(SIN, w)  # [sn,3]
    cir = bCir_ * rad + cp
    if mNam!='':
        ps2cFids(cir, mNam, None, 1, 1.)
    return cir, lambda r, p=cp: bCir_*r+p
# tag psMic: 点集最大内切圆
def psMic(pds, inGps=None,
            nor=None, stp=1.0, 
            mNam='', mxIt=20):
    """计算点集的最大内切圆(向量化版本)
    🧮 使用KD树和向量化计算加速搜索

    🔱 pds: 点集
       stp: 步长
       mNam: 模型名
       mxIt: 最大迭代次数

    🎁 圆心,半径,内切圆点集
    """
    # 初始化点集和KD树
    ps = getArr(pds)
    psT = kdT(ps)
    
    # 平面拟合
    if nor is None:
        nor = psFitPla(ps)
        
    else:
        nor = ndA(nor)
        if nor.shape == (2,3):
            nor, drt = nor
    # 初始化搜索点
    if inGps is None:
        gCp = ps.mean(0)
        mnDt = findPs(ps, gCp)[1]
        gps = obGps(ps, nor, flat=True)
        inGps = gps[kdOlbs_(gps, mnDt, gCp, False)[0]]
    else:
        inGps = getArr(inGps)
        if pds is None:
            # 当使用inGps作为基准时更新搜索点
            gCp = inGps.mean(0)
            mnDt = findPs(ps, gCp)[1]

    # 预计算搜索角度
    cir_ = vCir30(nor)[-1]

    def mnMx(gps):
        """找到最优点和实际半径
        🧮 计算点到边界的最小距离作为实际半径
        """
        dts = psT.query(gps, k=1)[0]
        mxId = np.argmax(dts)
        cp = gps[mxId]
        # 计算实际半径(到边界最小距离)
        rad = psT.query(cp[None], k=1)[0][0]
        return cp, rad

    # 初始最优解
    cp, rad = mnMx(inGps)
    best_cp, best_rad = cp, rad
    i = 0

    while (stp >= EPS and i < mxIt):
        # 生成候选点
        ps_ = cir_(stp, cp)
        cp_, rad_ = mnMx(ps_)

        # 计算相对改进
        rOpt = (rad_ - rad) / rad  # 半径的相对改进
        dOpt = norm(cp-cp_) / (stp * 2.0)  # 距离惩罚项

        # 更新条件
        if (dOpt <= 1.0 and  # 搜索范围约束
            rOpt > -0.05 and  # 半径约束
                rOpt/dOpt > -0.1):  # 改进效率约束
            cp, rad = cp_, rad_
            # 更新全局最优
            if rad > best_rad:
                best_cp, best_rad = cp, rad

        stp *= 0.7
        i += 1

    # 使用全局最优结果
    cp, rad = best_cp, best_rad

    # 最终验证
    actual_rad = psT.query(cp[None], k=1)[0][0]
    if abs(actual_rad - rad) > EPS:
        print(f"警告: 实际半径({actual_rad:.4f})与计算半径({rad:.4f})不匹配")
        rad = actual_rad  # 使用实际半径
    arr = vCir30(nor, rad, cp, mNam)[0]
    return cp, rad, arr


# psMic_('bSc_ctPs', mNam='mic')
def lnXpln(pn, p0, p1=None):
    """计算平面和直线的交点
    """
    p, n = ndA(pn)
    if p1 is not None:
        p0, p1 = ndA(p0, p1)
        v = uNor(p1 - p0)
    else:
        p0, v = ndA(p0)
    d = np.dot(v, n)
    if abs(d) <= 1e-8:
        raise ValueError('plnXln: The line is parallel to the plane.')
    # Calculate the distance along the line to the intersection point
    def dt(p=p): return np.dot(p - p0, n) / d
    return p0 + dt() * v, dt



def rPlnOpt(pds, pln0, sDrt, eps=1e-6, max_iter=50, mNam=''):
    """在sDrt平面上找到最佳对称切分平面
    """
    # 1. 预处理
    ps = getArr(pds)
    op, aNor = ndA(pln0)
    sDrt = uNor(sDrt)  # 单位化法向量
    # 3. 构建平面坐标系
    # 初始法向量在平面上的投影作为第一基向量
    v1 = aNor - np.dot(aNor, sDrt) * sDrt
    if np.allclose(v1, 0, atol=eps):
        # 如果投影为零,选择任意垂直向量
        v1 = np.array([1, 0, 0]) if not np.allclose(sDrt, [1, 0, 0]) \
            else np.array([0, 1, 0])
        v1 = v1 - np.dot(v1, sDrt) * sDrt
    v1 = uNor(v1)
    v2 = np.cross(sDrt, v1)  # 第二基向量

    # 4. 投影到平面坐标系
    ps2d = np.column_stack([
        np.dot(ps - op, v1),
        np.dot(ps - op, v2)
    ])
    tree = kdT(ps2d)

    def objective(theta):
        """计算对称性度量"""
        # 计算旋转后的2D法向量
        c, s = np.cos(theta), np.sin(theta)
        nor2d = ndA([c, s])

        # 计算镜像点
        dists = np.dot(ps2d, nor2d)
        mirror = ps2d - 2 * np.outer(dists, nor2d)

        # 计算Hausdorff距离
        max_dist = np.max(tree.query(mirror, k=1)[0])

        # 转回3D法向量
        nor3d = c * v1 + s * v2
        return max_dist, nor3d

    # 5. 网格搜索
    thetas = np.linspace(-np.pi/4, np.pi/4, 8)
    dists = [objective(t)[0] for t in thetas]
    best_t = thetas[np.argmin(dists)]
    best_d = min(dists)
    best_n = objective(best_t)[1]

    # 6. 局部优化
    res = sOpt(
        lambda t: objective(t[0])[0],
        [best_t],
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': eps}
    )

    # 7. 获取最优结果
    final_d, final_n = objective(res.x[0])
    if final_d < best_d:
        best_n = final_n
    if mNam:
        vtkPln((op, best_n), mNam=mNam)
    return op, best_n

def thrGrid(
    p0, p1, rad=10., tLg=None, sn=12, mNam=''  # 每圈的采样点数
):
    '''
    🧮 在圆柱体表面生成一层螺旋网格点
    '''
    p0, p1 = ndA(p0, p1)
    nor, lg, px = p2pLn(p0, p1)[1:]
    cArr, cir_ = vCir30(nor, rad)
    if tLg is None:
        tLg = rad*.3
    stmLg = lg-rad
    thrPs = px(rgN_(tLg, tLg/sn), cArr)
    stmPs = list(px(rgN_(stmLg, tLg)[:, None], thrPs)+p0)
    stmPs += [list(cArr+p0),] # ; print(ndA(stmPs).shape)
    # stmPs = nx3ps_(stmPs)
    tiPs = [cir_(rad*.7)+px(stmLg+rad*.3),]
    tiPs += [cir_(rad*.4)+px(stmLg+rad*.6),]
    tiPs += [cir_(rad*.1)+p1] # ; print(ndA(tiPs).shape)
    thrPs = ndA(stmPs+tiPs) # ; print(thrPs.shape)
    if mNam:
        ps2cFids(nx3ps_(thrPs), mNam)
    return thrPs
# thrPs = thrGrid(self.p0, self.p1, rad=self.rad, mNam='gps')    
def psXpln(pds, pln, eps=1e-6, mNam=''):
    """计算点集与平面的交点
    🔱 参数:
        pds: (n,3)点集，自动连接相邻点构成线段
        pln: (op, nor)平面参数
        eps: 容差阈值
        mNam: 输出模型名称
    🎁 返回交点坐标数组
    """
    ps = getArr(pds)
    op, nor = ndA(pln)
    nor = uNor(nor)
    
    # 计算所有点到平面的带符号距离
    dists = np.dot(ps - op, nor)
    
    # 寻找跨越平面的线段
    cross_mask = np.abs(np.diff(np.sign(dists))) > 0
    # mxId = np.argmax(cross_mask)
    # mnId = np.argmin(cross_mask)
    # return p2pLn(ps[mnId], ps[mxId], mNam=mNam)
    # 计算交点
    idx_pairs = np.where(cross_mask)[0]
    
    # 计算交点
    xps = []
    for i in idx_pairs[[0,-1]]:
        p0, p1 = ps[i], ps[i+1]
        d0, d1 = dists[i], dists[i+1]
        t = np.clip(d0 / (d0 - d1), 0, 1)
        xp = p0 + t * (p1 - p0)
        xps.append(xp)
    
    xps = np.array(xps)
    
    if mNam:
        p2pLn(xps[0], xps[1], mNam=mNam)
    return xps

rgn_ = lambda d1, d0=0, stp=.3: (
        np.linspace(d0, d1, 
                    num=int(abs((d1-d0)/stp))+1) 
            if abs(d1 - d0) > EPS else 
        None
        )[:,None]
lGps_ = lambda n, d, ps, pad=0, stp=.3: (
    (ps 
        if ps.ndim==1 else 
    ps[:,None,:])
    + rgn_(d+pad, -pad, stp)
    * n        
    )
norX_ = lambda v, n: v * np.sign(v @ n)

def pjPsCells(ps=None, siz=None, nor=None, op=None, mNam='', alpha=None):
    from scipy.spatial import Delaunay, cKDTree
    from math import sqrt
    from collections import defaultdict
    if ps is None: 
        ps = obGps(siz=siz, pad=0)
        ps2d = ps[:, :, :2]
    else:
        ps = nx3ps_(ps) 
        # pjps = ndA(ps)[:]
        if nor is None:
            nor = psFitPla(ps)
        if op is None:
            op = ps.mean(0)
        u, v = p2pXyz(nor, False)
    # 选择两个正交向量作为平面的基
    # 将投影后的点转换到平面的 2D 坐标系
    # points_2d = convert_3d_to_2d(projected_points, centroid, u, v)
        ps_ = ps-op
        ps2d = np.column_stack((np.dot(ps_, u), np.dot(ps_, v)))
    # 进行 Delaunay 三角剖分
    if alpha is None:
        tri = Delaunay(nx3ps_(ps2d,2))
        triCells = tri.simplices
    else:
        kd_tree = cKDTree(ps2d)
        # 批量获取所有满足条件的候选边
        edges = kd_tree.query_pairs(2 * alpha, output_type='set')
        
        # 转换边格式为frozenset集合
        edges = {frozenset(pair) for pair in edges}
        
        # Alpha Shapes算法过滤边
        valid_edges = []
        for edge in edges:
            i, j = sorted(edge)
            p1, p2 = ps2d[i], ps2d[j]
            dx, dy = p2 - p1
            length = sqrt(dx**2 + dy**2)
            
            if length < 2*alpha:
                mid = (p1 + p2)/2
                radius = sqrt(alpha**2 - (length/2)**2)
                normal = np.array([-dy, dx])/length
                circle_center = mid + normal*radius
                
                # 检查周围是否存在其他点在圆内
                nearby_points = kd_tree.query_ball_point(circle_center, radius+1e-6)
                if not any((np.linalg.norm(ps2d[k]-circle_center) < radius and k not in (i,j)) 
                          for k in nearby_points):
                    valid_edges.append((i, j))
        
        # 从边重建三角网格
        triCells = _edges_to_triangles(valid_edges)
    if mNam!='':
        pds2Mod(ps, mNam, refPd = {'cells': triCells},)
    return triCells


def _edges_to_triangles(edges):
    """将边列表转换为三角面片"""
    edge_dict = dfDic(list)
    for i,j in edges:
        edge_dict[i].append(j)
        edge_dict[j].append(i)
    
    triangles = set()
    for a in edge_dict:
        for b in edge_dict[a]:
            if b in edge_dict:
                for c in edge_dict[b]:
                    if c in edge_dict[a] and a != c and (a, b, c) not in triangles:
                        tri = tuple(sorted((a, b, c)))
                        triangles.add(tri)
    return np.array(list(triangles))

def pjPsGps(pjps, r=.3, mNam=''):
    """为投影点集生成网格点和拓扑结构
    
    参数:
        pjps: 投影点集
        r: 网格半径
        mNam: 模型名称
        
    返回:
        gps: 网格点集
        gCells: 网格单元
        pjGps: 投影点对应的网格点
        pjpsCells: 映射后的投影点集单元拓扑
    """
    pjps = getArr(pjps)
    # 1. 生成网格点
    gps = obGps(pjps, stp=r, mNam=sNam(mNam, 'gps'))
    
    # 2. 创建KD树并查询最近点
    _, pjT, kdO_ = kdOlbs_(pjps, r=r)
    inds = kdO_(gps)
    pjGps = gps[inds > 0]
    
    # 3. 获取网格单元的拓扑结构
    gCells = pjPsCells(pjGps, mNam=sNam(mNam, 'cells'))
    
    # 4. 获取投影点对应的网格点索引
    pjInds = pjT.query(pjGps)[1]

    # 5. 单元索引映射与拓扑验证
    # 将网格单元中的局部索引转换为全局投影点索引
    pjpsCells = pjInds[gCells.astype(np.int32)]
    
    # 验证单元拓扑完整性并去除重复单元
    pjpsCells = np.unique(pjpsCells.reshape(-1, gCells.shape[1]), axis=0)
    # if mNam!='':
    #     pds2Mod(pjGps, sNam(mNam, 'pjGps'))
    #     pds2Mod(pjpsCells, sNam(mNam, 'pjpsCells'))
    return gps, gCells, pjGps, pjpsCells

print('funEnd')
#%%
# %%
# //MARK: TEST
def cluster_points(in_gps, 
                  eps=3.0,               # 邻域半径(毫米)
                  min_samples=5,         # 最小邻域点数
                  min_cluster_size=10,   # 最小簇点数
                  max_clusters=None,     # 最大保留簇数
                  visualize=True         # 可视化结果
                  ):
    """使用DBSCAN进行三维点云分群"""
    from sklearn.cluster import DBSCAN
    
    # 转换为Nx3格式的点云数组
    points = np.asarray(in_gps).reshape(-1, 3)
    
    # 执行DBSCAN聚类
    clt = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    
    # 获取有效标签(排除噪声点)
    labels = cLbs + 1  # -1表示噪声点转为0
    
    # 按簇大小过滤
    valid_labels = []
    for lb in np.unique(labels):
        if lb == 0:  # 跳过噪声点
            continue
        mask = labels == lb
        if np.sum(mask) >= min_cluster_size:
            valid_labels.append(lb)
    
    # 按最大簇数限制
    if max_clusters and len(valid_labels) > max_clusters:
        sizes = [(lb, np.sum(labels == lb)) for lb in valid_labels]
        valid_labels = [lb for lb, _ in sorted(sizes, key=lambda x: -x[1])[:max_clusters]]
    
    return labels

#%%
class DotCrop:
    '''点集裁切类 (合并了 dotPlnX 功能) (DotCrop)
    🧮 类: 用于点集基于平面进行各种裁切操作
    '''
    def __init__(self, pds, pln,
                 cp=None, dst=0,
                 thr=(.5, -.5)):
        """初始化并计算投影距离
        🔱 参数:
            pds: 点集 (array-like)
            pln: 裁切平面定义 (tuple(op, nor), array(nor), SlicerNode)
            cp: 参考点, 用于确定法向量方向 (可选)
            dst: 距离阈值, 用于 'dist' 模式 (可选)
            thr: 交界区域阈值 (min_dist, max_dist), 用于 ctPs (可选)
        """
        self.pds_ = pds # 保留原始输入形式
        self.ps = nx3ps_(pds) # 统一处理后的点集 (n, 3)
        self.pln_ = pln # 保留原始平面定义
        self.cp = cp
        self.dst = dst
        self.thr = ndA(thr)

        if self.ps.size == 0:
            print("警告: 输入点集为空")
            self.pjs = np.array([])
            self.op = OP
            self.nor = NZ
            return

        # --- 平面定义和投影距离计算 ---
        if isinstance(pln, (tuple, list, np.ndarray)):
            pln_arr = ndA(pln)
            if len(pln_arr) == 2 and isinstance(pln_arr[0], (tuple, list, np.ndarray)):
                # 单个平面 (op, nor)
                self.op, self.nor = ndA(pln_arr[0]), ndA(pln_arr[1])
            else:
                # 只有法向量 nor
                self.nor = ndA(pln_arr)
                self.op = self.ps.mean(0) # 使用点集中心作为原点
        elif isinstance(pln, slicer.vtkMRMLMarkupsPlaneNode):
            # Slicer 平面节点
            pln_node = getNod(pln)
            self.op, self.nor = ndA(pln_node.GetOrigin()), ndA(pln_node.GetNormal())
        else:
            raise TypeError(f"不支持的平面定义类型: {type(pln)}")

        # 调整法向量方向 (如果提供了参考点 cp)
        self.op, self.nor = rePln_((self.op, self.nor), self.cp)
        # 计算投影距离
        self.pjs = (self.ps - self.op) @ self.nor
    def getIds(self, cFun):
        """根据条件函数获取布尔索引"""
        """根据条件函数或布尔索引获取点集"""
        if self.ps.size == 0:
            return np.array([])

        if ids is None:
            if cFun is None:
                ids = np.ones(len(self.ps), dtype=bool)
            else:
                ids = self.getIds(cFun)

        if isinstance(ids, (int, np.integer)):
            if 0 <= ids < len(self.ps):
                return self.ps[ids]
            raise IndexError("索引超出范围")
        elif isinstance(ids, np.ndarray) and ids.dtype == bool:
            if len(ids) != len(self.ps):
                raise ValueError("布尔索引长度与点集数量不匹配")
            return self.ps[ids]
        raise TypeError("无效的索引类型，需要整数或布尔数组")
        
    @property
    def pjs(self):
        """获取所有点到平面的投影距离"""
        return self._pjs

    @pjs.setter
    def pjs(self, value):
        self._pjs = value

    @property
    def pjx_(self, cond=None):
        """获取不同侧的点集
        """
        if callable(cond):
            # 自定义条件函数
            return self.getPs(cond)
        else:
            raise ValueError(f"无效的侧面参数: {cond}")

    @property
    def crop(self):
        """获取平面正侧的点集(不含边界)"""
        return self.getPs(lambda pjs: pjs >= 0)

    @property
    def crop_(self):
        """获取平面负侧的点集(不含边界)"""
        return self.getPs(lambda pjs: pjs <= 0)

    @property
    def cut(self):
        """获取裁切线 (在阈值内的点投影到平面)"""
        msk = (self.pjs <= self.thr[0]) & (self.pjs >= self.thr[1])
        points = self.getPs(ids=msk)
        return psPj(points, (self.op, self.nor))

    @property
    def cropEg(self):
        """获取正向裁切点集 (包含交界点)"""
        ps = self.crop
        ctPs = self.cut
        if ctPs.size > 0:
            if ps.size == 0:
                return ctPs
            return np.vstack((ps, ctPs))
        return ps

    @property
    def cropEg_(self):
        """获取负向裁切点集 (包含交界点)"""
        ps = self.crop_
        ctPs = self.cut_
        if ctPs.size > 0:
            if ps.size == 0:
                return ctPs
            return np.vstack((ps, ctPs))
        return ps

    @property
    def slip(self):
        """获取双向裁切点集 [positive_cut, negative_cut]"""
        return self.cropEg, self.cropEg_

    @property
    def dstCrop(self):
        """获取距离裁切点集 (0 < pjs < self.dst)"""
        assert self.dst is not None, "请设置裁切距离"
        return self.getPs(lambda pjs: (pjs > 0) & (pjs < self.dst))
def rayCast_t(ps, p2=None, mPd='', nor=None, plus=60, oneP=True, inOut=False):
    """射线投射检测(支持点集矢量计算)
    🧮 使用numpy广播进行射线与三角形的批量相交检测

    🔱 ps: 起点或点集 (n,3)
        p2: 终点(可选)
        mPd: 目标模型
        nor: 方向(当p2为None时使用) (3,)或(n,3)
        plus: 射线延伸长度
        oneP: 是否只返回第一个交点
        inOut: 是否返回内外状态

    🎁 交点坐标数组(n,3)或内外状态数组(n,)
    """
    # 1. 准备射线参数 (n,3)
    ps = ndA(ps)
    if ps.ndim == 1:
        ps = ps[None]
    n_rays = len(ps)

    if p2 is not None:
        p2 = ndA(p2)
        if p2.ndim == 1:
            p2 = np.tile(p2, (n_rays, 1))
        ray_dirs = p2 - ps
    else:
        nor = ndA(nor)
        if nor.ndim == 1:
            nor = np.tile(nor, (n_rays, 1))
        ray_dirs = nor

    ray_dirs = ray_dirs / norm(ray_dirs, axis=1, keepdims=True)
    pts = ps + ray_dirs * plus

    # 2. 准备三角形数据 (m,3,3)
    pd = getPd(mPd)
    points = getArr(pd)
    cells = pd2Dic(pd)['cells']
    triangles = points[cells]  # (m,3,3)

    # 3. 使用kdTree筛选潜在三角形
    bbox_min = np.minimum(ps, pts).min(0)
    bbox_max = np.maximum(ps, pts).max(0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_radius = norm(bbox_max - bbox_min) / 2 + 1e-6

    tree = kdT(points)
    potential_points = tree.query_ball_point(bbox_center, bbox_radius)

    mask = np.array([any(p in potential_points for p in cell)
                    for cell in cells])
    triangles = triangles[mask]  # (k,3,3) k为潜在三角形数

    # 4. 矢量化射线-三角形相交测试
    # 准备广播: rays(n,1,3) vs triangles(1,k,3,3)
    ps = ps[:, None, :]  # (n,1,3)
    ray_dirs = ray_dirs[:, None, :]  # (n,1,3)
    triangles = triangles[None, :, :, :]  # (1,k,3,3)

    # 计算三角形参数
    v0, v1, v2 = triangles[..., 0, :], triangles[...,
                                                 1, :], triangles[..., 2, :]  # (1,k,3)
    edge1 = v1 - v0  # (1,k,3)
    edge2 = v2 - v0  # (1,k,3)

    # Möller–Trumbore算法
    h = np.cross(ray_dirs, edge2)  # (n,k,3)
    a = np.sum(edge1 * h, axis=-1)  # (n,k)

    # 处理平行情况
    mask = np.abs(a) > 1e-8  # (n,k)
    if not np.any(mask):
        if inOut:
            return np.zeros(n_rays)
        return ndA(None) if n_rays == 1 else np.full((n_rays, 3), np.nan)

    f = 1.0 / a  # (n,k)
    s = ps - v0  # (n,k,3)
    u = np.sum(s * h, axis=-1)  # (n,k)
    u = f * u  # (n,k)

    mask &= (u >= 0.0) & (u <= 1.0)
    if not np.any(mask):
        if inOut:
            return np.zeros(n_rays)
        return ndA(None) if n_rays == 1 else np.full((n_rays, 3), np.nan)

    q = np.cross(s, edge1)  # (n,k,3)
    v = np.sum(ray_dirs * q, axis=-1)  # (n,k)
    v = f * v  # (n,k)

    mask &= (v >= 0.0) & (u + v <= 1.0)
    if not np.any(mask):
        if inOut:
            return np.zeros(n_rays)
        return ndA(None) if n_rays == 1 else np.full((n_rays, 3), np.nan)

    t = np.sum(edge2 * q, axis=-1)  # (n,k)
    t = f * t  # (n,k)
    mask &= (t > 0.0) & (t < plus)

    # 5. 计算交点
    if inOut:
        results = np.any(mask, axis=1).astype(int)
    else:
        intersections = ps + ray_dirs * t[..., None]  # (n,k,3)
        intersections = np.where(mask[..., None], intersections, np.nan)

        if oneP:
            # 获取每条射线的最近交点
            dists = np.where(mask, t, np.inf)  # (n,k)
            nearest_idx = np.argmin(dists, axis=1)  # (n,)
            results = np.take_along_axis(
                intersections, nearest_idx[:, None, None], axis=1)[:, 0, :]
        else:
            # 返回所有有效交点
            results = [intersections[i][mask[i]] for i in range(n_rays)]

    return results

def gridBxcells(cns, mNam='gridBox', nDiv=(10, 10)):
    """为平面矩形边界框生成网格并包含三角面片
    
    参数:
        cns: 边界框顶点坐标，shape(4,3)的numpy数组，表示平面矩形的四个角点
        mNam: 网格模型名称
        nDiv: 网格分割数 (nx,ny)，默认为(10,10)
        
    返回:
        包含网格和三角面片的模型
    """

    # 验证输入
    if not isinstance(cns, np.ndarray) or cns.shape != (4, 3):
        raise ValueError("cns必须是shape为(4,3)的numpy数组")
    
    # 计算边向量 - 使用矢量计算
    v1 = cns[1] - cns[0]  # 第一条边向量
    v2 = cns[3] - cns[0]  # 第二条边向量
    
    # 计算法向量
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0, 0, 1])
    
    # 创建点和单元
    nx, ny = nDiv
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    
    # 使用矢量化操作生成所有网格点的参数化坐标
    u = np.linspace(0, 1, nx+1)
    v = np.linspace(0, 1, ny+1)
    U, V = np.meshgrid(u, v)
    
    # 将网格展平为一维数组
    U_flat = U.flatten()
    V_flat = V.flatten()
    
    # 计算所有点的3D坐标 - 使用矢量化操作
    points_array = cns[0].reshape(1, 3) + \
                  np.outer(U_flat, v1) + \
                  np.outer(V_flat, v2)
    
    # 创建点ID映射
    point_ids = np.arange((nx+1) * (ny+1)).reshape(ny+1, nx+1)
    
    # 将所有点添加到VTK点集合
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(points_array.shape[0])
    for i in range(points_array.shape[0]):
        vtk_points.SetPoint(i, points_array[i])
    
    # 使用矢量化操作生成所有三角形的顶点索引
    i_indices, j_indices = np.meshgrid(np.arange(nx), np.arange(ny))
    i_indices = i_indices.flatten()
    j_indices = j_indices.flatten()
    
    # 计算每个四边形的四个顶点索引
    p00_indices = point_ids[j_indices, i_indices]
    p10_indices = point_ids[j_indices, i_indices + 1]
    p11_indices = point_ids[j_indices + 1, i_indices + 1]
    p01_indices = point_ids[j_indices + 1, i_indices]
    
    # 创建所有三角形
    for quad_idx in range(len(i_indices)):
        p00 = p00_indices[quad_idx]
        p10 = p10_indices[quad_idx]
        p11 = p11_indices[quad_idx]
        p01 = p01_indices[quad_idx]
        
        # 第一个三角形
        triangle1 = vtk.vtkTriangle()
        triangle1.GetPointIds().SetId(0, p00)
        triangle1.GetPointIds().SetId(1, p10)
        triangle1.GetPointIds().SetId(2, p11)
        cells.InsertNextCell(triangle1)
        
        # 第二个三角形
        triangle2 = vtk.vtkTriangle()
        triangle2.GetPointIds().SetId(0, p00)
        triangle2.GetPointIds().SetId(1, p11)
        triangle2.GetPointIds().SetId(2, p01)
        cells.InsertNextCell(triangle2)
    
    # 创建多边形数据
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtk_points)
    polyData.SetPolys(cells)
    
    # 计算法向量
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polyData)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.ConsistencyOn()
    normals.Update()
    
    # 获取带法向量的结果
    result = normals.GetOutput()
    
    # 添加到场景
    try:
        getPd(result, mNam)
    except NameError:
        print(f"警告: addPolyData函数未定义，无法添加到场景，但已创建网格数据")
    
    return result
# 
# 
# p.load('/Users/liguimei/Documents/PTP/paper0/util/allCases0123.npz', allow_pickle=True)
# //MARK: END

#%%
