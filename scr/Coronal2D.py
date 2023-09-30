# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# # 框截脊柱正位
# - 框截脊柱
#%%
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from ipywidgets import interact
from scipy import ndimage as ndIm
from scipy.ndimage import label
import shelve
from tqdm import tqdm
import argparse
from typing import Literal as Lit
from typing import List as Lst
from typing import Sequence as Seq
from typing import Union as Un
from typing import Optional as Opt
from typing import List
import typing
from itertools import combinations as ittCom
ndA = np.asanyarray
#%%
# workDir = f'/Users/liguimei/Documents/GitHub/SAMed/2Dto3D/'
import os
os.chdir('/Users/liguimei/Documents/GitHub/SAMed/D22D3/')
import skLogics as sk
# output = workDir + f'/output'
# img5 = sk.img518
# img7 = sk.img7
# msk5 = sk.msk5
# msk7 = sk.msk7
# bImg5 = sk.bImg5
# bImg7 = sk.bImg7
# bBx5 = sk.bBx5
# bBx7 = sk.bBx7
bBx0 = sk.bBx5
import pyvista as pv
# from pyvista import examples
from totalsegmentator.python_api import totalsegmentator as ttSeg
# vol = examples.download_brain()

from itkwidgets import view

#%%
#%%
class CutCoronalMax:
  def __init__(self, im):
    self.im     = sk.isImg255(im)
    self.imArr  = sk.isNdaU8(im)

    self.bImg, self.bBx0, _\
                = sk.getBoneSeg(self.im)
    self.bBx    = np.copy(self.bBx0)
    self.lbMin  = 15 # T7 下缘
    zwArr       = np.max(self.imArr, axis=1)
    self.zwArr  = sk.isMsk01(zwArr)
    self.midCut(show = 1)
    # self.gPs    = self.getTlGps(self.cArrZ)
    # self.cutSpTL()
    # self.midPs, (self.lLn, self.rLn), self.mks, xys, (self.pls, self.prs) = mkXyMps(self.tArr, show=1)
    # self.bxx = bBxTvb(self.midPs, msk=self.tArr, show=1)
    # [self.p0, self.p1], self.plrs, self.ndaC, self.mkxy, self.bBxx = cutCut(self.tArr, (self.rLn-self.lLn), self.midPs, self.bBoxs[0], show=1)

  def upBbx(self, bBx, bBx0):
    ''' 更新bBx
    参: bBx : 新的bBx
      bBx0: 原始bBx
    返: bBx : 更新后的bBx
    '''
    bBx  = ndA(bBx)
    bBx0 = ndA(bBx0)
    return ndA([bBx0[0]+bBx[0],
          bBx0[1]-bBx[1]])

  def midCut(self, show = False):
    ''' 中切
    正位最大像素值压缩, 取中线(正位质心[Y])
    返回: 中位正位和3d图像
    '''
    zp , yp     = self.zwArr.shape
    cLin        = int(np.mean(
                      np.nonzero(self.zwArr),
                      axis=1)[1])       # 正位中线
    lxLn        = int(cLin-cLin/2)
    rxLn        = int(cLin+cLin/2)
    xyzs        = ndA([ [lxLn, yp-cLin, 0 ],
                        [rxLn, yp     , zp]])
    if show:
        print('show')
    cIm, bBx    = sk.cutBbx(self.im, xyzs) # 中切<2>图像
    self.cArr   = sk.isNdaU8(cIm)
    self.cArrZ  = np.max(self.cArr, axis=1)
    # self.bBx    = self.upBbx(bBx, self.bBx0)
    print(f'{self.bBx=}')
    if show:
      zwShow(     self.zwArr,
                  yLns = [self.zwArr.shape[1],
                          lxLn, rxLn, cLin]) # , bxs=bBx)
    return
  xVys = lambda ys, x: np.vstack([[x]*len(ys),ys]).T
  def getGaps(self):
    ''' 骶肋点肋间点
    参: mArr     : 正位图
    返: (lP1, rP1): 左右骶肋点
      (ps0, ps1): 左右肋间点
    '''
    mArr_     = 1-self.cArrZ # 初始化
    xLen      = self.cArrZ.shape[1]
    pLs, ps0, ps1\
              = [], [], []
    for y    in range(xLen): # 遍历每一列
      yLine   = mArr_[:, y:y+1]
      print(f'{yLine.shape=}')
      yy      = self.lbMkBxs(yLine)[1]
      pLs    += np.diff(yy)[:,0].tolist()
      ps0    += xVys(yy[:,0], y).tolist()
      ps1    += xVys(yy[:,1], y).tolist()
      if y   == xLen//2:
        pId   = len(pLs)
        idx   = np.argmax(pLs)
        self.lP1\
              = ps1[idx]
        pLs   = []
        print(len(pLs))
    self.rP1  = ps1[np.argmax(pLs) + pId]
    return      ndA([ps0, ps1])

  def getTlGps(self,
      show     = False):
    ''' 肋骶线及肋骨上下点
    参: bImg     : 正位图
      lb       : 标签
      show     : 是否显示
    返: (lGpsLb, rGpsLb): 左右肋骨上下点
      (lP1, rP1)      : 左右骶肋点
      lb1St           : 肋骨起始标签
    '''
    lb          = (self.lbMin*2+1)-20
    ps          = getGaps()
    minInd      = min(self.lP1[1], self.rP1[1])
    gPs         = [ p for p in ps              # 分上下
            if p[1] < minInd]\
          + [self.lP1, self.rP1]
    gPs         = ndA(gPs)
    lGps        = gPs[gPs[:,0] == self.lP1[0]] # 定左右
    rGps        = gPs[gPs[:,0] == self.rP1[0]]
    self.lps    = abs(np.sort(-lGps,
              axis=0))[:(lb+1)][::-1]
    self.rps    = abs(np.sort(-rGps,
              axis=0))[:(lb+1)][::-1]
    assert len(self.lps) == len(self.rps), '左右点数不等'
    if show:
      zwShow(self.zwArr,
        ps   = (self.lps, self.rps),
        yLns = (self.rP1[0],self.lP1[0]),
        pLs  = (self.lps[-2:], self.rps[-2:]),)
    return


  def cutSpTL(self,
      im3D    = None):
    lVbDn       = max(self.lps[-1][1], # 腰椎下线
            self.rps[-1][1]) + 30
    tVbUp       = min(self.lps[0][1],  # 胸椎上线
            self.rps[0][1])
    tlVb        = max(self.lps[-2][1], # 胸椎下线
            self.rps[-2][1])
    lVbUp       = tlVb - 30            # 腰椎上线

    self.tArr   = self.cArrZ[tVbUp:tlVb]     # 2D分胸椎
    self.lArr   = self.cArrZ[lVbUp:lVbDn]    # 2D分腰椎
    (x0, y0, z0),\
    (x1, y1, _) = self.bBx
    self.tBbx   = ndA([ x0, y0, tVbUp + z0,
              x1, y1, tlVb  + z0])
    self.lBbx   = ndA([ x0, y0, lVbUp + z0,
              x1, y1, lVbDn + z0])
    if  im3D   is not None:
      im3D    = sk.isImg255(im3D)
      self.tCnn, self.tbbx\
          = sk.cutBbx(im3D, self.tBbx)
      self.lCnn, self.lbbx\
          = sk.cutBbx(im3D, self.lBbx)
    self.lps    = self.lps - [0, tVbUp]
    self.rps    = self.rps - [0, tVbUp]
    zwShow(       (self.tArr, self.lArr),
        ps  = (self.lps, self.rps))
    return

  def mkXyMps(self,
        show    = False):

    nda_            = 1 - self.tArr        # 反像
    lrSum           = np.sum(nda_, axis=0)
    lrInd           = np.where(lrSum == 0)[0]
    lLn, rLn        = lrInd[0], lrInd[-1]  # 空白界
    mLn             = (lLn + rLn)/2        # 中间
    lDif            = rLn - lLn
    lLln, rLln      = ( lLn - lDif,
              rLn + lDif)
    pd_Nda          =np.pad(nda_, 1, 'constant',
              constant_values=0)
    msks, xys       = getLbXy(pd_Nda)           # 分肋间
    pls             = []
    prs             = []
    for i, xy in enumerate(xys): # 遍历每个框
      x, x1       = xy[1], xy[3]
      mEg         = [x, x1]\
              [(x < mLn) *1]    # 左右内侧
      if lLln < mEg < rLln:         # 限制在内
        if mEg == x:                      # 左侧
          pj  = msks[i][:, mEg:mEg+2]   # 截图
          pi  = np.where(pj!=0)[0].max()    # 截图内点集
          prs+= [[mEg,pi],]
        else:
          pj  = msks[i][:, mEg-2:mEg]
          pi  = np.where(pj!=0)[0].max()
          pls+= [[mEg,pi],]
    mPs             = ndA(pls + prs)
    if show:
      zwShow(nda_,
        ps  = mPs,
        yLns = (lLn, rLn, lLln, rLln),
        bxs = xys
            )
    return mPs, (lLn, rLn), msks, xys, (pls, prs)

  def lbMkBxs(self,
        imArr):
    ''' 标签分割
    参: imArr: 二值图
    返: mks  : 标签集
      bxs  : 框集
    '''
    imArr    = sk.isMsk01(imArr)
    lbs      = label(imArr)[0]
    objs     = ndIm.find_objects(lbs)
    mks, bxs = [], []
    for obj in objs:
      mk     = imArr[obj]
      if np.any(mk):
        mks += [mk]
        bxs += [ndA([ [ obj[0].start,
                        obj[0].stop  ],
                      [ obj[1].start,
                        obj[1].stop  ]]),]
    if imArr.shape[-1]\
            == 1:
      bxs    = abs(np.diff(
                      bxs, axis=1
                      ).reshape(-1, 2))
    return     mks, bxs

#%%

#ac  <a id='_itrPlot'></a>
def _itrPlot()  :
  """Create an interactive window for using widgets."""
  plot = pv.Plotter(
      off_screen  = False,
      window_size = (1024, 768),
      notebook    = False,
      lighting    = "light_kit"
      )
  plot.background_color = "white"
  plot.add_camera_orientation_widget()
  return plot

#ac  <a id='itrSlice'></a>
def itrSlice(
    mesh  : pv.PolyData,
    method: Lit["axis",
            "orthogonal"]
        = "axis",
    axis  : Lit["x", "y", "z"]
        = "x"
    ):
  p = _itrPlot()
  p.add_volume(mesh, cmap="bone",
          opacity="sigmoid")
  if method == "axis":
    p.add_mesh_slice(
      mesh,
      normal=axis,
      tubing=True,
      widget_color="black",
      line_width=3.0,
    )
  else:
    p.add_mesh_slice_orthogonal(
      mesh,
      tubing=True,
      widget_color="black",
      line_width=3.0,
    )
  p.show(cpos="iso")

#ac  <a id='pvShow'></a>
def pvShow( skImgs,
      nNam  : str   = '',
      slice : bool  = False,
      msk   : bool  = False
      )->pv.MultiBlock:
  if nNam   == '':
    nNam   = 'temp'
  meshs      = pv.MultiBlock()
  if isinstance(skImgs, (sitk.Image or np.ndarray)):
    skImgs = [skImgs]
  for i , im in enumerate(skImgs):
    im     = sk.isImg255(im)
    fNam   = nNam + str(i) + '.nrrd'
    sitk.WriteImage(im, fNam)
    reader = pv.get_reader(fNam)
    mesh   = reader.read()
    if msk:
      mesh = mesh.threshold(0.5)
    meshs.append(mesh)
  if slice:
    meshs  = meshs.slice_orthogonal()
  meshs.plot(cmap='gray_r', background='black')
  # mesh.plot(volume=True, cmap="bone")
  return meshs
#%%
#ac  <a id='maxPj'></a>
def maxPj(imArr, axis=1):
  imArr = sk.isNdaU8(imArr)
  if imArr.ndim != 3:
    raise('必须是3D图')
  return np.max(imArr, axis=axis)

def upBbx(bBx, bBx0=sk.bBx5):
  ''' 更新bBx
  参: bBx : 新的bBx
  返: bBx : 更新后的bBx
  '''
  return bBx + bBx0[0]
#ac  <a id='midCut'></a>
def midCut(img,
    show = False):
  ''' 中切
  正位最大像素值压缩, 取中线(正位质心[Y])
  返回: 中位正位和3d图像
  '''
  # bBx0    = sk.getBbox(img)
  # print(f'{bBx0=}')
  # bImg, bBx1, _\
  #         = sk.getBoneSeg(img)
  # print(f'{bBx1=}')
  imArr   = sk.isNdaU8(img)
  zp, yp  = imArr.shape[:2]
  zwArr   = np.max(imArr, axis=1)
  cLn     = int(np.mean(
                np.nonzero(zwArr),
                axis=1)[1]) # 正位中线
  lxLn    = int(cLn-cLn/2)
  rxLn    = int(cLn+cLn/2)
  xyzs    = ndA([ [lxLn, yp-cLn, 0 ],
                  [rxLn, yp    , zp]])
  cIm, bBx     \
          = sk.cutBbx(img, xyzs)
  print(f'{bBx=}')
  cArr    = sk.isNdaU8(cIm)
  cArrZ   = np.max(cArr, axis=1)
  bBox    = upBbx(bBx)
  if show:
    zwShow( cArrZ,
      yLns=[lxLn-cLn*.5, rxLn-cLn*.5, cLn*.5]) #, bxs=[bBx])
  return    cArrZ, cArr, (lxLn, rxLn), bBx, bBox
#ac  <a id='zwShow'></a>
def zwShow(
    imgs  = None,
    ps   = None,
    pLs  = None,
    yLns  = None,
    xLns  = None,
    bxs  = None,
    dim  = 2
    ):
  def __tpl2Ls(data, dim=2):
    # if not isinstance(data, seq):
    #     data = [data]
    datas = []
    for d in data:
      if isinstance(d, np.ndarray):
        datas += ndA(d).reshape(-1,dim).tolist()
    return ndA(datas)

  if imgs is not None:
    if isinstance(imgs, sitk.Image):
      imgs = sitk.GetArrayFromImage(imgs)
  if isinstance(imgs, np.ndarray):
    imgs = [imgs]
  exY, exX = imgs[0].shape
  for i, img in enumerate(imgs):
    if i == 0:
      alfa = 1
    else:
      alfa = 0.3
    plt.imshow(img, cmap= 'gray', alpha=alfa)
  # 绘制点
  if ps is not None:
    ps  = __tpl2Ls(ps)
    if imgs is None:
      exX = np.max(ps, axis=0)
      exY = np.max(ps, axis=1)
    # for ps in pss:
    # for p in ps:
    if ps.shape[1] == 1:
      plt.plot(ps[0], ps[1], 'r.')
    else:
      plt.plot(ps[:,0], ps[:,1], 'b.')
  # 绘制纵线
  if yLns is not None:
    # yLns = __tpl2Ls(yLns, 1)
    for yl in yLns:
      plt.vlines(yl, 0, exY, 'g', 'dashed')
  # 绘制横线
  if xLns is not None:
    # xLns = __tpl2Ls(xLns, 1)
    for xl in xLns:
      plt.hlines(xl, 0, exX, 'y', 'dashed')
  # 绘制线
  if pLs is not None:
    pLs = __tpl2Ls(pLs)
    for i in range(0, len(pLs), 2):
      plt.plot(pLs[i:i+2][:,0],pLs[i:i+2][:,1], 'r')
  # 绘制矩形
  if bxs is not None:
    # bxs = __tpl2Ls(bxs, dim=4)
    def __getBxs(bx, ax):
      (y, x),(y1, x1) = bx
      ax.add_patch(plt.Rectangle((x,y),
      x1-x, y1-y, edgecolor='r',
      facecolor=(0,0,0,0), lw=1.5))
    for bx in bxs:
      __getBxs(bx, plt.gca())
  plt.axis('off')
  plt.show()

p2LnDst = lambda p, b, k:\
      abs(b * p[0] - p[1] + k)\
      /   np.sqrt(b ** 2 + 1)

#ac  <a id='xyLin'></a>
def xyLin(p0,
          p1   = None,
          drt  = None,
          xy   = [0, 0]
                  ):
  ''' 线性方程: y = kx + b '''
  x0, y0       = p0
  if p1       is not None:
    drt        = p1 - p0
  if drt[0]   == 0:
    return       np.array([x0, xy[1]])
  k            = drt[1] / drt[0]
  b            = y0 - k * x0
  line         = np.poly1d([k, b])
  if (xy[0] & xy[1])\
              is None: # 若没有x, y, 则返回k, b
    return       k, b
  elif xy[0]  is not None: # 若只有x, 则返回y
    return       ndA([xy[0], line(xy[0])])
  elif xy[1]  is not None: # 若只有y, 则返回x
    return       ndA([(xy[1] - b) / k, xy[1]])
  else: # 若有x, y, 则返回垂足
    d = np.abs(k*x0 - y0 + b)\
      / np.sqrt(k**2 + 1)
    x = (k*x0 - k*d - y0 + b)\
      / (k**2 + 1)
    y = k*x + b
    return       ndA([x,y])
#ac  <a id='dectPs'></a>
def dectPs(imArr, xy, show=True):
  ''' 纵横侦测交点 '''
  imArr          = sk.isNdaU8(imArr)
  xy            *= 1
  bN             = (xy[1] == 0) * 1
  if xy[1]      == 0:
    d          = 1
    mEg        = xy[0]
    cMk        = imArr[:, mEg-1:mEg+1]
    cys        = np.nonzero(cMk)[0]
    arr        = np.column_stack(( # 横向堆叠, 生成点集
            np.full_like(cys, mEg),
            cys))
  elif xy[0]    == 0: # [TODO: 不知为何筛选不理想]
    d          = 0
    mEg        = xy[1]
    cMk        = imArr[mEg-1:mEg+1, :]
    cys        = np.nonzero(cMk)[1]
    arr        = np.column_stack((
            cys,
            np.full_like(cys, mEg)))
  diff           = np.diff(arr[:, d])
  keep           = np.insert(diff > 2,
            0, 1)
  ps             = arr[keep]
  if show:
    zwShow([imArr,cMk], ps=ps)
  return ps

#ac  <a id='linSecCont'></a>
def linSecCont(cArr, p0, p1,
    show    = 0,
    end     = True):
  ''' 直线与轮廓相交
  参: cArr: 轮廓数组
    p0, p1: 直线起点, 终点
    show: 是否显示
  返: secPs: 相交点集
  '''
  def __kp1StEnd(arr,
      k   = 2):
    # 过滤相邻元素之间距离小于 2 的元素
    arr     = ndA(arr)
    shp     = arr.shape[0]
    if shp  < 2:
      dif = np.diff(arr[:, 0])
    else:
      dif = np.diff(arr[:, 1])
    # kpIds   = [0, [0, -1]][end*1]
    keep    = np.insert((dif > k)*1,
          0, 1)

    return    arr[keep[np.where(keep==1)]]
  lnArr       = linImg(cArr, p0, p1)
  # 相交图
  secArr      = cArr\
        * lnArr
  # 提取相交点
  y, x        = np.where(secArr)
  secPs       = [[x[i], y[i]]
          for i
          in range(len(x))]
  # 过滤相邻元素之间距离小于 2 的元素
  secPs       = __kp1StEnd(secPs)
  if show:
    zwShow([cArr, lnArr],
      ps  = [secPs])
  return        secPs


#ac  <a id='plnCrop'></a>
def plnCrop(im3D, pp, pn):
  """ 平面裁切
  参: im3D(sitk.Image): 3D图像
    pp(ndA)         : 平面上一点
    pn(ndA)         : 平面法向量或通过pp的法线上的一点
  返: cropped_image(sitk.Image): 裁切后的图像
  """
  pp  = ndA(pp)
  pn  = ndA(pn)
  norm= (pp - pn)/np.linalg.norm(pp - pn)
  img = sk.isImg255(im3D)

  size = ndA(img.GetSize())
  spc  = ndA(img.GetSpacing())

  x, y, z = np.meshgrid(          # 计算像素到平面的距离 生成网格
        np.arange(size[0]),
        np.arange(size[1]),
        np.arange(size[2]),
        indexing='ij')
  coords = np.stack((x, y, z),    #   生成坐标
        axis=-1)
  dsts = np.abs(np.dot(           #   计算距离
        coords - pp, norm))

  min_dst = np.min(dsts)
  max_dst = np.max(dsts)

  siz = np.ceil(        (max_dst\
            -  min_dst)\
            / spc).astype(int)
  p0 = pp - pn * min_dst - 0.5 * (siz - 1) * spc
  # roi = sitk.ImageRegion(siz.tolist(), p0.tolist())
  roi = sitk.Image(size.tolist(), sitk.sitkUInt8)
  roi.SetOrigin(p0.tolist())
  imCrop = sitk.RegionOfInterest(img, roi)
  return imCrop


#ac  <a id='linImg'></a>
def linImg(cArr, p0, p1):
  ''' 生成直线图
  参: cArr: 轮廓数组
    p0, p1: 直线起点, 终点
  返: lnArr: 直线图
  '''
  xShp, yShp  = cArr.shape
  y0, x0      = p0
  y1, x1      = p1
  dst         = sk.p2p_dst(ndA(p0),
          ndA(p1)).astype(int)
  # 生成间隔为1的整数序列
  if x0      != x1:
    xx      = np.linspace(x0, x1,
          num = dst)
    yy      = np.linspace(y0, y1,
          num = dst)
    lin     = np.vstack([yy, xx]).T
  else:
    lin     = np.linspace(ndA(p0), ndA(p1),
          num = int(abs(y1 - y0)))
    # lin必须在cArr内
  lin         = lin[lin[:, 0] < yShp]
  lin         = lin[lin[:, 1] < xShp]
  # 生成line的罩和原图大小一致
  lnArr       = np.zeros_like(cArr)
  indx, indy  = ( lin[:, 1].astype(int), #
          lin[:, 0].astype(int))
  lnArr[indx,
    indy]   = 1
  # if plus:
  #     lnArr  += cArr
  return        lnArr

#ac  <a id='plaImg3D'></a>
def plaImg3D(arr3D, pp, pN, pix=0):
  ''' 在arr3D里面加一个平面,像素为pix
  参: arr3D: 3D数组
    pp: 平面上一点
    pnorm: 平面法向量
    pix: 像素值
  返: arr3D: 添加平面后的3D数组
  '''
  xShp, yShp, zShp  = arr3D.shape
  xx, yy            = np.meshgrid(
              np.arange(yShp),
              np.arange(xShp))
  zz                = (-pN[0]*xx - pN[1]*yy - pp.dot(pN)) / pN[2]
  zz                = np.clip(zz, 0, zShp-1).astype(int)
  arr3D[xx, yy, zz] = pix
  return arr3D , ndA([xx, yy, zz])


xVys = lambda ys, x: np.vstack([[x]*len(ys),ys]).T

#ac  <a id='getGaps'></a>
def getGaps(mArr):
  ''' 骶肋点肋间点
  参: mArr     : 正位图
  返: (lP1, rP1): 左右骶肋点
    (ps0, ps1): 左右肋间点
  '''
  mArr_     = 1 - sk.isMsk01(mArr) # 初始化
  sizX      = mArr.shape[1]
  pLs, ps0, ps1\
        = [], [], []
  for y in range(sizX):             # 遍历每一列
    yLine   = mArr_[:, y: y+1]
    # print(f'{yLine.shape=}')
    yy      = lbMkBxs(yLine)
    print(f'{yy.shape=}')
    pLs    += np.diff(yy)[:,0].tolist()
    ps0    += xVys(yy[:,0], y).tolist()
    ps1    += xVys(yy[:,1], y).tolist()
    if y   == sizX//2:
      pId   = len(pLs)
      idx   = np.argmax(pLs)
      lP1   = ps1[idx]
      pLs   = []
  print(len(pLs))
  rP1       = ps1[np.argmax(pLs) + pId]
  return      (lP1, rP1), (ps0, ps1)



#ac  <a id='getTlGps'></a>
def getTlGps(zwArr,
    lb: int   = 11,
    show = False):
  ''' 肋骶线及肋骨上下点
  参: bImg     : 正位图
    lb       : 标签
    show     : 是否显示
  返: (lGpsLb, rGpsLb): 左右肋骨上下点
    (lP1, rP1)      : 左右骶肋点
    lb1St           : 肋骨起始标签
  '''
  zwArr   = sk.isMsk01(zwArr)
  assert zwArr.ndim == 2, '必须是2D图像'
  assert 38 > lb > 0, 'lb必须在2-38之间'
  (lP1, rP1), (p0s, p1s)\
      = getGaps(zwArr)
  ps      = ndA(p0s + p1s)
  minInd  = min(lP1[1], rP1[1])
  gPs     = [                # 分上下
        p for p in ps
        if p[1] < minInd]\
      + [lP1, rP1]
  gPs     = ndA(gPs)
  lGps    = gPs[gPs[:,0] == lP1[0]] # 定左右
  rGps    = gPs[gPs[:,0] == rP1[0]]

  lps  = abs(np.sort(-lGps,
        axis=0))[:(lb+1)][::-1]
  rps  = abs(np.sort(-rGps,
        axis=0))[:(lb+1)][::-1]
  lLen, rLen\
      = len(lps), len(rps)
  if lLen != rLen:
    gLen = min(lLen, rLen)
    lps  = lps[:gLen]
    rps  = rps[:gLen]
  else:
    gLen = lLen
  lb1St    =  20 - (gLen-1)//2
  # print(f'{gLen=}: From {sk.LBLVS[lb1St]} to {sk.LBLVS[19]}')
  if show:
    zwShow(zwArr,
      ps   = (lps, rps),
      yLns = (rP1[0],lP1[0]),
      pLs  = (lps[-2:], rps[-2:]),)
  return         (lps, rps)


#ac  <a id='getLbXy'></a>
def getLbXy(imArr,
      xy      = True,
      crop    = False
      ):
  ''' 分标签, 提罩框
  参: imArr     : 2D图像
    xy        : 是否返回坐标
    crop      : 是否裁切
  返: mkArrSt   : 标签罩
    xyArrSt   : 标签框
  '''
  # def __2dLbXy(xys, siz):
  #   xyArr       = ndA(xys)
  #   print(f'{xyArr.shape=}')
  #   xyArr, ind2 = ySort(siz, xyArr)
  #   return xyArr[ind2]

  # def ySort(siz, xyArr):
  #   inds        = np.argsort(xyArr[:, 1])
  #   xyArr       = xyArr[inds]
  #   df          = np.diff(xyArr)[:,0][1:]
  #   dfInd       = ((df<siz*.8)&(df>2))
  #   ind2        = np.insert(dfInd, 0, True)
  #   return xyArr,ind2
  imArr           = sk.isMsk01(imArr)
  sizX            = imArr.shape[0]
  lbs, nums       = label(imArr)
  print(            f'{nums=}')
  xys, mks        = [], []
  for i in range(1, nums+1):
    mk          = (lbs == i)*1
    x, y        = np.where(mk)
    if  len(x) == 0\
    |   len(y) == 0:
      print(f'Label {i} is empty')
      continue
    if mk.shape[1]<2:  # 1D分线
      xys    += [[x.min(), x.max()],]
    else:              # 2D分框
      x0, y0  = x.min(), y.min()
      x1, y1  = x.max(), y.max()
      xys    += [ndA([    [y0, x0],
                [y1, x1]]),]
      mks    += [mk,]
    if crop:
      mks    += [[mk[x0:x1, y0:y1]],]
  if mks         == []:
    print(        ndA(xys).shape)
    return        __2dLbXy(xys,sizX)
  xyArr           = ndA(xys)
  mkArr           = ndA(mks)
  inds            = np.argsort( # 按y排序
                        xyArr[:, [0, 3][xy*1]])
  xyArr           = xyArr[inds]
  mkArr           = mkArr[inds]
  dfY             = np.diff(xyArr[:, 1])
  ind2            = np.insert(dfY > 2, 0, 1)  # 过滤距离小于2的元素
  xyArr           = xyArr[ind2]
  mkArr           = mkArr[ind2]
  return            mkArr, xyArr

#ac  <a id='getContour'></a>
def getContour(img):
  img  = sk.isImg01(img)
  cImg = sitk.LabelContour(img,
        fullyConnected=True)
  return sitk.GetArrayFromImage(cImg)

# 裁切胸腰
#ac  <a id='cutSpTL'></a>
def cutSpTL(
      im2D,
      gps,
      bBx,
      im3D  = None
          ):
  ''' 裁切胸腰
  参: im2D     : 2D图像
    gps      : 胸腰椎质心
    im3D     : 3D图像
  返: (tVbZw, lVbZw): 胸腰椎图像
    (lps, rps)    : 左右分割线
    (tVb3D, lVb3D): 胸腰椎3D图像
    (tBbx, lBbx)  : 胸腰椎3D框
  '''
  im2D    = sk.isMsk01(im2D)
  lps, rps      \
          = gps
  lVbDn   = max(  lps[-1][1],
          rps[-1][1])\
      + 30
  tVbUp   = min(  lps[ 0][1],
          rps[ 0][1])
  tlVb    = max(  lps[-2][1],
          rps[-2][1])
  lVbUp   = tlVb - 30
  tVbZw   = im2D[tVbUp:tlVb]
  tVb3D   = im3D[tVbUp:tlVb]
  lVbZw   = im2D[lVbUp:lVbDn]
  lVb3D   = im3D[lVbUp:lVbDn]
  tBx       = bBx.copy()
  tBx[0][2] = tVbUp
  tBx[1][2] = tlVb
  # tBox      = upBbx(tBx, bBox)
  lBx       = bBx.copy()
  lBx[0][2] = lVbUp
  lBx[1][2] = lVbDn
  # lBox      = upBbx(tBx, bBox)
  lps     = lps - [0, tVbUp]
  rps     = rps - [0, tVbUp]
  zwShow((tVbZw, lVbZw), ps=(lps, rps))
  return  (tVbZw, lVbZw), (lps , rps),\
      (tVb3D, lVb3D), (tBx, lBx)


#ac  <a id='getMidSpc'></a>
def getMidSpc(img, Sum=0):
  img    = sk.isMsk01(img)
  lrSum  = np.sum(img, axis=0)
  lrInd  = np.where(lrSum == Sum)[0]
  return [lrInd[0], lrInd[-1]]


#ac  <a id='mkXyMps'></a>
def mkXyMps(imArr,
      show    = False):
  ''' 分肋间框内点
  参: imArr     : 胸椎图像
    show      : 是否显示
  返: mPs       : 分肋间框内点
    (lLn, rLn): 左右分割线
    msks      : 胸椎分割图
    xys       : 胸椎分割框
    (pls, prs): 左右分割框
  '''
  nda         = sk.isMsk01(imArr)
  nda_        = 1 - nda         # 反像
  lLn, rLn    = getMidSpc(nda_)  # 空白界
  mLn         = (lLn + rLn)/2   # 中间
  lDif        = rLn - lLn
  lLln, rLln  = ( lLn - lDif,
                  rLn + lDif)
  pd_Nda      =np.pad(nda_, 1, 'constant',
                      constant_values = 0)
  msks, xys   = lbMkBxs(pd_Nda)           # 分肋间
  pls, prs    = [], []
  for i, xy  in enumerate(xys): # 遍历每个框
    x, x1     = xy[0][1], xy[1][1]
    mEg       = [x, x1]\
                [(x < mLn) *1]    # 左右内侧
    if lLln < mEg < rLln:         # 限制在内
      if mEg == x:                      # 左侧
        pj    = msks[i][:, mEg:mEg+2]   # 截图
        pi    = np.where(pj!=0)[0].max()    # 截图内点集
        prs  += [[mEg,pi],]
      else:
        pj    = msks[i][:, mEg-2:mEg]
        pi    = np.where(pj!=0)[0].max()
        pls  += [[mEg,pi],]
  mPs         = ndA(pls + prs)
  if show:
    zwShow(nda_,
      ps  = mPs,
      yLns= (lLn, rLn, lLln, rLln),
      bxs = xys
          )
  return mPs, (lLn, rLn), msks, xys, (pls, prs)

#ac  <a id='cutCut'></a>
def cutCut(imArr, lDif, mPs, bBx, img = None, show=0):
  ''' 分肋间框内点
  参: imArr     : 胸椎图像
    lDif      : 左右分割线差
    mPs       : 分肋间框内点
  返: (p0, p1)  : 中线起点, 终点
    (pl0, pr0): 左右分割线起点
    ndac      : 中线截图
    cCutDic   : 分肋间框
    (psl, psr): 左右分割线
  '''

  nda         = sk.isMsk01(imArr)
  hi          = nda.shape[0]
  p0, p1      = psFitLine(mPs, hi)    # 拟合中线
  # lDf         = lDif*.5
  lx0, rx0    = p0[0] - lDif, p0[0] + lDif
  lx1, rx1    = p1[0] - lDif, p1[0] + lDif
  pl0, pr0    = ndA([lx0, 0]) , ndA([rx0, 0])
  pl1, pr1    = ndA([lx1, hi]), ndA([rx1, hi])
  id0, id1    = ( min([lx0, lx1]).astype(int),\
          max([rx0, rx1]).astype(int))

  xl, xr  = ( min(mPs[:len(mPs)//2][:,0]),
          max(mPs[len(mPs)//2:][:,0]))
  ndac        = linImg(nda ,
          [xl,0], [xl,hi])
  ndac       += linImg(ndac,
          [xr,0], [xr,hi])
  ndac       -= nda
  ndaC        = sk.isMsk01(ndac[:, id0:id1]) # 中线截图
  x0, y0, z0  = bBx[:3]
  x1, y1, z1  = bBx[3:]
  bBx         = ndA([x0+id0, y0, z0,
          x0+id1, y1, z1])
  # psl         = linSecCont(nda, xl, xl)
  # psr         = linSecCont(nda, xr, xr)
  cCutDic     = getLbXy(ndaC)
  if show:
    zwShow(nda,
      # ps  = (psl, psr),
      pLs = (p0,p1,pl0,pl1,pr0,pr1),
      # bxs = cCutDic[1]
        )
  return [p0,p1], ndA([[pl0,pl1],[pr0,pr1]]), ndaC, cCutDic, bBx

#ac  <a id='psFitLine'></a>
def psFitLine(ps,
    pMax = None):
  ''' 点集拟合直线
  参: ps: 点集
  返: p0, p1, cP: 起点, 终点, 质心
  '''
  ps       = ndA(ps)
  cP       = np.mean(ps,          # 质心
                axis = 0)
  ps_c     = ps - cP
  print(f'{ps_c.shape=}')
  drt      = np.linalg.svd(
                ps_c)[2][0] # 奇异值分解
  # print(f'方向向量: {drt}')
  # 投影ps到drt上
  ps_d     = np.dot(ps_c, drt)
  # 筛选离散点, 删除相邻点距离大于3的点
  ps_d     = np.diff(ps_d)
  ps_d     = np.insert(ps_d > 3,
                0, 1)
  ps       = ps[ps_d]
  # print(f'{ps_d.shape=}')
  # 拟合直线
  if pMax == None:
    p0   = xyLin(cP, drt=drt,
                xy=[0, None])
    if pMax == 0:
      pMax = ps[:,1].max()
    p1       = xyLin(cP, drt=drt,
                xy=[None,pMax])
    return     p0, p1, drt, cP
  else:
    return     cP, drt

#ac  <a id='bBxTvb'></a>
def bBxTvb(cPs,
    msk   = None,
    lb1St = 13,
    show  = False):
  ''' 胸椎分框
  参: mPs   : 质心集
    msk   : 胸椎正位图
    lb1St : 起始标签
    show  : 是否显示
  返: xys   : 框集
    cPs   : 质心集
  '''
  pArr      = ndA(cPs)
  pArr      = pArr[pArr[:,1].argsort()]
  p2s       = pArr.reshape(-1, 2, 2)
  cPs       = np.mean(p2s, axis=1)
  xys       = []
  p4s       = []
  for i in range(len(p2s)-1):
    p4    = p2s[i:i+2].reshape(-1, 2)
    y, y1 = p4[:,0].min(),\
        p4[:,0].max()
    x, x1 = p4[:,1].min(),\
        p4[:,1].max()
    xys  += [[i+lb1St,    # 标签
          x, y, x1, y1], ]
    p4s  += [p4]
  if show:
    zwShow( msk,
        xLns = ndA(cPs[:,1]),
        bxs = ndA(xys)[:,1:]
        )
    # samShow(msk = msk,
    #         p4s = ndA(bxs),
    #         )
  return       xys, cPs

#ac  <a id='obb2d'></a>
def obb2d(imArr,
    isPs   = True,
    show   = False
          ):
  if isPs   is False:                 # 是否是坐标集
    imArr  = sk.isNdaU8(imArr)
    assert imArr.ndim == 2, "Input must be 2D array"
    x, y   = np.where(imArr > 0)
    ps     = np.vstack([x, y]).T
  else:
    ps     = imArr
  ps         = ndA(ps)
  # ps         = ps - np.mean(ps, axis=0)
  cov        = np.cov(ps,
          y      = None,
          rowvar = 0,
          bias   = 1)
  vec        = np.linalg.eig(cov)[1]
  vecT       = np.transpose(vec)
  psR        = np.dot(ps,
          np.linalg.inv(vecT))

  cMin       = np.min(psR, axis=0)
  cMax       = np.max(psR, axis=0)

  xmin, xmax = cMin[0], cMax[0]
  ymin, ymax = cMin[1], cMax[1]

  x_x        = xmax - xmin
  y_y        = ymax - ymin
  # idx        = np.argsort([x_x, y_y])[::-1]
  xDif, yDif = ndA([x_x/2, y_y/2]) # [idx]
  xMin, yMin = ndA([xmin, ymin]) # [idx]

  cx         = xMin + xDif
  cy         = yMin + yDif

  cnr        = ndA([
          [cx - xDif, cy - yDif],
          [cx - xDif, cy + yDif],
          [cx + xDif, cy + yDif],
          [cx + xDif, cy - yDif],
          ])
  cn         = np.dot(cnr, vecT)
  if show:
    cns    = ndA([cn[0], cn[1], cn[2], cn[3], cn[0]])
    zwShow(
        [imArr, None][isPs*1],
      ps =[None, ps][isPs*1],
      pLs=[cns])
  return cn

#ac <a id='plnImg3D'></a>
#ac  <a id='plnImg3D'></a>
def plnImg3D(   im3D,
        pp   : Seq,
        pn   : Seq,
        pix  : Opt[int]
            = None,
        bools: Lit['+','-','x']
            = '+',
        # full : bool
        #         = False):
        )-> sitk.Image:
  ''' 在im3D里面创建一个像素为0的平面
  参: im3D: 三维图像数组
    pp: 平面上的一点
    pn: 平面法向量或与pp法线的图外一点
    pix: 平面像素值
    bools: 操作符(合差交集布尔计算)
  返: plnArr: 平面图
  '''
  pp, pn      = ndA(pp), ndA(pn)
  img         = sk.isImg255(im3D)
  imArr       = sk.isNdaU8(img)
  size        = img.GetSize()
  if pix is None:
    pix     = imArr.max()
  v           = (pp - pn)
  norm        = v / np.linalg.norm(v)
  if norm[2] == 0:
    norm[2] = 1e-5
  d           = -np.dot(pp, norm)
  x, y        = np.meshgrid(
          np.arange(size[0]),
          np.arange(size[1]))
  z           = (   norm[0] * x\
          + norm[1] * y\
          + d)\
        / -norm[2] # 平面方程
  # if full: # 平面全图贯穿
  #     plane   = np.zeros(size)
  #     plane[z>-0.1]\
  #             = 1
  pArr        = np.vstack([x.flatten(),
          y.flatten(),
          z.flatten()]).T
  print(f'{pArr.shape=}')
  self.im        = sitk.Image(
          size[0],
          size[1],
          size[2],
          sitk.sitkUInt8)\
      * 0
  self.im.CopyInformation(img)
  imArr0      = sk.isMsk01(self.im)
  pArr        = pArr[
        (  pArr[:, 0] >= 0)
        & (pArr[:, 0] < size[0])
        & (pArr[:, 1] >= 0)
        & (pArr[:, 1] < size[1])
        & (pArr[:, 2] >= 0)
        & (pArr[:, 2] < size[2])]
  pArr        = np.round(pArr).astype(int)
  imArr0[     pArr[:, 1],
        pArr[:, 0],
        pArr[:, 2]]\
        = pix
  if bools   == '+':
    imArr  += imArr0
  elif bools == '-':
    imArr  -= imArr0
  elif bools == 'x':
    imArr  *= imArr0
  pImg = sitk.GetImageFromArray(imArr)
  pImg.CopyInformation(img)
  return        pImg

#ac  <a id='getMaxInCir'></a>
def getMaxInCir(img,
                show = False):
  ''' 最大内切圆
  参: img   : 2D图像
  返: (x, y): 中心
    r     : 半径
  '''
  im01    = sk.isImg01(img)
  dstMap  = sitk.SignedMaurerDistanceMap(im01,
                insideIsPositive=True,
                squaredDistance =False,
                useImageSpacing =False)
  dstArr  = sitk.GetArrayFromImage(dstMap)
  r       = dstArr.max()
  if r    < 1.4:
    return
  idMax   = np.unravel_index(dstArr.argmax(),
                dstArr.shape)
  x, y    = idMax[1], idMax[0]
  print(f'圆心坐标: ({x}, {y}); 半径: {r}')
  if show:
    plt.imshow(sk.isMsk01(im01), cmap='gray')
    # plt.imshow(dstArr, cmap='jet', alpha=0.5)
    cir = plt.Circle((x, y), r, color='r', fill=False)
    plt.plot(x, y, 'ro')
    plt.gca().add_patch(cir)
    plt.show()
  return    ndA([x, y]), r

#ac  <a id='spCanalCps'></a>
def spCanalCps(imArr):
  ''' 椎管质心
  参: imArr : 2D图像
  返: cps   : 椎管质心集
  '''
  im01            = sk.isMsk01(imArr)
  zgs, cps, rds   = [], [], []
  for i, im in enumerate(np.pad(im01,1)):
    lb, nm        = label(1 - im) # 反求椎管标签
    if nm        == 0:
      continue
    if nm         > 1:
      zg          = lb == 2
      # print(np.sum(zg))
      if np.sum(zg) < 33:
        continue
      (x, y), r   = getMaxInCir(zg)
      zgs        += [zg]
      # pss += [ps]
      cps        += [[x, y, i],]
      rds        += [r]
  return            zgs, cps, rds

#ac  <a id='getCanalCLn'></a>
def getCanalCLn(imArr,
        thr  : int
            = 3,
        slice: Lit['X','Y','Z','']
            = 'X'
        ):
  ''' 拟合椎管中线
  参: pArr : 椎管点集
    thr  : 拟合阈值
  返: p0, p1, p: 起点, 终点, 拟合参数

  '''
  imArr       = sk.isMsk01(imArr)
  hi          = imArr.shape[1]
  cps         = spCanalCps(imArr)[1]
  if   slice == 'X': # 侧
    pArr    = cps[..., [1,2]]
  elif slice == 'Y': # 正
    pArr    = cps[..., [0,2]]
  elif slice == 'Z': # 轴
    pArr    = cps[..., [0,1]]
  else:
    pArr    = cps
  x0          = pArr[:,0]  # X轴筛选
  df          = np.diff(x0)
  ids         = np.where(df > thr)[0]
  ps          = np.delete(pArr,
          ids, axis=0)
  x, y        = ps[:,0], ps[:,1]
  ps          = np.polyfit(x, y, 1)
  p0          = ndA([-ps[1]/ps[0],
          0])
  p1          = ndA([(hi-ps[1])/ps[0],
          hi])
  return        [p0, p1], ps

#ac <a id='lbMksBxs'></a>
def lbMkBxs(imArr,
            sort = False
                    ):
    ''' 标签分割
    参: imArr: 二值图
    返: mks  : 标签集
      bxs  : 框集
    '''
    imArr        = sk.isMsk01(imArr)
    y, x         = imArr.shape
    lbs          = label(imArr)[0]
    obs          = ndIm.find_objects(lbs)
    mks, bxs     = [], []
    if   x      == 1:
      for ob   in obs:
          if np.any(ob):
            bxs += [ndA([ ob[0].start,
                          ob[0].stop  ])\
                  - ndA([ ob[1].start,
                          ob[1].stop  ]),]
      return       ndA(bxs)
    elif x       > 1:
      for lb, ob   in enumerate(obs):
        if np.any(ob):
          bxs   += [ndA([ [ ob[0].start,
                            ob[1].start ],
                          [ ob[0].stop,
                            ob[1].stop  ]]),]
          mks   += [(lbs == lb+1)*1,]
      if sort:
        return     xySort(bxs, [y, x], mks)
      else:
        return     mks, bxs

isList = lambda xs: [ list(x) for x in xs]

arrs2List = lambda arrs: [arr.tolist()
                          if isinstance(arr, np.ndarray)
                          else arr
                          for arr in arrs
]

isNda = lambda xs: [ndA(x) for x in xs]

def sortYx(ys,xs,
          reVs=False):
  ys, xs = arrs2List([ys,xs])
  zpXy = zip(xs,ys)
  zpSt = sorted(zpXy,
                key=lambda x:x[0],
                reverse=reVs)
  zps  = zip(*zpSt)
  ySb  = [list(x)
          for x in zps][1]
  return ySb

def xySort(arr,
          size,
          arr1=None):
  '''

  '''
  arr     = ndA(arr)
  siz     = size[0]
  print(f'{siz=}')
  ind     = np.argsort(arr[0][:,0]) # 按y排序
  print(f'{ind=}')
  arr     = arr[ind]             # 排序
  if arr1 is not None:
    assert len(arr) == len(arr1),\
    f'{len(arr)=} != {len(arr1)=}'
    arr1  = sortYx(arr1,ind)
  # 计算y差
  df      = np.diff(arr, axis = 1)[:,0,0]
  dfInd   = ((df<siz*.8)&(df>2))*1    # 过滤距离小于2的元素
  print(f'{len(dfInd)=}')
  ind2    = np.insert(dfInd, 0, 1) # 插入第一个元素
  if arr1 is not None:
    arr1  = sortYx(arr1,ind2)
    return  arr[ind2], arr1
  else:
    return arr[ind2]

#%%
ST1 = 1
# d.close()
d = shelve.open(f'/Users/liguimei/Documents/GitHub/SAMed/2Dto3D/output/output5')
list(d.keys())
#%%
im = sk.readImgSk(f'/Users/liguimei/Documents/GitHub/SAMed/datasets/workDir/data/sub-verse518_dir-ax_ct.nii.gz')
#%%
print(f'{bBx0=}')
#%%
bImg, bBx1, re\
        = sk.getBoneSeg(im)
print(f'{bBx1=}')
#%%
# bImg, bBx, imRef = sk.bImg5, sk.bBx5, sk.im5

rArr = sk.isNdaU8(im)
rXarr=np.sum(rArr, axis=2)
rYarr=np.sum(rArr, axis=1)
rZarr=np.sum(rArr, axis=0)
bBx0    = sk.getBbox(im)
imArr = sk.isNdaU8(im)
#%%
bImg = sk.bImg5

(x,y,z),(x1,y1,z1) = bBx0
(x0,y0,z0),(x01,y01,z01) = sk.bBx5
bArr = sk.isMsk01(bImg)
zwArr = np.max(bArr, 1)
zwShow(rYarr, yLns=[x0, x01])
#%%
# bArr = sk.isMsk01(bImg)
bArrF = fillPad(zwArr,
                ndA([[x0, z0], [x01, z01]]),
                ndA([[x, z],[x1, z1]]))
# ##
# 1. 读取数据
# 2. 自动阈值分割--'getBoneSeg'-->'bImg'
# 3. 正位取中截半(二维三维同时)
# 4. 正位找最大间隙(骶肋线)
#%%
zwShow([rYarr, bArrF], yLns=[x0, x01])

#%%
# 3. 正位中线裁切(二维三维同时)
cArrZ, cImg, (lxLn, rxLn), bbx, bBox\
  = midCut(bImg, show = 1)
#%%
# 4. 正位找最大间隙(骶肋线)
gPs = getTlGps(cArrZ, show = 1)
#%%
# 5. 截脊柱,分腰胸(二维三维同时)
# 6. 正位胸段取轮廓, 分肋间
(tArr, lArr), (lps, rps), (tVb3D, lVb3D), (tBbx, lBbx)\
  = cutSpTL(cArrZ, gPs, bbx, im3D = cImg)
tBox = np.copy(bBox)
tBox[:,2] = tBbx[:,2]
lBox = np.copy(bBox)
lBox[:,2] = lBbx[:,2]
#%%

# 7. 中轴空白界线, 肋间点及中点
# 8. 肋间内点 (是否拟合中线, 求胸椎正位倾斜角)
midPs, (lLn, rLn), mks, xys,(pls, prs)\
  = mkXyMps(tArr, show=1)
#%% 胸椎分框
bxx = bBxTvb(midPs, msk=tArr, show=1)
#%%
[p0,p1], plrs, ndaC, mkxy, bBxx\
= cutCut(tArr, (rLn-lLn), midPs, bBoxs[0], show=1)
#%% 根据肋间内点拟合中线
#ac  <a id='samShow'></a>
def samShow(img=None, msk=None, boxes=None,
      bxCapt=['',''], clsDic=LBLVS,                   p4s=None,
      fSize=(10, 10), dpi = 96, margin=.5, pad=None, opacity=0.4):
  def __showMsk(masks=msk, ax=None, opacity=opacity, clsDic=clsDic, pad=pad, lbMin=13):
    labels = np.unique(masks)[1:]
    print(labels)
    # if labels > 0: # or not np.isnan(labels):
    if p4s is not None:
      plt.imshow(msk, cmap='gray')
      for i, p in enumerate(ndA(p4s)):
        lb = lbMin + i
        level = clsDic[lb] if lb < 30 else str(i+1)
        vClr = COLORS[lb]
        __showPs(p, ax)
        x0, x1 = p[:,0].min(), p[:,0].max()
        y0, y1 = p[:,1].min(), p[:,1].max()
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor=vClr, facecolor=(0,0,0,0), lw=2))
        locLb = level + ' ' + bxCapt[0]
        iouLb = f' {bxCapt[1]}'
        plt.text(x0, y0, locLb+iouLb, fontsize=12, color='white',backgroundcolor=vClr.tolist()+[.3])
        plt.text(x0, y0, locLb+iouLb, fontsize=12, color='white',backgroundcolor=vClr.tolist()+[.3])
    else:
      for i, lb in enumerate(labels):
        level = clsDic[lb] if lb < 30 else str(i+1)
        vClr = COLORS[int(lb)]
        if masks.ndim == 2:
          mask = ndA(masks==lb)
        else:
          mask = ndA(masks[i]==lb)
        h, w = mask.shape[-2:]
        vClr = ndA(vClr.tolist() + [opacity])
        mask_image = mask.reshape(h, w, 1) * vClr.reshape(1, 1, -1)
        ax.imshow(mask_image)
        yInd, xInd = np.where(mask==1)
        x0, x1 = np.min(xInd), np.max(xInd)
        y0, y1 = np.min(yInd), np.max(yInd)
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor=vClr, facecolor=(0,0,0,0), lw=2))
        if pad is not None:
          h, w = mask.shape
          xP0 = max(0, x0 - np.random.randint(0, pad))
          xP1 = min(w, x1 + np.random.randint(0, pad))
          yP0 = max(0, y0 - np.random.randint(0, pad))
          yP1 = min(h, y1 + np.random.randint(0, pad))
          ax.add_patch(plt.Rectangle((xP0,yP0),xP1-xP0,yP1-yP0,edgecolor=vClr,facecolor=(0,0,0,0), lw=.5))
        locLb = level + ' ' + bxCapt[0]
        iouLb = f' {bxCapt[1]}'
        plt.text(x0, y0, locLb+iouLb, fontsize=12, color='white',backgroundcolor=vClr)
        plt.text(x0, y0, locLb+iouLb, fontsize=12, color='white',backgroundcolor=vClr)
      # plt.text(x1, y1, 'A\n R', fontsize=12, color='white',backgroundcolor=vClr)

  def __showPs(ps, ax, marker_size=255):
    ax.scatter(ps[:, 0], ps[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

  # def __showBox(box, ax):
  #     x0, y0 = box[0], box[1]
  #     w, h = box[2] - box[0], box[3] - box[1]
  #     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
  if img is not None:
    if isinstance(img, sitk.Image):
      img = readImgSk(img=img, img3C=True)
  if msk is not None:
    if not isinstance(msk, np.ndarray):
      msk = sitk.GetArrayFromImage(msk)
  # if img is not None and msk is not None:
  #     assert img.shape[:2] == msk.shape[:2], 'img and msk must have the same shape!'
  hws = [img,msk][img is None]
  print(hws.shape)
  if fSize is None:
    h, w = hws.shape[1:3]
    fSize = (h/dpi, w/dpi)
    figsize = (1 + margin) * h / dpi, (1 + margin) * w / dpi
  else:
    h,w = np.asarray(fSize)*dpi

  def callback(zI_to_S=None):
    plt.figure(figsize=fSize, dpi=dpi)
    # plt.gca().clear()
    if zI_to_S is None:
      if img is not None:
        plt.imshow(img, cmap='gray')
      if msk is not None:
        __showMsk(msk, plt.gca())
      # plt.imshow(img, cmap='gray')
    else:
      if img is not None:
        plt.imshow(img[zI_to_S,...], cmap='gray')
      if msk is not None:
        __showMsk(msk[zI_to_S,...], plt.gca()) # gca: get current axis
    # if points is not None:
    #     __showPs(points, labels, plt.gca())
    # if boxes is not None:
    #     for box in [boxes,[boxes]][int(len(boxes)==1)]:
    #         __showBox(box, plt.gca())
    plt.text(0,0,'O L>>>R\nA\nV\nV\nP', ha='left',va='top',fontsize=16, color='white')
    plt.axis('off')
    plt.show()

    return plt

  # if imgs.ndim == 2:
  #     callback()
  if img is not None:
    if img.ndim in[3,4]:
      if img.shape[2] in [1,3]: # 若是单通道或三通道图像
        callback()
      else:
        zAxis = img.shape[0]-1
        # print(imgs.shape)
        interact(callback, zI_to_S=(0, zAxis))
  if msk is not None:
    if msk.ndim > 2:
      zAxis = msk.shape[0]-1
      interact(callback, zI_to_S=(0, zAxis))
    else:
      callback()
# return mask, phi0, res
#%%

from itertools import combinations as ittCom
def getBbox(imArr,
  pad    = 0):
  """获取2D和3D图像的bBox
  参: img: 图像
      pad: 填充
  返: bBx: xyxy或xyzxyz
  """
  if isinstance(imArr, sitk.Image):
    im01    = sk.isImg01(imArr)
    dim     = im01.GetDimension()
    ls      = sitk.LabelShapeStatisticsImageFilter()
    ls      . Execute(im01)
    bb      = np.array(ls.GetBoundingBox(1)) # (x,y,z,w,h,d)
    bXy     = ndA([bb[:dim], bb[dim:]+bb[:dim]]) # (z,y,x)
  elif isinstance(imArr, np.ndarray):
    dim     = imArr.ndim
    out     = []
    for ax in ittCom(range(dim),
                        dim - 1):
      non0    = np.any(imArr, axis = ax)
      xxyy    = ndA(np.where(non0)[0][[0, -1]])
      out     . extend([xxyy],)
      # print(f'{len(out)=}')

    bXy     = np.vstack(ndA(out)).T
    bXy[1] += 1
  if pad   != 0:
    bXy     = padBbx(bXy, pad, imArr)
  return bXy
#%%
getBbox(bImg)
#%%
def padBbx(bBx, pBx,
  imArr   = None,
  fill    = True
          ):
  bBx     = ndA(bBx),
  pBx     = ndA(pBx)
  if not isinstance(pBx, np.ndarray):
    ini, fin  \
            = bBx
    ini    -= pBx
    fin    += pBx
  else:
    ini, fin  \
            = pBx
  if imArr  is not None:
    imArr   = sk.isImg01(imArr)
    dim     = imArr.ndim
    siz     = imArr.shape
    ini[ini<0] \
            = 0 # 防止ini小于0
    for i  in range(dim): # 防止fin超出图像的范围
      fin[i]  = min(fin[i], siz[i])
  pBx     = ndA([ini, fin])
  if fill & (imArr is not None):
    pArr    = fillPad(imArr, pBx, bBx)
    return pArr, pBx
  else:
    return pBx
#%%
def fillPad(imArr, bBx, bBx0):
  imArr    = sk.isMsk01(imArr)
  df       = abs(ndA(bBx0)\
            -ndA(bBx))
  pad      = np.vstack(df).T
  pArr     = np.pad(imArr,
              pad[::-1],
              'constant')
  return pArr
#%%
def getLargestCnn(img):
  img = isImg01(img)
  cnn = sitk.ConnectedComponent(img)
  rbl = sitk.RelabelComponent(cnn)
  lb  = (rbl==1)*1
  mk  = sitk.Mask(img, lb)
  bbox = sitk.LabelStatisticsImageFilter ()
  bbox.Execute(mk, mk)
  bb = ndA(bbox.GetBoundingBox(1))
  return mk, bb
#%%
vbs = [ 'vertebrae_L2' , 'vertebrae_L1' ,
        'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10',
  ]
from totalsegmentator.python_api import totalsegmentator

totalsegmentator('//Users/liguimei/Documents/GitHub/TotalSegmentator/tests/reference_files/example_ct.nii.gz', '/Users/liguimei/Documents/GitHub/TotalSegmentator/tests/prediction_vbs.nii.gz', ml=True, roi_subset=vbs,device="cpu", radiomics=True)




#%%
END = 1

[EOF]

