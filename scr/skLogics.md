# 医学影像预处理


```python title="import"
import shelve
import pyvista as pv
import vtk
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import pickle
import itertools as itt
from collections import defaultdict as dfDic
from ipywidgets import interact
import copy
from matplotlib.patches import Circle, Rectangle
from typing import Literal as Lit
from typing import Sequence as Seq
from typing import Union as Un
from typing import Optional as Opt
import typing

import os
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import center_of_mass
# import nibabel as nib
# import nibabel.processing as nip
# import nibabel.orientations as nio
import tqdm
import json
import typing
from typing import Optional as opt
import itertools

ndA = np.array

LBLVS = {
    1 : 'C1',  2: 'C2',  3: 'C3',    4: 'C4',   5: 'C5',   6: 'C6',   7: 'C7',
    8 : 'T1',  9: 'T2',  10: 'T3',  11: 'T4',  12: 'T5',  13: 'T6',  14: 'T7',
    15: 'T8', 16: 'T9',  17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',  21: 'L2',
    22: 'L3', 23: 'L4',  24: 'L5',  25: 'L6',
    26: 'Sacrum',        27: 'Cocc',           28: 'T13', 29: 'Lv?',
    }

COLORS = (1/255)*ndA([
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # Label 1-7 (C1-7)
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122], [165, 42, 42],  # Label 8-19 (T1-12)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205],  # Label 20-26 (L1-6, sacrum)
    [255,235,205], [255,228,196],  # Label 27 cocc, 28 T13,
    [218,165, 32], [  0,128,128], [188,143,143], [255,105,180],
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # 29-39 unused
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122],   # Label 40-50 (subregions)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205], [255,105,180], [165, 42, 42], [188,143,143],
[255,235,205], [255,228,196], [218,165, 32], [  0,128,128], [30,14,255]# rest unused
    ])
CMITK = ListedColormap(COLORS)
CMITK.set_bad(color='w', alpha=0)  # set NaN to full opacity for overlay
SBWIN = Normalize(vmin=-500, vmax=1300, clip=True) # 软骨窗是-500到1300, 用于显示软骨
HBWIN = Normalize(vmin=-200, vmax=1000, clip=True) # 骨窗是-200到1000, 用于显示骨头

```

#ac  <a id='pk2file'></a>

### [[pk2file]]
```python
def pk2file(file, data=None):
    # if not file.endswith('.pickle' or '.pkl'):
    #     file = file+'.pickle'
    if data is not None:
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

np.savez

```

#ac  <a id='isNdaU8'></a>

### [[isNdaU8]]
```python
def isNdaU8(nda):
    if not isinstance(nda, np.ndarray):
        nda = sitk.GetArrayFromImage(nda)
    if isBinImg(nda):
        return (nda*1).astype(int)
    else:
        return nda.astype(np.uint8)

```

#ac  <a id='isImg255'></a>

### [[isImg255]]
```python
def isImg255(img):
    if not isinstance(img, sitk.Image):
        img = sitk.GetImageFromArray(img)
    if isBinImg(img):
        return img
    else:
        img = sitk.RescaleIntensity(img)
        return sitk.Cast(img, sitk.sitkUInt8)

```

#ac  <a id='isMsk01'></a>

### [[isMsk01]]
```python
def isMsk01(mkArr, thrdMin=0, lb =1):
    mkArr        = isNdaU8(mkArr)
    mkArr[mkArr>thrdMin] = lb
    return mkArr.astype(int)

```

#ac  <a id='isImg01'></a>

### [[isImg01]]
```python
def isImg01(img, thrdMin=0, lb =1):
    img = isMsk01(img, thrdMin, lb)
    return sitk.GetImageFromArray(img)

```

#ac  <a id='isBinImg'></a>

### [[isBinImg]]
```python
def isBinImg(img):
    def __isBin(nda):
        uq = np.unique(nda)
        bi = len(uq)==2
        # if bi:
            # print(f' is a binary image: {uq[0:2]}')
        return bi
    if isinstance(img, sitk.Image):
        if img.GetPixelID() > 3:
            return False
        else:
            return __isBin(sitk.GetArrayFromImage(img))
    elif isinstance(img, np.ndarray):
            return __isBin(img)

```

#ac  <a id='readDcm'></a>

### [[readDcm]]
```python
def readDcm(dcm_path, drt='IPL', spc=(1,1,1)):
    """
    读取DICOM图像序列,返回3D Image对象
    """
    # 读取DICOM图像序列
    reader = sitk.ImageSeriesReader()
    dcms   = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dcms)

    img3D  = reader.Execute()
    if img3D is None:
        print('Error: less than 3 dimensions')
        return None

    imgIso = reOrIso(img3D, drt, spc)

    print('Image data type: ', imgIso.GetPixelIDTypeAsString())
    print('Image size: '     , imgIso.GetSize()               )
    print('Image spacing: '  , imgIso.GetSpacing()            )
    print('Image origin: '   , imgIso.GetOrigin()             )
    print('Image dimension: ', imgIso.GetDimension()          )

    return imgIso
```

#ac  <a id='writeImg'></a>

### [[writeImg]]
```python
def writeImg(
        filePath=None,
        img=None,
        msk=False,
        img3C=False,
        outPath='',
        outFile='',
    ):
    if filePath is not None:
        filePath, fileName = os.path.split(filePath)
    if outPath == '':
        outPath = filePath+'/skImgs/'
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    if outFile == '':
        outFile = fileName
    if msk or img3C:
        path = outPath + fileName + '.npy'
        if not os.path.exists(path):
            np.save(path, img)
    else:
        path = outPath + fileName
        if not os.path.exists(path):
            sitk.WriteImage(img, path)

```

#ac  <a id='getDrt'></a>

### [[getDrt]]
```python
def getDrt(img):
    dic = dict( R = (-1.,  0.,  0.), L = ( 1.,  0.,  0.),
                A = ( 0., -1.,  0.), P = ( 0.,  1.,  0.),
                S = ( 0.,  0., -1.), I = ( 0.,  0.,  1.))
    if isinstance(img, sitk.Image):
        d = img.GetDirection()
        drtStr = ''
        for i in range(len(d)//3):
            for k,v in dic.items():
                if d[i*3:i*3+3] == v:
                    drtStr += k
        return drtStr
    elif isinstance(img, str):
        drt = ()
        for d in img:
            if d not in dic.keys():
                raise ValueError(
                    'Invalid drtString: %s' % img)
            drt += dic[d]
        return drt

```

#ac  <a id='reOrIso'></a>

### [[reOrIso]]
```python
def reOrIso(img, isMsk=False, spc=(1, 1, 1), drt='LPI'):
    '''
    '''
    img = isImg255(img)
    spcOr = img.GetSpacing()
    if spc != spcOr:
        # Image is already isotropic, just return a copy.
        # Make image isotropic via resampling.
        sizOr = img.GetSize()
        if spc is None:
            spcMin = min(spcOr)
            spc = [spcMin]*img.GetDimension()
        siz = [int(round(osz*ospc/sp))
                    for osz,ospc,sp
                    in zip(sizOr, spcOr, spc)]
        itpltor = [ sitk.sitkLinear,
                    sitk.sitkNearestNeighbor
                    ][int(isMsk)]
        reSpl = sitk.ResampleImageFilter()
        reSpl.SetOutputSpacing(spc)
        reSpl.SetSize(siz)
        reSpl.SetOutputDirection(img.GetDirection())
        reSpl.SetOutputOrigin(img.GetOrigin())
        reSpl.SetTransform(sitk.Transform())
        reSpl.SetDefaultPixelValue(img.GetPixelIDValue())
        reSpl.SetInterpolator(itpltor)
        img = reSpl.Execute(img)
    if drt == '':
        return img
    elif drt != getDrt(img):
        return sitk.DICOMOrient(img, drt)
    else:
        return img
# @staticmethod
```

#ac  <a id='readImgSk'></a>

### [[readImgSk]]
```python
def readImgSk(  filePath=None,
                img = None,
                isMsk = False,
                img3C = False,
                imgIso = True,

            ):
    '''sitk读图
    '''
    if img is None:
        if filePath.endswith('/'):
            img = readDcm(filePath)
        else:
            img = sitk.ReadImage(filePath)
            if imgIso:
                img = reOrIso(img, isMsk)
    if isMsk:
        return isNdaU8(img)

    img         = isImg255(img)
    if img3C:            # 三通道无法做魔法糖
        img     = sitk.GetArrayFromImage(  # (z,y,x,c)
                    sitk.Compose([img]*3))
    return img

```

#ac  <a id='cropOtsu'></a>

### [[cropOtsu]]
```python
def cropOtsu(image):
    ''' Otsu阈值裁图
    '''
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter() # 用于计算图像的轴对齐边界框。
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    # cnnMax是outside_value的最大连通域
    bounding_box = label_shape_filter.GetBoundingBox(outside_value) # 获取边界框
    print('bounding_box', bounding_box)
    return sitk.RegionOfInterest(
        image,
        bounding_box[int(len(bounding_box) / 2) :],
        bounding_box[0 : int(len(bounding_box) / 2)],
        )

```

#ac  <a id='img2Nii'></a>

### [[img2Nii]]
```python
def img2Nii(img, savePath=''):
    if not isinstance(img,sitk.Image):
        img=sitk.ReadImage(img)
    #将图片转化为数组
    img_arr=sitk.GetArrayFromImage(img)
    out = sitk.GetImageFromArray(img_arr)
    out.SetDirection(img.GetDirection())
    if savePath!='':
        sitk.WriteImage(out, savePath)
    return out

```

#ac  <a id='sk3C2gd'></a>

### [[sk3C2gd]]
```python
def sk3C2gd(img):
    # 三通道转灰度
    # if img.GetDimension() == 3:
    return ( sitk.VectorIndexSelectionCast(img, 0)
            +sitk.VectorIndexSelectionCast(img, 1)
            +sitk.VectorIndexSelectionCast(img, 2)
            ) / 3

```

#ac  <a id='skShow'></a>

### [[skShow]]
```python
def skShow( img,
            xyz = 'Z',
            msk = None,
            bbx = None,
            lbs = None,
            title=None,
            margin=0.05,
            dpi=96,
            fSize=None,
            cmap="gray"):
    nda = isNdaU8(img)
    img = isImg255(img)
    if msk is not None:
        if isinstance(msk, np.ndarray):
            ndaM = np.copy(msk)*1.
            ndaM[ndaM==0] = np.nan
            msk = sitk.GetImageFromArray(msk)
        else:
            ndaM = sitk.GetArrayFromImage(msk)
            ndaM[ndaM==0] = np.nan
            msk = copy.deepcopy(msk)
    spc = img.GetSpacing()
    siz = img.GetSize()
    if nda.ndim == 3: # 若3维数组
        c = nda.shape[-1] # 通道数
        if c in (3, 4): # 若通道数为3或4, 则认为是2D图像
            nda = nda[:,:,0]
    elif nda.ndim == 4: # 若4维数组
        c = nda.shape[-1]
        if c in (3, 4): # 若通道数不为3或4, 则认为是3Dv(4D)图像, 退出
            # 去掉最后一维
            nda = nda[:,:,:,0]
        else:
            raise RuntimeError("Unable to show 3D-vector Image")
    if nda.ndim == 2: # 若2维数组
        nda  = nda[np.newaxis, ...] # nda增加后的维度为3维, 且最后一维为1
        ndaM = ndaM[np.newaxis, ..., np.newaxis] if msk is not None else None
        siz = siz + (1,) # size增加后的维度为3维, 且最后一维为1
        spc  = 1.
    # nda.shape = shape# nda的方向为LPS
    # size = nda.shape # size为z,y,x
    print('size:',siz)
    xyzSize = [ (i+1)*1
                for i
                in ( ndA(spc)
                    *ndA(siz))
                ]
    sInd = {'X':2, 'Y':1, 'Z':0}[xyz]
    # sDic = dfDic(fdic)
    sDic = [dict(drt=['P==>A', 'L==>R'],
                arr = nda, # nda的方向为LP
                arrM = ndaM if msk is not None else None,
                x = xyzSize[0],
                y = xyzSize[1],
                z = siz[2],
                extent = (0, xyzSize[0], 0, xyzSize[1]) # (left, right, bottom, top)
                ),
            dict(drt=['I==>S', 'L==>R'],
                arr=np.transpose(nda, (1,0,2)), # nda的方向为LS
                arrM = np.transpose(ndaM, (1,0,2)) if msk is not None else None,
                x = xyzSize[0],
                y = xyzSize[2],
                z = siz[1],
                extent = (0,xyzSize[0],xyzSize[2],0) # (left, right, bottom, top)
                ),
            dict(drt=['I==>S', 'A<==P'],
                arr=np.transpose(nda, (2,0,1)), # nda的方向为SP
                arrM = np.transpose(ndaM, (2,0,1)) if msk is not None else None,
                x = xyzSize[1],
                y = xyzSize[2],
                z = siz[0],
                extent = (0,xyzSize[1],xyzSize[2],0) # (left, right, bottom, top)
                )
            ][sInd]
    def callback(axe=None):
        figsize = (1 + margin) * sDic['y'] / dpi, (1 + margin) * sDic['x'] / dpi
        fig = plt.figure(figsize=[figsize, fSize][fSize is not None], dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
        ax.imshow(sDic['arr'][axe, ...], extent=sDic['extent'], interpolation=None,  cmap=cmap) #, norm=SBWIN)
        if msk is not None:
            mArr = sDic['arrM'][axe, ...]
            ax.imshow(mArr, extent=sDic['extent'], interpolation=None, cmap=CMITK, alpha=0.3, vmin=1, vmax=64)
            if bbx is not None and xyz=='Z': # [TODO]总是定不准位置
                ls = np.unique(mArr)
                print('ls:', ls)
                for l in ls:
                    if not np.isnan(l):
                        def __bBx(mArr, label):
                            y, x = np.where(mArr == label) # x,y为msk中值为label的点的坐标
                            xy0 = ndA([np.min(x), np.min(y)])
                            xy1 = ndA([np.max(x), np.max(y)])
                            pad = np.random.randint(0, 10)
                            return xy0, xy1, pad
                        xy0, xy1, pad = __bBx(mArr, l)
                        x0, y0 = xy0
                        x1, y1 = xy1
                        lw = 2 + pad
                        z = axe
                        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor=COLORS[int(l-1)], facecolor=(0,0,0,0), lw=lw))
        if lbs is not None:
            llbs = np.unique(lbs)
            leg = ax.legend(handles=[plt.Rectangle((0, 0), 5, 5, color=COLORS[int(lb)]) for lb in llbs], labels=[LBLVS[int(lb)] for lb in llbs], loc='center left')
            leg_height = leg.get_frame().get_height()
            fig_height = fig.get_figheight()
            if leg_height > fig_height:
                fig.set_figheight(leg_height)  # Set fig height to match legend height
    # 在图像上标注坐标轴
        ax.set_ylabel(sDic['drt'][0])
        ax.set_xlabel(sDic['drt'][1])
        # 根据颜色标lv图例

        if title:
            plt.title(title)
        return plt.show()
    interact(callback, axe=(0, sDic['z'] - 1))

```

#ac  <a id='skObb'></a>

### [[skObb]]
```python
def skObb(msk):
    msk = isImg255(msk)
    shpSts = sitk.LabelShapeStatisticsImageFilter()
    shpSts.ComputeOrientedBoundingBoxOn()
    shpSts.Execute(msk)
    lb = shpSts.GetLabels()
    return shpSts.    GetOrientedBoundingBoxVertices(lb[1])
# def showObbs(xyxys):
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111)
#     ax.imshow(labels)
#     for i in obbs.keys():
#         corners, centre = obbs[i]
#         ax.scatter(centre[1],centre[0])
#         ax.plot(corners[:,1],corners[:,0],'-')
#     plt.show()

```

#ac  <a id='xywh2xy'></a>

### [[xywh2xy]]
```python
def xywh2xy(bBx)-> np.ndarray:
    xywh  = ndA(bBx)
    dHf   = len(xywh)//2
    xyxy  = xywh[:dHf].tolist()
    xyxy += (xywh[:dHf]+xywh[dHf:]).tolist()
    return ndA(xyxy)

```

#ac  <a id='xy2xywh'></a>

### [[xy2xywh]]
```python
def xy2xywh(xyxy):
    xyxy = ndA(xyxy)
    xywh = ndA([xyxy[0],  xyxy[1],
                xyxy[2] - xyxy[0],
                xyxy[3] - xyxy[1]])
    return xywh

```

#ac  <a id='getIou'></a>

### [[getIou]]
```python
def getIou(xyxy1: list, xyxy2: list):
    box1_area = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
    box2_area = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
    interSection = max(0, min(xyxy1[2], xyxy2[2]) - max(xyxy1[0], xyxy2[0])) * max(0, min(xyxy1[3], xyxy2[3]) - max(xyxy1[1], xyxy2[1]))
    iou = interSection / (box1_area + box2_area - interSection)
    return iou

```

#ac  <a id='getXyz'></a>

### [[getXyz]]
```python
def getXyz(img, thrd = 1):
    ''' 获取图像中大于thrd的点的坐标

    '''
    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img)
    xyz = np.where(img>thrd)
    return xyz

```

#ac  <a id='psFitPla'></a>

### [[psFitPla]]
```python
def psFitPla(
    arr: np.ndarray,
    modName: str="",
    ):
    '''psFitPla 点云拟合平面

        根据输入的XYZ坐标数组，计算平面的法向量和中心位置，
        如果输入modName参数，则会在Slicer中创建一个PlaneNode节点，
        并将计算出来的法向量和中心位置赋值到该节点上。

    Args:
        arr (XYZ): 点云.
        modName (str, optional): 是否显示平面. Defaults to "".

    Returns:
        - 平面(正方向)

    Note:
        - _description_
    '''
    arr = np.asarray(arr)
    cp = np.mean(arr,axis=0)
    norm = np.linalg.svd((arr-cp).T)[0][:,-1]
    norm = norm*[1,-1][int(norm[2]<0)]
    # if modName!="":
    #     Helper.getPla(norm, cp, modName)
    return norm, cp

# cnn = sitk.ConnectedComponent(img)
# lbs = sitk.RelabelComponent(cnn)[1:10]
# bbx = []
# for lb in lbs:
#     msk = sitk.Mask(img, cnn==lb)
#     mkArr = sitk.GetArrayFromImage(msk)
#     sumPxl = mkArr.sum()
#     if sumPxl > 33:
#         y, x = np.where(mkArr>0)
#         bbx.append([x.min(), x.max(), y.min(), y.max()])

```

#ac  <a id='linSecCont'></a>

### [[linSecCont]]
```python
def linSecCont(cArr, p0, p1, show=1):
    ''' 直线与轮廓相交
    参: cArr: 轮廓数组
        p0, p1: 直线起点, 终点
        show: 是否显示
    返: secPs: 相交点集
    '''
    def __kp1St(arr,
            k   =2):
        # 过滤相邻元素之间距离小于 2 的元素
        arr     = ndA(arr)
        diff    = np.diff(arr[:, 1])
        keep    = np.insert(diff > k,
                    0, True)
        arr     = arr[keep]
        return arr
    x0, y0      = p0
    x1, y1      = p1
    dst         = p2p_dst(ndA(p0),
                    ndA(p1))
    # 生成间隔为1的整数序列
    lin         = np.arange(x0, x1)
    if dst     != 0:
        yy      = np.arange(y0, y1,
                    (y1-y0)/len(lin))
        lin     = np.vstack([yy, lin]).T
    # 生成line的罩和原图大小一致
    lnArr       = np.zeros_like(cArr)
    lnArr[  lin[:, 1].astype(int), #
            lin[:, 0].astype(int)]\
                = 1
    # 相交图
    secArr      = isMsk01(cArr)\
                * lnArr
    # 提取相交点
    y, x        = np.where(secArr)
    secPs       = [[x[i], y[i]]
                    for i
                    in range(len(x))]
    # 过滤相邻元素之间距离小于 2 的元素
    secPs       = __kp1St(secPs)
    if show:
        zwShow([cArr, lnArr], ps=secPs)
    return        secPs, secArr


#ac <a id='getMskBbx'></a>
class getMskBbx:
    ''' 罩机
        para:
            msk: 罩
            pad: 罩厚
        retun:
            Dic: 罩字典
                'bBxes': Gt_xyxy, pad, z,
                'pBbx': pad_xyxy,
                'lbLy': label, layer,
                'lvBbx': lv_xyxy,
                'bBx': bBx_xyxy(optional)
    '''
    def __init__(self, msk, pad=10):
        """Initializes the GetBbx class."""
        if isinstance(msk, sitk.Image):
            self.mArr = sitk.GetArrayFromImage(msk)
        else:
            self.mArr = msk
        self.pad = pad
        self.Dic = dfDic(dict)
        labels = np.unique(self.mArr)
        if len(labels) >= 2:
            self.labels = labels[1:]
            if len(self.labels) > 1:
                self.get_3d_bbox()
            elif len(self.labels) == 1:
                if self.labels==255:
                    level = LBLVS[29]
                else:
                    level = LBLVS[self.labels[0]]
                if len(self.mArr)==2:
                    bBx, pBbx = self.get_2d_bbox(
                        self.mArr[1],
                        self.labels[0])
                    self.Dic['bBx'][level] = bBx
                    self.Dic['pBbx'][level] = pBbx
                else:
                    self.get_3d_bbox()
        else:
            raise ValueError('No labels found in mask.')
        #     label = 29
        #     print('No labels found in mask.')

    def get_2d_bbox(self, arr2D, label):
        """Gets the 2D bounding box for a label."""
        cnnImg = getLargestCnn(arr2D)
        cnnArr = sitk.GetArrayFromImage(cnnImg)
        y, x = np.where(cnnArr == label)
        bBx = ndA([np.min(x), np.min(y),
                        np.max(x), np.max(y)
                        ])
        pxSum = np.sum(arr2D == label)
        if self.pad is not None:
            h, w = arr2D.shape
            pBbx = np.asarray([
                max(0,
                    bBx[0]\
                   -np.random.randint(0, self.pad)),
                max(0,
                    bBx[1]\
                   -np.random.randint(0, self.pad)),
                min(w,
                    bBx[2]\
                   +np.random.randint(0, self.pad)),
                min(h,
                    bBx[3]\
                   +np.random.randint(0, self.pad))
                                ])
        return bBx, pBbx, pxSum

    def get_3d_bbox(self):
        """Gets the 3D bounding box for a label."""
        for label in self.labels:
            z, y, x = np.where(self.mArr == label)
            z0, z1 = np.min(z), np.max(z)
            xyxy = ndA([np.min(x), np.min(y), z0,
                             np.max(x), np.max(y), z1])
            # xyz1 = ndA([np.max(x), np.max(y), z1])
            if label==255:
                level = LBLVS[29]
            else:
                level = LBLVS[label]
            self.Dic['lvBbx'][level]=xyxy
            print(level)
            for z in range(z0, z1 + 1):
                try:
                    bBx, pBbx, pxSum = self.get_2d_bbox(self.mArr[z],label)
                    self.Dic['bBx'][level][z] = bBx
                    self.Dic['pBbx'][level][z] = pBbx
                    self.Dic['pxSum'][level][z] = pxSum
                except:
                    print(f'No 2D bbox found for slice {z-z0}')
                    self.Dic['bBx'][level][z] = None
                    self.Dic['pBbx'][level][z] = None
                    self.Dic['pxSum'][level][z] = None

# 正侧位最大值投射
```

#ac  <a id='maxProj'></a>

### [[maxProj]]
```python
def maxProj(imArr, xy='x', max=True):
    imArr = isNdaU8(imArr)
    if xy == 'y':
        yArr = np.copy(imArr)
        yArr = np.transpose(yArr, (1, 0, 2))
        # yArr = np.flip(yArr, axis=0)
        yArr2d=None
        if max:
            yArr2d= np.max(yArr,
                axis=0).astype(np.uint8)
            y = np.where(yArr2d>0)[1]
            y0, y1 = y.min(), y.max()
            # if xy == 'kt':
            #     yArr2d = yArr2d[:,
            #         (y0-x0)//2:(y0+x0)//2]
            #     yArr   = imArr[:,:,
            #         (y0-x0)//2:(y1+x1)//2]
            # else:
            yArr2d = yArr2d[:,y0:y1+1]
            yArr   = imArr[:,:,y0:y1+1]
        return yArr, yArr2d
    elif xy == 'x':
        xArr = np.copy(imArr)
        xArr = np.transpose(xArr, (2, 0, 1))
        # xArr = np.flip(xArr, axis=0)
        xArr2d = None
        if max:
            xArr2d = np.max(xArr,
                axis=0).astype(np.uint8)
            # print(xArr.shape)
            x = np.where(xArr2d>0)[1]
            x0, x1 = x.min(), x.max()
            xArr2d = xArr2d[:, x0:x1+1]
            xArr   = imArr[:, x0:x1+1]
            print(f'{xArr.shape=}')
        return xArr, xArr2d
    elif xy == 'kt':
        return maxProj(maxProj(imArr, max=max, xy='x')[0], max=max, xy='y')
    else:
        raise ValueError('Invalid xy: %s' % xy)


```

### [[cutBbx]]

```python
def cutBbx( img,
            bBx   = None,
            cnnLg = True):
    ''' 框裁图
        参:
            img  : 图像
            bBx  : 框(xyxy)
            cnnLg: 最切否
        返:
            cImg: 裁图
    '''
    img         = isImg255(img)
    if  bBx    is None:
        bBx     = getBbox(img)
    dim         = img.GetDimension()
    if   dim   == 2:
        cIm     = img[  bBx[0]:bBx[2],
                        bBx[1]:bBx[3]]
    elif dim   == 3:
        cIm     = img[  bBx[0]:bBx[3],
                        bBx[1]:bBx[4],
                        bBx[2]:bBx[5]]
    if cnnLg:
        img     = getLargestCnn(img)
        cbBx    = getBbox(img)
        cIm, cbBx\
                = cutBbx(img, cbBx, False)
        bBx     = [ (ndA(bBx) + cbBx[:3]*2).tolist(),
                    bBx][bBx is None]
    return        cIm, bBx

```

### [[getBbox]]
```python
def getBbox(img,
            pad = 0)-> list:
    """获取2D和3D图像的bBox
    参: img: 图像
        pad: 填充
    返: bBx: xyxy或xyzxyz
    """
    img         = isImg01(img)
    dim         = img.GetDimension()
    ls          = sitk.LabelShapeStatisticsImageFilter()
    ls          .Execute(img)
    bb          = np.array(ls.GetBoundingBox(1))
    ini, size   = bb[:dim], bb[dim:]
    fin         = ini + size
    if pad     != 0:
        ini, fin          \
                = padBbx(img, pad, ini, fin)
    bBx         = ini.tolist()\
                + fin.tolist()
    return        bBx

```

### [[padBbx]]
```python
def padBbx(img, pad, ini, fin):
    dim        = img.GetDimension()
    siz        = ndA(img.GetSize())
    ini       -= pad
    fin       += pad
    ini[ini<0] = 0 # 防止ini小于0
    for i     in range(dim): # 防止fin超出图像的范围
        fin[i] = min(fin[i], siz[i])
    return       ini, fin

```

#ac  <a id='dataIso'></a>

### [[dataIso]]
```python
def dataIso(img_nib, msk_nib, ctdList, spc=(1,1,1), axes='IPL'):
    # Resample and Reorient data
    if spc is None:
        spc = img_nib.header.get_zooms()
        mSpc = msk_nib.header.get_zooms()
        img_iso = img_nib
        msk_iso = [msk_nib, resample_mask_to(msk_nib, img_iso)][mSpc==spc]
        ctds = [ctdList, rescale_centroids(ctdList, img_nib, spc)][mSpc==spc]
    else:
        img_iso = resample_nib(img_nib, voxel_spacing=spc, order=3) # order是插值方法, 3是三次样条插值
        msk_iso = resample_nib(msk_nib, voxel_spacing=spc, order=0) # order是插值方法, 0是最近邻插值
        ctds = rescale_centroids(ctdList, img_nib, spc) # 质心缩放
    img_iso = reorient_to(img_iso, axcodes_to=axes)
    msk_iso = reorient_to(msk_iso, axcodes_to=axes)
    ctds = reorient_centroids_to(ctds, img_iso)
    return img_iso, msk_iso, ctds

```

#ac  <a id='samShow'></a>

### [[samShow]]
```python
def samShow(img=None, msk=None, boxes=None,
            bxCapt=['',''], clsDic=LBLVS,                   # points=None,
            fSize=(10, 10), dpi = 96, margin=.5, pad=None, opacity=0.4):
    def __showMsk(masks=msk, ax=None, opacity=opacity, clsDic=clsDic, pad=pad):
        labels = np.unique(masks)[1:]
        print(labels)
        # if labels > 0: # or not np.isnan(labels):
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
            if boxes is not None:
                x0, y0, x1, y1 = boxes
            else:
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

    def __showPs(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

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
                # plt.gca().patches.clear()
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
p2p_dst = lambda p, p1:\
            np.linalg.norm(p - p1)

p2p_nor = lambda p, p1:\
            (p1 - p)\
        /   p2p_dst(p, p1)

p2p_lin = lambda p, p1, drt, length:\
            (p2p_nor(p, p1)\
            if drt is None
            else drt)\
        /   np.linalg.norm(drt)\
        *   (p2p_dst(p,p1)
            if length is None
            else length)\
        +   p

```

#ac  <a id='p2pExLine'></a>

### [[p2pExLine]]
```python
def p2pExLine(p,
        p1       =None,
        drt      =None,
        len      =0):
    ''' 点点成线
    有p1无drt结果为点点成延长线
    无p1有drt结果为点点连线
    参:
        p: np.ndarray 点
        p1: np.ndarray 点
        drt: np.ndarray 方向
        len: float 长度
    返:
        px: np.ndarray 点
    '''
    p            = ndA(p)
    if  (p1     != None).all:
        p1       = ndA(p1)
        drt      = p2p_nor(p, p1)
        if  len != 0:
            len += p2p_dst(p, p1)
    drt          = ndA(drt)
    if  len     == 0 and p1 is None:
        len     *= drt
    px           = np.linalg.norm(drt)\
                 * len
    return px

```

#ac  <a id='obb2d'></a>

### [[obb2d]]
```python
def obb2d(imArr,
        isPs   = True,
        show   = False
                    ):
    if isPs   is False:                 # 是否是坐标集
        imArr  = isNdaU8(imArr)
        assert imArr.ndim == 2, "Input must be 2D array"
        x, y   = np.where(imArr > 0)
        ps     = np.vstack([x, y]).T
    else:
        ps     = imArr
    ps         = ndA(ps)
    ps         = ps - np.mean(ps, axis=0)
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
        zwShow( [imArr, None][isPs*1],
            pLs=cns,
            pss=[None, [ps]][isPs*1] )
    return cn

    # xyzList = [xDst, yDst]
    # cnList = [xDs, yDs]

    # zip_a_b = zip(xyzList,cnList)
    # sorted_zip = sorted(zip_a_b, key=lambda x:x[0], reverse=True)
    # result = zip(*sorted_zip)

    # xyz, cns = [list(x) for x in result]
    # return

```

#ac  <a id='prelvCT'></a>

### [[prelvCT]]
```python
def prelvCT(imgF,
          mskF,
          lbCls=LBLVS,
          lbs=[15,24],
          sam_model=None,
          device='cuda:0',
          imSize=256, # 256|224
          imgEncoder=None,
          npz=''):
    ''' CT和标签的预处理
    读取CT图像和标签，返回预处理后的CT图像和标签
        以及imgEmbeds和inputImgs
    参:
        imgF: str 图像文件路径
        mskF: str 标签文件路径
        lbCls: dict 标签类别
        lbs: list 标签类别
        sam_model: nn.Module 模型 opt
        device: str 设备
        imSize: int 图像尺寸
        imgEncoder: nn.Module 图像编码器
        npz: str npz文件路径
    返:
        img: np.ndarray 图像
        msk: np.ndarray 标签
        imgEmbeds: np.ndarray 图像嵌入 opt
        inputImgs: np.ndarray 输入图像 opt
    '''
    imgData = nib.load(imgF)
    lblData = nib.load(mskF)
    a_min, a_max = 0, 255
    imgIPL = nio.reorient_to(imgData, axcodes_to='IPL').get_fdata()
    imArr = np.clip(imgIPL, a_min, a_max) # 将图像的值限制在a_min和a_max之间
    imArr[imgIPL==0] = 0 # 将图像中的0值像素置为0
    imArr = np.uint8(imArr) # 转为uint8
    _, h, w = imArr.shape

    mskIPL = nio.reorient_to(lblData, axcodes_to='IPL').get_fdata()
    mskArr = mskIPL.astype(np.float32)

    if lbs == []:
        lbs = np.unique(mskArr)[1:]
    else:
        lbs = np.arange(lbs[0], lbs[1]+1)
    dic = dfDic(dict)
    lbBar = tqdm(lbs, total=len(lbs))
    for lb in lbBar:
        mask = np.uint8(mskArr==lb)
        if np.sum(mask) > 0:
            msk = mask
        else:
            continue
        zs = np.where(msk!=0)[0]
        print(len(zs))
        z0, z1 = np.min(zs), np.max(zs)
        imgs=[]
        msks=[]
        ipImgs=[]
        imgEmds=[]
        for z in range(z0, z1+1):
            img_i = zoom(imArr[z], (imSize / h, imSize / w), order=3)
            # img_i = transform.resize(img_i, (256, 256), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            # 重采样: 保持原图像的像素值, 重采样到256x256, 三次样条插值, 常数填充, 抗锯齿
            img3C = np.stack([img_i, img_i, img_i], axis=-1)
            imgs.append(img3C)
            msk_i = zoom(msk[z], (imSize / h, imSize / w), order=0)
            # msk_i = transform.resize(msk[i], (256, 256), order=0, preserve_range=True, mode='constant', anti_aliasing=False)
            msks.append(msk_i)
            if sam_model is not None:
                ipImg, imgEmbed = imgEmbeder(img_i, sam_model, device, imgEncoder)
                ipImgs.append(ipImg)
                imgEmds.append(imgEmbed)
        if len(imgs)>1:
            lv = lbCls[lb]
            dic[lv] = dict(imgs=np.asarray(imgs), msks=np.asarray(msks))
            print(dic[lv]['imgs'].shape, dic[lv]['msks'].shape)
            if sam_model is not None:
                dic[lv] = dict(imgEmds=np.asarray(imgEmds), ipImgs=np.asarray(ipImgs))
            if npz!='':
                np.savez_compressed(f'{npz}{lv}.npz', dic[lv])
        # count+=1
            # if count>3:
            #     break
    lbBar.close()
    return dic

```

#ac  <a id='imgEmbeder'></a>

### [[imgEmbeder]]
```python
def imgEmbeder(img, sam_model, device, imgEncoder=None ):
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    resize_img = sam_transform.apply_image(img) #Q: 为什么要resize? A: 因为模型的输入是1024*1024的
                                                        # resized_shapes.append(resize_img.shape[:2])
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                                        # model input: (1, 3, 1024, 1024)
    ipImg = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    print(ndA(ipImg).shape)
    assert ipImg.shape == (1, 3, sam_model.imgEncoder.img_size, sam_model.imgEncoder.img_size), 'input image should be resized to 1024*1024'
    # input_imgs.append(input_image.cpu().numpy()[0])
    if imgEncoder is not None:
        embedding = imgEncoder(ipImg)
    else:
        embedding = sam_model.image_encoder(ipImg)
    print(embedding.shape)
    return ipImg, embedding
```

#ac  <a id='getLargestCnn'></a>

### [[getLargestCnn]]
```python
def getLargestCnn(img, bl = 1):
    img = isImg255(img)
    cnn = sitk.ConnectedComponent(img)
    rbl = sitk.RelabelComponent(cnn)
    return sitk.Mask(img, rbl == bl)
```

#ac  <a id='pxlNumThrange'></a>

### [[pxlNumThrange]]
```python
def pxlNumThrange(
            img: sitk.Image,
            thRange: tuple = (50, 255, 5),
            show: bool = False
            )->np.ndarray:
    # 计算每个阈值下的像素总数
    pxNum = []
    thrange = np.arange(*thRange)
    for thr in thrange:
        sitk_seg = sitk.BinaryThreshold(
            img,
            lowerThreshold=int(thr),
            upperThreshold=thRange[1],
            insideValue=255, outsideValue=0
                                        )
        pxlNum = sitk.GetArrayFromImage(sitk_seg>thRange[0]).sum()
        # pixel_count = sitk.GetArrayFromImage(sitk_seg).sum() # 计算像素总数, 是前景像素的总数
        pxNum.append(pxlNum)
    if show:
        plt.plot(thrange, np.asarray(pxNum));
        plt.xlabel('Threshold');
        plt.ylabel('Pixel Count');
        plt.title('Threshold vs. Pixel Count');
    return thrange, np.asarray(pxNum)
from scipy.interpolate import interp1d # 用于插值的函数
from scipy.signal import argrelextrema # 寻找局部最大值和最小值的函数

```

#ac  <a id='getThre'></a>

### [[getThre]]
```python
def getThre(img: sitk.Image,
            thRange: tuple = (50, 255, 10),
            )->int:
    ''' 最佳阈值
    通过像素数和阈值的关系, 找到最佳阈值, 源于'kneed'的部分代码
        - (https://github.com/arvkevi/kneed...)
        ```
        Finding a “Kneedle” in a Haystack: Detecting Knee Points in System Behavior Ville Satopa † , Jeannie Albrecht† , David Irwin‡ , and Barath Raghavan§ †Williams College, Williamstown, MA ‡University of Massachusetts Amherst, Amherst, MA § International Computer Science Institute, Berkeley, CA
        ```
    '''

    def __pxlNumThrange(img, thRange):
        # 计算每个阈值下的像素总数
        pxSum      = []
        thrange    = np.arange(*thRange)
        for thr in thrange:
            skSeg  = sitk.BinaryThreshold(img,
                        int(thr), thRange[1],
                        255, 0)
            pxs    = sitk.GetArrayFromImage(
                        skSeg>thRange[0]).sum()
            pxSum.append(pxs)
        return       thrange, ndA(pxSum)

    x,y            = __pxlNumThrange(img, thRange)

    uspline        = interp1d(x, y)
    Ds_y           = uspline(x)

    x_normalized   =  (x - min(x))\
                    / (max(x) - min(x))
    y_normalized   =  (Ds_y - min(Ds_y))\
                    / (max(Ds_y) - min(Ds_y))
    y_difference   =  y_normalized\
                    + x_normalized

    indMin         = argrelextrema(y_difference,
                        np.less)[0]
    # if threshold_range[1] - threshold_range[0] > 5:
    # return getThre(img, (opt_threshold-5, opt_threshold+5))
    return           int(x[indMin][0])

```

#ac  <a id='getBoneSeg'></a>

### [[getBoneSeg]]
```python
def getBoneSeg( imgs,
                thrd=None,
                roi=True,
                show=True
                ):
    """
    """
    imgs       = isImg255(imgs)
    bBx        = ndA(imgs.GetSize())\
                *ndA(imgs.GetSpacing())
    if thrd is None:
        thrd0  = getThre(imgs)
        thrd   = getThre(imgs,
                    (thrd0-5, thrd0+5, 1))
        print(f'阈值为{thrd}')
    sitk_seg   = sitk.BinaryThreshold(
                    imgs, thrd,
                    255, 255, 0)
    sitk_open  = sitk.BinaryMorphologicalOpening(
                    sitk_seg != 0,
                    [3,3,3])
    sitk_open  = getLargestCnn(sitk_open)
    array_mask = sitk.GetArrayFromImage(sitk_open)\
                -sitk.GetArrayFromImage(sitk_seg)
    sitk_mask  = reImg(
                    array_mask,
                    sitk_seg)
    sitk_mask  = sitk.Median(sitk_mask)
    cnn = getLargestCnn(sitk_mask)
    if roi:
        cnn, bBx = cutBbx(cnn, cnnLg=False)
    gsImg      = sitk.RecursiveGaussian(cnn)
    # gsImg = sitk.Cast(gsImg, sitk.sitkUInt8)
    if show:
        skShow(gsImg, 'X')
    return gsImg, bBx

```

#ac  <a id='reImg'></a>

### [[reImg]]
```python
def reImg(mkArr, img):
    reIm = sitk.GetImageFromArray(mkArr)
    reIm.SetOrigin(img.GetOrigin())
    reIm.SetSpacing(img.GetSpacing())
    reIm.SetDirection(img.GetDirection())
    return reIm




```

#ac  <a id='projPla'></a>

### [[projPla]]
```python
def projPla(ps: np.ndarray,
            pm =ndA([0,0,0]),
            nm =ndA([0,1,1]),
            p0 =None,
            ) -> np.ndarray:
    ''' projPla 投影到平面
    将点投影到平面
    参:
        ps: 点集 (n,3)
        pm: 平面上一点 (3,)
        nm: 平面法向量 (3,)
        p0: 和pm的连线且垂直于nm的点 (3,)
    返:
        投影点 (n,3)
    '''
    assert (nm is not None)\
        != (p0 is not None),\
    "nm和p0是二选一"
    ps = np.asarray(ps)
    pm = np.asarray(pm)
    nm = np.asarray(nm)\
            if nm is not None\
            else\
         (p0 - pm)\
        /np.linalg.norm(p0 - pm)
    # 点到平面距离
    ds = np.dot((ps - pm), nm)\
        /np.dot(nm       , nm)
    return ps - ds * nm

```

#ac  <a id='zwShow'></a>

### [[zwShow]]
```python
def zwShow(
        imgs  = None,
        ps   = None,
        pLs  = None,
        vLs  = None,
        hLs  = None,
        bxs  = None,
        # ax   = None
        ):
    one2Ls = lambda data, dim: [data]\
                if ndA(data).ndim == dim\
                else data
    def __tpl2Ls(data, dim=2):
        data = one2Ls(data, dim=dim)
        if isinstance(data, tuple):
            datas = []
            for d in ndA(data).tolist():
                datas += [d]
            return ndA(datas)
        else:
            return data

    if imgs is not None:
        if isinstance(imgs, sitk.Image):
            imgs = sitk.GetArrayFromImage(imgs)
        imgs = __tpl2Ls(imgs)
        for i, img in enumerate(imgs):
            if i == 0:
                exV, exH = img.shape
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img, cmap= 'gray', alpha=0.3)
    # 绘制点
    if ps is not None:
        ps  = __tpl2Ls(ps)
        if img is None:
            exH = np.max(ps, axis=0)
            exV = np.max(ps, axis=1)
        # for ps in pss:
        for p in ps:
            plt.plot(p[:,0], p[:,1], 'b.')
    # 绘制纵线
    if vLs is not None:
        vLs = __tpl2Ls(vLs)
        for vl in vLs:
            plt.vlines(vl, 0, exV, colors='g')
    # 绘制横线
    if hLs is not None:
        hLs = __tpl2Ls(hLs)
        for hl in hLs:
            plt.hlines(hl, 0, exH, colors='y')
    # 绘制线
    if pLs is not None:
        pLs = __tpl2Ls(pLs)
        for pL in pLs:
            plt.plot(pL[:, 0], pL[:, 1], 'r--')
    # 绘制矩形
    if bxs is not None:
        bxs = __tpl2Ls(bxs, dim=1)
        def __getBxs(bx, ax):
            y,x,y1,x1 = bx
            ax.add_patch(plt.Rectangle((x,y),
                x1-x, y1-y, edgecolor='r',
                facecolor=(0,0,0,0), lw=1.5))
        for bx in bxs:
            __getBxs(bx, plt.gca())
    plt.axis('off')
    plt.show()

```

#ac  <a id='psFitLine'></a>

### [[psFitLine]]
```python
def psFitLine(ps):
    ''' 点集拟合直线
    参: ps: 点集
    返: p0, p1, cP: 起点, 终点, 质心
    '''
    # 已知直线上的始点方向长度求直线的终点
    pp  = lambda p0, drt, length:\
                drt\
            /   np.linalg.norm( drt)\
            *   length\
            +   p0
    #======================================
    ps  = ndA(ps)
    cP  = np.mean(ps,          # 质心
            axis = 0)
    dt  = ps-cP                # 减质心
    vv  = np.linalg.svd(dt)[2] # 奇异值分解
    #? 最大奇异值对应的右奇异向量...😓🙅🏻‍♀️不知道啥意思
    drt = vv[0]                # 矢量
    len = ps.max()             # 线的长度
    p0  = pp(cP, drt, -len/2)  # 线的起点
    p1  = pp(cP, drt,  len/2)  # 线的终点
    return p0, p1, cP
```
```python
DATA     = 1
dataPth  = f'/Users/liguimei/Documents/GitHub/SAMed/datasets/workDir/data/'
img502   = dataPth + f'sub-verse502_dir-iso_ct.nii.gz'
# msk502   = dataPth + f'sub-verse502_dir-iso_seg-vert_msk.nii.gz'
img768   = dataPth + f'verse768_CT-ax.nii.gz'
# msk768   = dataPth + f'verse768_CT-ax_seg.nii.gz'
# skMsk768 = dataPth + f'/skImgs/mskIso768.nii.gz'
# skMsk502 = dataPth + f'/skImgs/mskIso502.nii.gz'
img518   = dataPth + f'sub-verse518_dir-ax_ct.nii.gz'
```
```python
# im502 = readImgSk(img502)
# im768 = readImgSk(img768)
im518 = readImgSk(img518)
# msk5 = readImgSk(msk502, msk=True)
# msk7 = readImgSk(msk768, msk=True)
# ims502  = readImgSk(img502, img3C=True)
# ims768  = readImgSk(img768, img3C=True)
```
```python
getBbox(im518)
```
```python
SANDBOX = 1
```
```python title="\u8ddf\u8e2ax0y0z0, \u8bb0\u5f55sum, \u6700\u7ec8\u622a\u53d6xyx\u548cx0y0z0\u7684\u5dee\u503c"
# bImg5, bBx5 = getBoneSeg(img5)
# bImg7, bBx7 = getBoneSeg(img7)
# # #%%
# # import shelve
# #%%
# output = f'/Users/liguimei/Documents/GitHub/SAMed/2Dto3D/output/'
# op5 = output + 'output5'
# op7 = output + 'output7'

# bArr5 = isNdaU8(bImg5)
# d = shelve.open(op5)
# d['bArr'] = bArr5
# d['bBx'] = bBx5
# d.close()
# bArr7 = isNdaU8(bImg7)
# d = shelve.open(op7)
# d['bArr'] = bArr7
# d['bBx'] = bBx7
# d.close()
```
```python

```
```python

```
```python
# msk5 = sitk.GetImageFromArray(msk5[335])
```

#ac  <a id='getCnn'></a>

### [[getCnn]]
```python
def getCnn(img, minVol=33):
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img)
    cnn = sitk.ConnectedComponent(img)
    lbCnns = sitk.RelabelComponent(cnn)[1:]
    msks = []
    for lb in lbCnns:
        msk = sitk.Mask(img, lbCnns == lb)
        mskVol = sitk.GetArrayFromImage(msk>0).sum()
        if mskVol > minVol:
            msks.append(msk)
    return msks
```

```python
# mks, sizes = getCnn(bImg5[:,:,335])
```
```python
import numpy as np
import SimpleITK as sitk

```

#ac  <a id='crop_with_plane'></a>

### [[crop_with_plane]]
```python
def crop_with_plane(image, p0, norm=None, p1=None):
    """
    Crop a 3D image with a plane defined by a point and normal vector.

    Args:
        image (SimpleITK.Image): The input 3D image to be cropped.
        point (tuple): A tuple of three floats representing the point on the plane.
        normal (tuple): A tuple of three floats representing the normal vector of the plane.

    Returns:
        SimpleITK.Image: The cropped 3D image.
    """
    # Convert point and normal to numpy arrays
    p0 = np.asarray(p0)
    norm = np.asarray(norm)

    # Get the size and spacing of the input image
    size = ndA(image.GetSize())
    spacing = ndA(image.GetSpacing())

    # Calculate the distance from each voxel to the plane
    x, y, z = np.meshgrid(
        np.arange(size[0]), np.arange(size[1]), np.arange(size[2]), indexing='ij')
    coords = np.stack((x, y, z), axis=-1)
    distances = np.abs(np.dot(coords - p0, norm))

    # Find the minimum and maximum distances
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    # Calculate the new size and origin of the cropped image
    new_size = np.ceil((max_distance - min_distance) / spacing).astype(int)
    new_origin = p0 - norm * min_distance - 0.5 * (new_size - 1) * spacing

    # Create a cropping region based on the new size and origin
    new_region = sitk.ImageRegion(new_size.tolist(), new_origin.tolist())

    # Crop the input image with the new region
    cropped_image = sitk.RegionOfInterest(image, new_region)

    return cropped_image
```

```python
import SimpleITK as sitk

```

#ac  <a id='show_point'></a>

### [[show_point]]
```python
def show_point(image, point, label_value=1):
    """
    Show a point in a 3D image by setting a pixel to a specified label value.

    Args:
        image (SimpleITK.Image): The input 3D image.
        point (tuple): A tuple of three integers representing the point to be shown.
        label_value (int): The label value to use for the point.

    Returns:
        SimpleITK.Image: The modified 3D image.
    """
    # Create a new image with the same size and spacing as the input image
    label_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    label_image.SetSpacing(image.GetSpacing())
    label_image.SetOrigin(image.GetOrigin())
    label_image.SetDirection(image.GetDirection())

    # Set the pixel at the specified point to the label value
    label_image.SetPixel(point, label_value)

    # Combine the input image and label image using the MaximumImageFilter
    max_filter = sitk.MaximumImageFilter()
    max_filter.SetInput1(image)
    max_filter.SetInput2(label_image)
    combined_image = max_filter.Execute()

    return combined_image
```

```python

import numpy as np
import SimpleITK as sitk

```

#ac  <a id='show_plane_and_points'></a>

### [[show_plane_and_points]]
```python
def show_plane_and_points(image, p0, p1, label_value=1):
    """
    Show a plane and two points in a 3D image, and crop the image with the plane.

    Args:
        image (SimpleITK.Image): The input 3D image.
        p0 (tuple): A tuple of three floats representing a point on the plane.
        p1 (tuple): A tuple of three floats representing a point not on the plane.
        label_value (int): The label value to use for the plane and points.

    Returns:
        SimpleITK.Image: The modified 3D image.
    """
    # Convert points to numpy arrays
    p0 = ndA(p0)
    p1 = ndA(p1)

    # Calculate the normal vector of the plane
    normal = p1 - p0
    normal /= np.linalg.norm(normal)

    # Calculate the distance from each voxel to the plane
    size = ndA(image.GetSize())
    spacing = ndA(image.GetSpacing())
    x, y, z = np.meshgrid(
        np.arange(size[0]), np.arange(size[1]), np.arange(size[2]), indexing='ij')
    coords = np.stack((x, y, z), axis=-1)
    distances = np.abs(np.dot(coords - p0, normal))

    # Find the minimum and maximum distances
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    # Calculate the new size and origin of the cropped image
    new_size = np.ceil((max_distance - min_distance) / spacing).astype(int)
    new_origin = p0 - normal * min_distance - 0.5 * (new_size - 1) * spacing

    # Create a cropping region based on the new size and origin
    new_region = sitk.ImageRegion(new_size.tolist(), new_origin.tolist())

    # Crop the input image with the new region
    cropped_image = sitk.RegionOfInterest(image, new_region)

    # Create a new image with the same size and spacing as the input image
    label_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    label_image.SetSpacing(image.GetSpacing())
    label_image.SetOrigin(image.GetOrigin())
    label_image.SetDirection(image.GetDirection())

    # Set the pixels at the specified points to the label value
    label_image.SetPixel(tuple(p0.round().astype(int)), label_value)
    label_image.SetPixel(tuple(p1.round().astype(int)), label_value)

    # Set the pixels in the plane to the label value
    plane_mask = np.abs(np.dot(coords - p0, normal)) < 1e-6
    label_image[plane_mask] = label_value

    # Combine the input image and label image using the MaximumImageFilter
    max_filter = sitk.MaximumImageFilter()
    max_filter.SetInput1(cropped_image)
    max_filter.SetInput2(label_image)
    combined_image = max_filter.Execute()

    return combined_image
```

```python
```

#ac  <a id='center_of_mass'></a>

### [[center_of_mass]]
```python
def center_of_mass(input, labels=None, index=None):
    """
    计算标签处数组值的质心。

    参数
    ----------
        input: ndarray
            用于计算质心的数据。群众也可以
            是积极的还是消极的。
        labels: ndarray, 可选
            “input”中对象的标签, 由“ndimage.label”生成。
            仅与“索引”一起使用。维度必须与“输入”相同。
        index : int 或 int 序列，可选
            要计算质心的标签。如果没有指定，
            所有标签的组合质心大于零
            将被计算。仅与“标签”一起使用。

        退货
        --------
        center_of_mass ：元组或元组列表
            质心坐标。

    Examples
    --------
    >>> import numpy as np
    >>> a = ndA(([0,0,0,0],
    ...               [0,1,1,0],
    ...               [0,1,1,0],
    ...               [0,1,1,0]))
    >>> from scipy import ndimage
    >>> ndimage.center_of_mass(a)
    (2.0, 1.5)

    Calculation of multiple objects in an image

    >>> b = ndA(([0,1,1,0],
    ...               [0,1,0,0],
    ...               [0,0,0,0],
    ...               [0,0,1,1],
    ...               [0,0,1,1]))
    >>> lbl = ndimage.label(b)[0]
    >>> ndimage.center_of_mass(b, lbl, [1,2])
    [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]

    Negative masses are also accepted, which can occur for example when
    bias is removed from measured data due to random noise.

    >>> c = ndA(([-1,0,0,0],
    ...               [0,-1,-1,0],
    ...               [0,1,-1,0],
    ...               [0,1,1,0]))
    >>> ndimage.center_of_mass(c)
    (-4.0, 1.0)

    If there are division by zero issues, the function does not raise an
    error but rather issues a RuntimeWarning before returning inf and/or NaN.

    >>> d = ndA([-1, 1])
    >>> ndimage.center_of_mass(d)
    (inf,)
    """
    normalizer = sum(input, labels, index) # 计算标签处数组值的质心。
    grids = np.ogrid[[slice(0, i) for i in input.shape]] # 生成一个多维结构数组

    results = [sum(input * grids[dir].astype(float), labels, index) / normalizer
               for dir in range(input.ndim)] # 计算标签处数组值的质心。

    if np.isscalar(results[0]): # 如果是标量
        return tuple(results)

    return [tuple(v) for v in ndA(results).T]
```
```python


```

#ac  <a id='plaCrop'></a>

### [[plaCrop]]
```python
def plaCrop(image, plane_point, plane_normal):

    # 获取图像信息
    size = ndA(image.GetSize())
    spacing = ndA(image.GetSpacing())
    origin = ndA(image.GetOrigin())

    # 计算裁剪平面的法向量和点
    normal = ndA(plane_normal)
    point = ndA(plane_point)

    # 计算交点坐标
    x = origin + spacing * size # 图像对角线上的点

    intersection =  (np.dot(x - point, normal)\
                    /np.dot(normal, normal)) \
                    * normal\
                    + point # 计算交点坐标

    # 确定裁剪边界
    min_pt = np.min(intersection)
    max_pt = np.max(intersection)

    # 执行裁剪
    output = sitk.Crop(image,
                       [int(min_pt[0]), int(max_pt[0])],
                       [int(min_pt[1]), int(max_pt[1])],
                       [int(min_pt[2]), int(max_pt[2])])

    return output
#%%#%%
import numpy as np

```

#ac  <a id='get_obb'></a>

### [[get_obb]]
```python
def get_obb(image):
  """
  Calculate oriented bounding box of non-zero pixels in image.

  Args:
    image: 2D numpy array

  Returns:
    points: OBB corner points
    center: OBB center point
    size: OBB width and height
    angle: OBB angle in radians
  """

  # Validate input
  assert image.ndim == 2, "Input must be 2D array"

  # Get non-zero pixel coordinates
  x, y = np.where(image > 0)
  points = np.vstack([x, y]).T

  # Center the points
  centered = points - points.mean(axis=0)

  # Calculate covariance matrix
  cov = np.dot(centered.T, centered) / len(points)

  # Get principal components
  values, vectors = np.linalg.eig(cov)

  # Sort eigenvalues and eigenvectors
  idx = np.argsort(values)[::-1]
  values = values[idx]
  vectors = vectors[:,idx]

  # Validate principal components
  assert np.linalg.matrix_rank(vectors) == 2

  # Project points to principal component space
  rotated = np.dot(centered, vectors)

  # Get OBB edges
  edges = vectors

  # Calculate OBB boundaries
  xmin = rotated[:,0].min()
  xmax = rotated[:,0].max()
  ymin = rotated[:,1].min()
  ymax = rotated[:,1].max()

  # Get OBB width, height and center
  cx = (xmin + xmax) / 2
  cy = (ymin + ymax) / 2
  w = xmax - xmin
  h = ymax - ymin

  # Get corner points
  corners = np.array([
    [cx - w/2, cy - h/2],
    [cx - w/2, cy + h/2],
    [cx + w/2, cy - h/2],
    [cx + w/2, cy + h/2]
  ])

  # Rotate corners back to original space
  points = np.dot(corners, edges.T)

  # Calculate OBB angle
  angle = np.arctan2(edges[0,0], edges[1,0])

  return points, (cx, cy), (w, h), angle

END = 1
```
