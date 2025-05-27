# ========== 裁切相关依赖导入 ==========
import numpy as np
import vtk
import slicer
from pppUtil import *
from vtk.util.numpy_support import numpy_to_vtk

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


def vtkPlnCrop(mPd, fun, refP=None,inPd=False, mNam='', 
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
    funs = ndA(funs)
    clipFun = vtk.vtkImplicitBoolean() # 定义一个合集隐式布尔函数
    clipFun.SetOperationTypeToUnion()
    if funs.ndim == 2:
        clipFun.AddFunction(vtkPln(funs, refP=refP))
    elif funs.ndim == 3:
        for fun in funs:
            clipFun.AddFunction(vtkPln(fun, refP=refP))  
        
    # elif isinstance(fun, vtk.vtkImplicitFunction):
    #     clipFun.AddFunction(fun)
    # elif isinstance(fun, vtk.vtkImplicitBoolean):
    #     clipFun.AddFunction(fun.GetFunction())
    else:
        raise TypeError("Unsupported type for funs: {}".format(type(funs)))
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
        pns: any,  # 🔱 平面|点集
        mPd=None,
        mNam='',
        pdLs=False,
        cPlns=False,
        refP=None,
        **kw):
    '''vtkPlns 生成Vk平面s'''
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

# ========== 裁切相关API接口整理 ==========
__all__ = [
    'vtkPln', 'vtkPlns', 'vtkCut', 'dotCut', 'dotPlnX', 'DotCut',
    'vtkPlnCrop', 'rePln_', 'addPlns', 'vtkCplnCrop', 'vtkPs', 'vtkNors', 'SPln', 'ps_pn', 'dotPn'
]

# ========== 文件结尾注释 ========== 
# 裁切相关API全部集中于本文件，便于统一维护和调用。

