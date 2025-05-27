"""ctPj.py: 投影轮廓计算和内部网格点集处理模块

__author__ = "Jumbo Jing"
"""

import numpy as np
import vtk
from scipy.spatial import cKDTree as kdT
from scipy.ndimage import binary_dilation as dila, binary_erosion as erod, label as scLb
from functools import lru_cache
from typing import Optional, Union, Tuple
import slicer

# 从pppUtil导入必要的函数（假设已经存在）
from pppUtil import *

class CtPj:
    """计算投影轮廓和内部网格点集的类（优化版）"""
    
    def __init__(self, pds, pjNor=None, cJp=None, sp=None, clean: bool = False, 
                 eSp=None, mNam: str = '', rad: float = 1/3, thr=None, 
                 grid_step: float = 1/3, max_iterations: int = 50):
        """
        初始化投影轮廓计算类
        
        参数:
            pds: 输入点集数据
            pjNor: 投影方向法向量或包含投影参数的数组
            cJp: 投影中心点
            sp: 种子点
            clean: 是否清理投影结果
            eSp: 额外种子点
            mNam: 模型名称前缀
            rad: 半径参数
            thr: 阈值参数
            grid_step: 网格步长
            max_iterations: 最大迭代次数
        """
        # 输入验证
        self.pds = getArr(pds)
        if self.pds is None or len(self.pds) == 0:
            raise ValueError("输入点集为空")
        
        self.pd = getPd(pds)
        self.mNam = mNam
        self.r = rad
        self.thr = thr if thr is not None else self.r * 0.1
        self.eSp = eSp
        self.grid_step = grid_step
        self.max_iterations = max_iterations
        
        # 设置投影中心
        self.cJp = ndA(cJp if cJp is not None else self.pds.mean(0))
        self.sp = sp
        
        # 缓存字典
        self._cache = {}
        
        # 处理投影方向和计算投影点
        self._process_projection_direction(pjNor)
        
        # 处理投影点集
        self._process_projected_points(clean)
        
        # 初始化网格点
        self._initialize_grid_points()
    
    def _process_projection_direction(self, pjNor):
        """处理投影方向"""
        if pjNor is not None:
            pjNor = ndA(pjNor)
            if pjNor.ndim == 2:
                if len(pjNor) == 3:
                    # 包含投影参考点、法向量和侧向量
                    _, self.pjNor, self.sDrt = self.ras = pjNor
                else:
                    # 包含投影点和法向量
                    self.cJp, self.pjNor = pjNor
                    self.ras = pjNor
                    self.sDrt = None
            else:
                self.pjNor = pjNor
                self.ras = None
                self.sDrt = None
            
            # 计算投影点集
            self.pjPs_ = psPj(self.pds, (self.cJp, self.pjNor), 
                             mNam=sNam(self.mNam, 'pjPs_'), exTyp=None)
        else:
            self.pjPs_ = self.pds
            self.pjNor = psFitPla(self.pjPs_)
            self.ras = None
            self.sDrt = None
        
        # 确保法向量归一化
        self.pjNor = uNor(self.pjNor)
    
    def _process_projected_points(self, clean):
        """处理投影点集"""
        if clean:
            # 清理投影结果
            connected_component = cnnEx(self.pjPs_)
            self.pjPs = cleanPj(connected_component, mNam=sNam(self.mNam, 'pjPs'))
        else:
            # 直接使用最大连通区域
            self.pjPs = getArr(cnnEx(self.pjPs_, exTyp='Lg'))
        
        if len(self.pjPs) == 0:
            raise ValueError("投影后点集为空")
    
    def _initialize_grid_points(self):
        """初始化网格点和掩码"""
        try:
            # 生成网格点
            projection_reference = self.ras if self.ras is not None else self.pjNor
            self.gps = obGps(self.pjPs, projection_reference, 
                           stp=self.grid_step, mNam=sNam(self.mNam, 'gps'))
            
            # 计算点到投影点集的距离
            _, proj_tree, distance_query = kdOlbs_(self.pjPs, 1.0)
            interior_indicators = distance_query(self.gps)
            
            # 创建掩码（0表示外部，1表示内部）
            grid_mask = np.where(interior_indicators > 0, 0, 1)
            
            # 处理边界
            self._process_boundary_mask(grid_mask)
            
            # 处理种子点
            self._process_seed_points()
            
            # 提取内部点和轮廓点
            self._extract_interior_and_contour_points(proj_tree)
            
        except Exception as e:
            print(f"网格初始化失败: {str(e)}")
            self.inGps = np.array([]).reshape(0, 3)
            self.ctPs = np.array([]).reshape(0, 3)
            self.msk = np.array([])
    
    def _process_boundary_mask(self, grid_mask):
        """处理边界掩码"""
        boundary_mask = np.zeros_like(grid_mask, dtype=bool)
        boundary_mask[[0, -1], :] = grid_mask[[0, -1], :] == 1
        boundary_mask[:, [0, -1]] = grid_mask[:, [0, -1]] == 1
        
        # 膨胀边界
        structure = np.ones((3, 3), dtype=bool)
        dilated_boundary = dila(boundary_mask, structure, iterations=-1, 
                               mask=grid_mask == 1)
        
        # 更新掩码
        self.msk = grid_mask.copy()
        self.msk[dilated_boundary] = 0
    
    def _process_seed_points(self):
        """处理种子点"""
        if self.sp is not None:
            try:
                seed_index = findPs(self.gps, self.sp)[-1]
                labeled_mask, num_labels = scLb(self.msk)
                
                if num_labels > 0:
                    if isinstance(seed_index, (list, tuple)):
                        # 多维索引
                        seed_label = labeled_mask[seed_index]
                    else:
                        # 一维索引，需要转换为网格坐标
                        grid_shape = labeled_mask.shape
                        if seed_index < np.prod(grid_shape):
                            seed_coords = np.unravel_index(seed_index, grid_shape)
                            seed_label = labeled_mask[seed_coords]
                        else:
                            seed_label = 0
                    
                    if seed_label > 0:
                        self.msk = (labeled_mask == seed_label)
            
            except Exception as e:
                print(f"种子点处理失败: {e}")
    
    def _extract_interior_and_contour_points(self, proj_tree):
        """提取内部点和轮廓点"""
        # 提取内部点
        if self.msk.size > 0:
            flat_mask = self.msk.ravel() if self.msk.ndim > 1 else self.msk
            valid_indices = flat_mask > 0
            
            if len(valid_indices) <= len(self.gps):
                self.inGps = self.gps[valid_indices[:len(self.gps)]]
            else:
                self.inGps = self.gps[valid_indices[:len(self.gps)]]
        else:
            self.inGps = np.array([]).reshape(0, 3)
        
        # 创建内部点模型
        if len(self.inGps) > 0:
            pds2Mod(self.inGps, sNam(self.mNam, 'inGps'))
            
            # 找到对应的轮廓点
            try:
                contour_indices = proj_tree.query(self.inGps, k=1)[1]
                self.ctPs = self.pjPs[contour_indices]
                pds2Mod(self.ctPs, sNam(self.mNam, 'ctPs'))
            except:
                self.ctPs = np.array([]).reshape(0, 3)
        else:
            self.ctPs = np.array([]).reshape(0, 3)
    
    @lru_cache(maxsize=32)
    def _get_cached_result(self, method_name: str, *args):
        """缓存计算结果"""
        return getattr(self, f'_compute_{method_name}')(*args)
    
    def mic(self):
        """计算最大内切圆"""
        if not hasattr(self, 'inGps') or len(self.inGps) == 0:
            print("警告: 没有有效的内部点")
            return np.zeros(3), 0.0, np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        try:
            # 导入最大内切圆计算函数
            from pppUtil import psMic
            
            # 创建轮廓点模型
            pds2Mod(self.ctPs, mNam=sNam(self.mNam, 'ctPs'))
            pds2Mod(self.inGps, mNam=sNam(self.mNam, 'iPs_'))
            
            # 计算最大内切圆
            center, radius, circle_points = psMic(
                self.ctPs, self.inGps, self.pjNor,
                mNam=sNam(self.mNam, 'ctMic')
            )
            
            return center, radius, circle_points, self.ctPs
            
        except Exception as e:
            print(f"计算最大内切圆失败: {str(e)}")
            return np.zeros(3), 0.0, np.array([]).reshape(0, 3), self.ctPs
    
    def edPs(self):
        """计算椭圆弧和终板分离"""
        try:
            # 计算边界点
            internal_points, boundary_points = ctBd_ed(self.msk, self.gps, 9)
            
            if len(internal_points) == 0:
                raise ValueError("未找到有效内部点")
            
            # 创建内部点模型
            pds2Mod(internal_points, mNam=sNam(self.mNam, 'iPs'))
            
            # 反向投影到原始点集
            if hasattr(self, 'pjPs_') and len(self.pjPs_) > 0:
                proj_tree = kdT(self.pjPs_)
                reverse_indices = proj_tree.query(internal_points, k=1)[1]
                reverse_projected = self.pds[reverse_indices]
            else:
                reverse_projected = internal_points
            
            pds2Mod(reverse_projected, mNam=sNam(self.mNam, 'rjPs_'))
            
            # 椎体分割
            superior_points, inferior_points = psLbs(reverse_projected, num=2, rad=2)
            
            if len(superior_points) == 0 or len(inferior_points) == 0:
                raise ValueError("椎体分割失败")
            
            # 确定上下终板
            center_projection = findPs(superior_points, self.cJp)[0]
            separation_plane_point = center_projection - self.pjNor * 2
            
            # 分离上终板
            superior_endplate = dotPlnX(superior_points, 
                                     (separation_plane_point, self.pjNor), 1)
            pds2Mod(superior_endplate, mNam=sNam(self.mNam, 'eSps'))
            
            # 计算椎体方向
            spine_direction = psFitPla(superior_endplate)
            if not np.any(spine_direction):
                raise ValueError("椎体方向计算失败")
            
            # 椎体投影
            vertebral_projection = psPj(internal_points, (self.cJp, spine_direction))
            contour_projection = psPj(self.ctPs, (self.cJp, spine_direction))
            
            # 创建投影模型
            pds2Mod(vertebral_projection, mNam=sNam(self.mNam, 'vbIps'))
            pds2Mod(contour_projection, mNam=sNam(self.mNam, 'Ips'))
            
            # 存储结果用于后续访问
            self.iPs = contour_projection
            
            return (superior_endplate, inferior_points, spine_direction, 
                   vertebral_projection, contour_projection)
            
        except Exception as e:
            print(f"椭圆弧计算失败: {str(e)}")
            empty_result = np.array([]).reshape(0, 3)
            return (empty_result, empty_result, np.array([0, 0, 1]), 
                   empty_result, empty_result)
    
    def sCt(self):
        """计算椎体截面"""
        try:
            # 创建轮廓点模型
            pds2Mod(self.ctPs, mNam=sNam(self.mNam, 'ctPs'))
            
            # 反向投影
            if hasattr(self, 'pjPs_') and len(self.pjPs_) > 0:
                proj_tree = kdT(self.pjPs_)
                reverse_indices = proj_tree.query(self.ctPs, k=1)[1]
                reverse_projected = self.pds[reverse_indices]
            else:
                reverse_projected = self.ctPs
            
            # 平面分离
            projections = (reverse_projected - self.cJp) @ self.pjNor
            positive_mask = projections > 0
            
            filtered_reverse = reverse_projected[positive_mask]
            filtered_contour = self.ctPs[positive_mask]
            
            # 创建模型
            pds2Mod(filtered_reverse, mNam=sNam(self.mNam, 'rjPs'))
            
            return self.inGps, filtered_contour, filtered_reverse
            
        except Exception as e:
            print(f"计算椎体截面失败: {str(e)}")
            empty_result = np.array([]).reshape(0, 3)
            return empty_result, empty_result, empty_result
    
    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'num_original_points': len(self.pds),
            'num_projected_points': len(self.pjPs) if hasattr(self, 'pjPs') else 0,
            'num_grid_points': len(self.gps) if hasattr(self, 'gps') else 0,
            'num_interior_points': len(self.inGps) if hasattr(self, 'inGps') else 0,
            'num_contour_points': len(self.ctPs) if hasattr(self, 'ctPs') else 0,
            'projection_normal': self.pjNor.tolist() if hasattr(self, 'pjNor') else None,
            'projection_center': self.cJp.tolist() if hasattr(self, 'cJp') else None
        }
        return stats
    
    def clear_cache(self):
        """清理缓存"""
        self._cache.clear()
        if hasattr(self, '_get_cached_result'):
            self._get_cached_result.cache_clear()


def ctBd_ed(labels_mask, grid_points, erosion_count=3):
    """
    计算轮廓边界点并删除指定层数
    
    参数:
        labels_mask: 标签掩码
        grid_points: 网格点
        erosion_count: 腐蚀次数
    
    返回:
        内部点, 轮廓点
    """
    mask = (labels_mask == 1).copy()
    
    if mask.size == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # 计算边界
    boundary_mask = np.zeros_like(mask, dtype=bool)
    
    # 水平边界
    if mask.shape[0] > 1:
        boundary_mask[1:] |= (mask[1:] ^ mask[:-1])
        boundary_mask[:-1] |= (mask[:-1] ^ mask[1:])
    
    # 垂直边界
    if mask.shape[1] > 1:
        boundary_mask[:, 1:] |= (mask[:, 1:] ^ mask[:, :-1])
        boundary_mask[:, :-1] |= (mask[:, :-1] ^ mask[:, 1:])
    
    # 边界点属于背景
    boundary_mask &= (labels_mask == 0)
    boundary_indices = np.argwhere(boundary_mask)
    
    # 填充行和列
    if len(boundary_indices) > 0:
        # 按行分组
        row_groups = {}
        col_groups = {}
        
        for x, y in boundary_indices:
            if x not in row_groups:
                row_groups[x] = []
            row_groups[x].append(y)
            
            if y not in col_groups:
                col_groups[y] = []
            col_groups[y].append(x)
        
        # 填充行
        for x, y_coords in row_groups.items():
            if len(y_coords) >= 2:
                y_min, y_max = min(y_coords), max(y_coords)
                mask[x, y_min:y_max+1] = True
        
        # 填充列
        for y, x_coords in col_groups.items():
            if len(x_coords) >= 2:
                x_min, x_max = min(x_coords), max(x_coords)
                mask[x_min:x_max+1, y] = True
    
    # 应用腐蚀
    if erosion_count > 0:
        try:
            return erod_(mask, grid_points, erosion_count)
        except:
            # 备用方案
            pass
    
    # 提取结果点
    if mask.size == grid_points.size // 3:
        flat_mask = mask.ravel()
        interior_points = grid_points[flat_mask]
        
        # 计算轮廓点
        dilated_mask = dila_(mask, 1)
        contour_mask = dilated_mask ^ mask
        contour_points = grid_points[contour_mask.ravel()]
    else:
        # 尺寸不匹配时的处理
        valid_indices = np.arange(min(len(grid_points), mask.size))
        interior_points = grid_points[valid_indices[mask.ravel()[:len(valid_indices)]]]
        contour_points = np.array([]).reshape(0, 3)
    
    return interior_points, contour_points


def cleanPj(projected_polydata, plane_normal=None, radius=0.3, threshold=2, 
           model_name='vtCt'):
    """
    清理投影点云，去除孤立点
    
    参数:
        projected_polydata: 投影后的PolyData
        plane_normal: 平面法向量
        radius: 邻域半径
        threshold: 邻居数量阈值
        model_name: 模型名称
    
    返回:
        清理后的点集
    """
    try:
        if plane_normal is not None:
            # 重新投影到平面
            projected_node = getNod(projected_polydata)
            points = psPj(projected_node, plane_normal)
        else:
            points = getArr(projected_polydata)
        
        # 去除孤立点
        cleaned_points = kdCt_(points, radius, threshold)
        
        # 聚类
        if len(cleaned_points) > 0:
            clustered_points = psLbs(cleaned_points, mNam=model_name)
            return clustered_points
        
        return np.array([]).reshape(0, 3)
        
    except Exception as e:
        print(f"投影清理失败: {e}")
        return getArr(projected_polydata) if projected_polydata else np.array([]).reshape(0, 3)


def kdCt_(contour_points, radius=0.3, threshold=2, center_point=None, 
          search_radius=1.0, model_name=''):
    """
    使用KD树去除平面点集轮廓内的孤立岛点集
    
    参数:
        contour_points: 轮廓点
        radius: 邻域半径
        threshold: 邻居数量阈值
        center_point: 中心点
        search_radius: 搜索半径
        model_name: 模型名称
    
    返回:
        过滤后的点集
    """
    contour_points = getArr(contour_points)
    
    if len(contour_points) == 0:
        return np.array([]).reshape(0, 3)
    
    if center_point is None:
        center_point = np.mean(contour_points, axis=0)
    
    try:
        # 邻域过滤
        neighbor_counts, _, neighbor_query = kdOlbs_(contour_points, radius)
        filtered_points = contour_points[np.array(neighbor_counts) > threshold]
        
        # 中心点过滤
        if len(filtered_points) > 0:
            center_neighbors = neighbor_query(center_point.reshape(1, -1), search_radius)
            
            if center_neighbors[0] > 0:
                final_neighbor_counts = neighbor_query(filtered_points, search_radius)
                final_points = filtered_points[np.array(final_neighbor_counts) > center_neighbors[0]]
            else:
                final_points = filtered_points
        else:
            final_points = filtered_points
        
        if model_name:
            pds2Mod(final_points, mNam=model_name)
        
        return final_points
        
    except Exception as e:
        print(f"KD树过滤失败: {e}")
        return contour_points
    

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

# ========== ctPj相关API接口整理 ==========
__all__ = [
    'CtPj', 'ctBd_ed', 'cleanPj', 'kdCt_', 'erod_',
    'delBdMsk_', 'dila_', 'psLbs', 'psMic'
]

# ========== 文件结尾注释 ========== 
# 裁切相关API全部集中于本文件，便于统一维护和调用。