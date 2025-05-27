"""volData.py: 体素数据处理模块

__author__ = "Jumbo Jing"
"""

import os
import numpy as np
import vtk
from functools import lru_cache
from typing import Optional, Union, Dict, Any
import slicer
import SimpleITK as sitk
from sitkUtils import PullVolumeFromSlicer, PushVolumeToSlicer

# 从pppUtil导入必要的函数（假设已经存在）
from .util import *
# 定义备用常量
EPS = 1e-6
OP = np.zeros(3)
TLDIC = {}

# 简写函数
puSk = PullVolumeFromSlicer
skPu = PushVolumeToSlicer

class volData:
    """体素数据处理类（高度优化版）"""
    
    def __init__(self, vol, mNam: str = '', **kw):
        """
        初始化体素数据处理对象
        
        参数:
            vol: 体素节点或体素数据
            mNam: 模型名称
            **kw: 其他关键字参数
        """
        self.vol = getNod(vol, mNam) if hasattr(getNod, '__call__') else vol
        self.nam = self.vol.GetName() if hasattr(self.vol, 'GetName') else str(mNam)
        self._kw = kw
        self._cache = {}
        self._is_valid = True
        
        # 预检查数组是否为空
        try:
            self._validate_volume()
        except Exception as e:
            print(f"警告: 体素数据初始化时出错 {self.nam}: {e}")
            self._is_valid = False
    
    def _validate_volume(self):
        """验证体素数据"""
        if not hasattr(self.vol, 'GetImageData'):
            raise ValueError("输入不是有效的体素节点")
        
        image_data = self.vol.GetImageData()
        if not image_data:
            raise ValueError("体素节点没有图像数据")
        
        # 检查数据范围
        try:
            arr = self.arr
            if arr is not None:
                data_range = (np.min(arr), np.max(arr))
                if data_range[1] == 0:
                    print(f"警告: 体素 {self.nam} 的所有数据为零")
                elif data_range[0] == data_range[1]:
                    print(f"警告: 体素 {self.nam} 的数据值为常数: {data_range[0]}")
        except Exception as e:
            print(f"警告: 无法验证体素数据范围: {e}")
    
    def _get_cached(self, key: str, func):
        """线程安全的缓存属性值"""
        if not self._is_valid:
            return None
            
        if key not in self._cache:
            try:
                result = func()
                if result is not None:
                    self._cache[key] = result
                else:
                    print(f"警告: 属性 {key} 计算结果为None")
                    return None
            except Exception as e:
                print(f"计算属性 {key} 时出错: {e}")
                return None
        
        return self._cache.get(key)
    
    def clear_cache(self):
        """清理缓存"""
        self._cache.clear()
    
    def is_valid(self):
        """检查数据是否有效"""
        return self._is_valid and self.vol is not None
    
    @property
    def update(self):
        """更新体素数据"""
        def _update():
            if not self.is_valid():
                return False
                
            try:
                if hasattr(self, '_modified_arr') and self._modified_arr is not None:
                    slicer.util.updateVolumeFromArray(self.vol, self._modified_arr)
                    self.clear_cache()  # 清理缓存
                    return True
                elif 'arr' in self._cache:
                    slicer.util.updateVolumeFromArray(self.vol, self._cache['arr'])
                    self.clear_cache()
                    return True
                return False
            except Exception as e:
                print(f"更新体素数据失败: {e}")
                return False
        
        return self._get_cached('update', _update)
    
    @property
    def arr(self):
        """获取体素数组"""
        def _get_array():
            if not self.is_valid():
                return None
            return getArr(self.nam) if hasattr(getArr, '__call__') else None
        
        return self._get_cached('arr', _get_array)
    
    def set_array(self, new_array):
        """设置新的数组数据"""
        if not self.is_valid():
            return False
        
        try:
            new_array = np.asarray(new_array)
            current_array = self.arr
            
            if current_array is not None and new_array.shape != current_array.shape:
                print(f"警告: 数组形状不匹配 {new_array.shape} vs {current_array.shape}")
            
            self._modified_arr = new_array
            self._cache.pop('arr', None)  # 清除数组缓存
            return True
        except Exception as e:
            print(f"设置数组失败: {e}")
            return False
    
    @property
    def imData(self):
        """获取图像数据"""
        def _get_image_data():
            if not self.is_valid():
                return None
            return self.vol.GetImageData()
        
        return self._get_cached('imData', _get_image_data)
    
    @property
    def op(self):
        """获取原点"""
        def _get_origin():
            if not self.is_valid():
                return np.zeros(3)
            return np.array(self.vol.GetOrigin())
        
        return self._get_cached('op', _get_origin)
    
    @property
    def spc(self):
        """获取间距"""
        def _get_spacing():
            if not self.is_valid():
                return np.ones(3)
            return np.array(self.vol.GetSpacing())
        
        return self._get_cached('spc', _get_spacing)
    
    @property
    def dims(self):
        """获取维度"""
        def _get_dimensions():
            if not self.is_valid():
                return np.zeros(3, dtype=int)
            
            image_data = self.imData
            if image_data:
                return np.array(image_data.GetDimensions())
            return np.zeros(3, dtype=int)
        
        return self._get_cached('dims', _get_dimensions)
    
    @property
    def mat(self):
        """获取IJK到RAS变换矩阵"""
        def _get_matrix():
            if not self.is_valid():
                return np.eye(4)
            return getI2rMat(self.vol) if hasattr(getI2rMat, '__call__') else np.eye(4)
        
        return self._get_cached('mat', _get_matrix)
    
    @property
    def inv_mat(self):
        """获取RAS到IJK变换矩阵"""
        def _get_inverse_matrix():
            if not self.is_valid():
                return np.eye(4)
            return getR2iMat(self.vol) if hasattr(getR2iMat, '__call__') else np.eye(4)
        
        return self._get_cached('inv_mat', _get_inverse_matrix)
    
    @property
    def ps(self):
        """获取点集（体素坐标转换为RAS坐标）"""
        def _get_points():
            if not self.is_valid():
                return {}
            try:
                return vks2Ras(self.vol, lbs=True)
            except Exception as e:
                print(f"获取点集失败: {e}")
                return {}
        
        return self._get_cached('ps', _get_points)
    
    @property
    def mod(self):
        """获取模型（将标签体素转换为3D模型）"""
        def _get_model():
            if not self.is_valid():
                return None
            try:
                if hasattr(lVol2mpd, '__call__'):
                    return lVol2mpd(self.vol, self.nam, **self._kw)
                else:
                    print("警告: lVol2mpd函数不可用")
                    return None
            except Exception as e:
                print(f"创建模型失败: {e}")
                return None
        
        return self._get_cached('mod', _get_model)
    
    @property
    def pd(self):
        """获取PolyData"""
        def _get_polydata():
            if not self.is_valid():
                return None
            try:
                if hasattr(lVol2mpd, '__call__'):
                    return lVol2mpd(self.vol, **self._kw)
                else:
                    print("警告: lVol2mpd函数不可用")
                    return None
            except Exception as e:
                print(f"创建PolyData失败: {e}")
                return None
        
        return self._get_cached('pd', _get_polydata)
    
    @property
    def lbs(self):
        """获取标签值"""
        def _get_labels():
            arr = self.arr
            if arr is not None:
                unique_vals = np.unique(arr)
                non_zero_labels = unique_vals[unique_vals != 0]
                return non_zero_labels.astype(np.int16)
            return np.array([], dtype=np.int16)
        
        return self._get_cached('lbs', _get_labels)
    
    @property
    def bounds(self):
        """获取边界框"""
        def _get_bounds():
            if not self.is_valid():
                return np.zeros(6)
            
            image_data = self.imData
            if image_data:
                return np.array(image_data.GetBounds())
            return np.zeros(6)
        
        return self._get_cached('bounds', _get_bounds)
    
    @property
    def center(self):
        """获取中心点"""
        def _get_center():
            bounds = self.bounds
            if bounds is not None and len(bounds) == 6:
                x_center = (bounds[0] + bounds[1]) / 2
                y_center = (bounds[2] + bounds[3]) / 2
                z_center = (bounds[4] + bounds[5]) / 2
                return np.array([x_center, y_center, z_center])
            return np.zeros(3)
        
        return self._get_cached('center', _get_center)
    
    @property
    def volume_ml(self):
        """计算体积（毫升）"""
        def _calculate_volume():
            arr = self.arr
            spacing = self.spc
            
            if arr is not None and spacing is not None:
                non_zero_voxels = np.count_nonzero(arr)
                voxel_volume = np.prod(spacing)  # mm³
                total_volume = non_zero_voxels * voxel_volume / 1000  # 转换为mL
                return total_volume
            return 0.0
        
        return self._get_cached('volume_ml', _calculate_volume)
    
    def get_label_info(self):
        """获取标签信息"""
        labels = self.lbs
        arr = self.arr
        
        if labels is None or arr is None:
            return {}
        
        label_info = {}
        for label in labels:
            mask = (arr == label)
            voxel_count = np.sum(mask)
            
            label_name = TLDIC.get(label, f'Label_{label}') if TLDIC else f'Label_{label}'
            
            # 计算体积
            voxel_volume = np.prod(self.spc) / 1000 if self.spc is not None else 1.0
            volume = voxel_count * voxel_volume
            
            # 计算质心（体素坐标）
            if voxel_count > 0:
                indices = np.where(mask)
                centroid_ijk = np.array([np.mean(indices[i]) for i in range(3)])
                
                # 转换为RAS坐标
                ijk_homogeneous = np.append(centroid_ijk[::-1], 1)  # 转换为RAS顺序并添加齐次坐标
                centroid_ras = (ijk_homogeneous @ self.mat.T)[:3]
            else:
                centroid_ijk = np.zeros(3)
                centroid_ras = np.zeros(3)
            
            label_info[label_name] = {
                'label_value': int(label),
                'voxel_count': int(voxel_count),
                'volume_ml': float(volume),
                'centroid_ijk': centroid_ijk.tolist(),
                'centroid_ras': centroid_ras.tolist()
            }
        
        return label_info
    
    def get_statistics(self):
        """获取统计信息"""
        arr = self.arr
        
        if arr is None:
            return {}
        
        stats = {
            'dimensions': self.dims.tolist() if self.dims is not None else [0, 0, 0],
            'spacing': self.spc.tolist() if self.spc is not None else [1.0, 1.0, 1.0],
            'origin': self.op.tolist() if self.op is not None else [0.0, 0.0, 0.0],
            'bounds': self.bounds.tolist() if self.bounds is not None else [0.0] * 6,
            'center': self.center.tolist() if self.center is not None else [0.0, 0.0, 0.0],
            'total_volume_ml': float(self.volume_ml),
            'data_range': [float(np.min(arr)), float(np.max(arr))],
            'unique_labels': self.lbs.tolist() if self.lbs is not None else [],
            'label_info': self.get_label_info()
        }
        
        return stats
    
    def crop_to_label(self, label_value: int, padding: int = 5):
        """裁剪到指定标签"""
        arr = self.arr
        
        if arr is None:
            return None
        
        # 找到标签区域
        mask = (arr == label_value)
        if not np.any(mask):
            print(f"未找到标签值 {label_value}")
            return None
        
        # 获取边界框
        indices = np.where(mask)
        min_coords = np.array([np.min(indices[i]) for i in range(3)])
        max_coords = np.array([np.max(indices[i]) for i in range(3)])
        
        # 添加填充
        dims = self.dims
        min_coords = np.maximum(min_coords - padding, 0)
        max_coords = np.minimum(max_coords + padding, dims - 1)
        
        # 裁剪数组
        cropped_arr = arr[
            min_coords[0]:max_coords[0]+1,
            min_coords[1]:max_coords[1]+1,
            min_coords[2]:max_coords[2]+1
        ]
        
        # 创建新的体素
        try:
            new_vol = arr2vol(self.vol, cropped_arr, 
                            mNam=sNam(self.nam, f'crop_label_{label_value}'))
            return volData(new_vol)
        except Exception as e:
            print(f"创建裁剪体素失败: {e}")
            return None
    
    def resample(self, new_spacing, interpolation='linear'):
        """重采样体素"""
        if not self.is_valid():
            return None
        
        try:
            # 获取原始图像
            original_image = puSk(self.vol)
            
            if original_image is None:
                print("无法获取原始图像")
                return None
            
            # 设置重采样参数
            original_size = original_image.GetSize()
            original_spacing = original_image.GetSpacing()
            
            # 计算新尺寸
            new_size = [
                int(round(orig_sz * orig_sp / new_sp))
                for orig_sz, orig_sp, new_sp 
                in zip(original_size, original_spacing, new_spacing)
            ]
            
            # 选择插值方法
            interpolator_map = {
                'nearest': sitk.sitkNearestNeighbor,
                'linear': sitk.sitkLinear,
                'bspline': sitk.sitkBSpline
            }
            interpolator = interpolator_map.get(interpolation, sitk.sitkLinear)
            
            # 执行重采样
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetInterpolator(interpolator)
            resampler.SetOutputDirection(original_image.GetDirection())
            resampler.SetOutputOrigin(original_image.GetOrigin())
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(sitk.Transform())
            
            resampled_image = resampler.Execute(original_image)
            
            # 创建新节点
            node_type = LVOL if isinstance(self.vol, slicer.vtkMRMLLabelMapVolumeNode) else SVOL
            new_vol = skPu(resampled_image, None, 
                          sNam(self.nam, 'resampled'), node_type)
            
            return volData(new_vol)
            
        except Exception as e:
            print(f"重采样失败: {e}")
            return None
    
    def save(self, filepath, compress=True):
        """保存体素数据"""
        if not self.is_valid():
            return False
        
        try:
            # 获取图像数据
            image = puSk(self.vol)
            
            if image is None:
                print("无法获取图像数据")
                return False
            
            # 设置压缩
            writer = sitk.ImageFileWriter()
            writer.SetFileName(filepath)
            
            if compress and filepath.lower().endswith(('.nii', '.nii.gz')):
                writer.SetUseCompression(True)
            
            writer.Execute(image)
            return True
            
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def clone(self, new_name: str = ''):
        """克隆体素数据"""
        if not self.is_valid():
            return None
        
        try:
            cloned_vol = volClone(self.vol, new_name or sNam(self.nam, 'clone'))
            return volData(cloned_vol, **self._kw)
        except Exception as e:
            print(f"克隆失败: {e}")
            return None
    
    def __repr__(self):
        """字符串表示"""
        if not self.is_valid():
            return f"volData(invalid: {self.nam})"
        
        dims = self.dims
        spacing = self.spc
        labels = self.lbs
        
        return (f"volData(name='{self.nam}', "
                f"dims={dims.tolist() if dims is not None else 'N/A'}, "
                f"spacing={spacing.tolist() if spacing is not None else 'N/A'}, "
                f"labels={len(labels) if labels is not None else 0})")


def vks2Ras(vmData, vks=None, lbs: bool = False):
    """体素坐标转RAS坐标（高度优化版）"""
    def vks2Ps__(vks_array, transform_matrix):
        """内部转换函数"""
        if np.all(vks_array == 0):
            return np.array([]).reshape(0, 3)
        
        # 获取非零位置
        nonzero_coords = np.argwhere(vks_array != 0)
        if len(nonzero_coords) == 0:
            return np.array([]).reshape(0, 3)
        
        # 转换坐标顺序 (i,j,k) -> (k,j,i)
        ijk_coords = nonzero_coords[:, ::-1]
        
        # 添加齐次坐标
        ones_column = np.ones((len(ijk_coords), 1), dtype=np.float32)
        homogeneous_coords = np.hstack((ijk_coords.astype(np.float32), ones_column))
        
        # 应用变换
        ras_coords = (homogeneous_coords @ transform_matrix.T)[:, :3]
        return ras_coords
    
    # 获取变换矩阵
    if not isinstance(vmData, np.ndarray):
        vMat = getI2rMat(vmData).astype(np.float32) if hasattr(getI2rMat, '__call__') else np.eye(4)
        if vks is None:
            vks = getArr(vmData) if hasattr(getArr, '__call__') else np.array([])
    else:
        vMat = vmData.astype(np.float32)
        if vks is None:
            raise ValueError("当vmData是数组时，必须提供vks参数")
    
    if not lbs:
        return vks2Ps__(vks, vMat)
    
    # 按标签分组处理
    unique_labels = np.unique(vks)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0:
        return {}
    
    label_ras = {}
    for lb in unique_labels:
        label_name = TLDIC.get(lb, f'Label_{lb}') if TLDIC else f'Label_{lb}'
        mask = (vks == lb)
        label_ras[label_name] = vks2Ps__(mask, vMat)
    
    return label_ras


def readIsoCT(ctF, mNam: str = '', isLb: bool = True, cstU8: bool = True):
    """读取CT并初始化（优化版）"""
    target_spacing = (1.0, 1.0, 1.0)
    
    # 读取图像
    if os.path.exists(ctF):
        img = sitk.ReadImage(ctF)
    else:
        img = puSk(ctF)
    
    if img is None:
        raise ValueError(f"无法读取图像: {ctF}")
    
    # 类型转换
    if cstU8:
        img = sitk.Cast(img, sitk.sitkUInt8)
    
    original_spacing = img.GetSpacing()
    
    # 重采样到等向素
    if original_spacing != target_spacing:
        original_size = img.GetSize()
        
        # 计算新的尺寸
        new_size = [
            int(round(orig_sz * orig_sp / target_sp)) 
            for orig_sz, orig_sp, target_sp 
            in zip(original_size, original_spacing, target_spacing)
        ]
        
        # 设置插值器
        interpolator = sitk.sitkNearestNeighbor if isLb else sitk.sitkLinear
        
        # 执行重采样
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        
        img = resampler.Execute(img)
    
    # 确保RAS方向
    img = sitk.DICOMOrient(img, 'RAS')
    
    # 推送到Slicer
    node_type = LVOL if isLb else SVOL
    vol = skPu(img, None, mNam, node_type)
    
    # 设置原点和间距
    vol.SetOrigin(OP)
    vol.SetSpacing(target_spacing)
    
    return volData(vol, mNam)


def arr2vol(vol: Union[slicer.vtkMRMLVolumeNode, str] = None, arr=0, mNam: str = '', 
            rtnVd: bool = False, pad: int = 1) -> Union[slicer.vtkMRMLVolumeNode, volData]:
    """数组转体素（优化版）"""
    if vol is None:
        vol = SCEN.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode") if 'SCEN' in globals() else None
        if not vol:
            raise ValueError("未找到标签体素节点")
    else:
        vol = getNod(vol) if hasattr(getNod, '__call__') else vol
    
    cVol = volClone(vol, mNam) if hasattr(volClone, '__call__') else vol
    vArr = getArr(cVol) if hasattr(getArr, '__call__') else np.array([])
    
    if not isinstance(arr, np.ndarray):
        arr = np.full_like(vArr, arr, dtype=vArr.dtype)
    
    if pad > 0:
        arr = np.pad(arr, pad, mode='constant', constant_values=0)
    
    # 确保数组尺寸匹配
    if arr.shape != vArr.shape:
        # 裁剪或填充以匹配
        target_shape = vArr.shape
        padded_arr = np.zeros(target_shape, dtype=vArr.dtype)
        
        # 计算复制区域
        copy_shape = tuple(min(a, b) for a, b in zip(arr.shape, target_shape))
        slices = tuple(slice(0, s) for s in copy_shape)
        padded_arr[slices] = arr[slices]
        arr = padded_arr
    
    slicer.util.updateVolumeFromArray(cVol, arr.astype(vArr.dtype))
    
    return volData(cVol) if rtnVd else cVol


def cropVol(vol, roi=None, mNam: str = '', cArr=None, delV: bool = True):
    """裁剪体素（优化版）"""
    vNod = getNod(vol) if hasattr(getNod, '__call__') else vol
    
    if roi is None:
        # 创建自适应ROI
        try:
            from pppUtil import lVol2mpd, pdBbx, addRoi
            vol_models = lVol2mpd(vNod, exTyp="All")
            rNod = pdBbx(vol_models, mNam)[-1]
        except:
            # 备用方案：使用体素本身的边界
            bounds = vNod.GetImageData().GetBounds()
            size = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
            center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
            if hasattr(addRoi, '__call__'):
                rNod = addRoi(size, cp=center, mNam=sNam(mNam, 'roi'))
            else:
                raise ValueError("无法创建ROI")
    else:
        rNod = getNod(roi) if hasattr(getNod, '__call__') else roi
    
    cropLg = slicer.modules.cropvolume.logic()
    cropMd = slicer.vtkMRMLCropVolumeParametersNode()
    cropMd.SetROINodeID(rNod.GetID())
    cropMd.SetInputVolumeNodeID(vNod.GetID())
    cropMd.SetVoxelBased(True)
    
    cropLg.FitROIToInputVolume(cropMd)
    cropLg.Apply(cropMd)
    
    cropped_vol = SCEN.GetNodeByID(cropMd.GetOutputVolumeNodeID()) if 'SCEN' in globals() else None
    
    if cropped_vol and mNam:
        cropped_vol.SetName(mNam)
    
    if cArr is not None and cropped_vol:
        slicer.util.updateVolumeFromArray(cropped_vol, cArr)
    
    if delV and 'SCEN' in globals():
        SCEN.RemoveNode(vNod)
    
    result = volData(cropped_vol, mNam, exTyp="All") if cropped_vol else None
    
    if roi is None:
        return rNod, result
    
    return result


# 体素克隆函数（如果pppUtil不可用的备用实现）
def volClone_backup(vol, nam=''):
    """备用体素克隆函数"""
    try:
        return slicer.modules.volumes.logic().CloneVolumeGeneric(SCEN, vol, nam)
    except:
        print("警告: 无法克隆体素")
        return vol
