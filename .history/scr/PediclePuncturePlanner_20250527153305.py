# PediclePuncturePlanner.py
# 椎弓根穿刺自动规划 PPP - Pedicle Puncture Planner

from pppUtil import *

RL = rad, L = 'RL'
AP = A, P = 'AP'
SI = S, I = 'SI'
TLDIC = {14: 'T7',
         15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12',
         20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'S1',
         26: 'Sc', 27: 'vBs'}  # 标签🔠
# TLDIC.values()
TLs_ = ['T7',
        'T8', 'T9', 'T10', 'T11', 'T12',
        'L1', 'L2', 'L3', 'L4', 'L5', 'S1',
        'Sc', 'vBs']  # 签🗒️(含L6)
LVS = TLs_[1:-3]
_rs = ['vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4',
       'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
       'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10',
       'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7']
_sc = ['spinal_cord']  # 脊椎分割任务(37-26, 79)
# LBS_ = np.arange(14, 28)  # 标🗒️
LBS = np.arange(15, 25)
# TLDIC_ = ls2dic_(LBS_, TLs_)
fDic = dfDic(dict)  # 建🗞📃🔠
rPth = r"/Users/liguimei/Documents/PTP/paper0/resources/"   # 🗞📃路径
vPth = rPth+"ttSegNiiGz/"                                   # 🗞📃路径
join_ = lambda *args: os.path.join(*args)
for dir in os.listdir(vPth):                                # 🔄📃夹
    pths = join_(vPth, dir)   # 🗞📃路径
    if os.path.isdir(pths):  # 🚦是📃夹
        for file in os.listdir(pths):  # 🔄📃
            def fNam(nam): return join_(vPth, dir, nam+file[2:])
            if file.endswith('.nii.gz'):  # 🚦是.nii.gz
                # nams = ['ct', 'vt', 'vb']
                nams = ['ct', 'vs', 'vb', 'vt']
                for nam in nams:
                    fDic[dir][nam] = fNam(nam)
            if file.endswith('.json'):
                fDic[dir]['cp'] = fNam('cp')
            fDic[dir]['npz'] = join_(vPth, dir, 'npz_0117x.npz')
# dict_keys(['v586/', 'v519/', 'v542', 'v533', 'v593/',
#            'v514', 'v577', 'v539', 'v768', 'v536'])
# dict_keys(['v091', 'v096', 'v097', 'v127', 'v145', 'v533', 'v135', 'v104', 'v151', 'v082', 'v112', 'v536', 'v074'])
MDPTH = rPth+'screwMods/'  # 🔩路径
# //MARK: 椎弓根穿刺规划
# tag PediclePuncturePlanner: 椎弓根穿刺规划


class PediclePuncturePlanner(object):                   # 🎬 椎弓根穿刺规划
    def __init__(self,                                    # 🧮 初始化
                 ctFile='',                                # 🔱 CT📁
                 vtFile='',                                 # gt脊椎📁
                 vsFile='',                                 # gt脊柱📁
                 vbFile='',                                 # gt椎体📁
                 #  cpFile='',                                 # gt椎弓根📁
                 pId='',                                    # 患者ID
                 lbs=None,                                  # 标签🔢
                 npzFile='',                                # npz📁
                 ):
        # 场景初始化
        self.tim0 = time.time()
        SCEN.Clear(0)
        vNods = ut.getNodes("vtkMRMLSliceDisplayNode")
        for vNod in vNods:
            vNod.SetIntersectingSlicesVisibility(1)
            vNod.SetLineWidth(3)
        # 参数初始化
        self.ctFile = ctFile
        self.vtFile = vtFile
        self.vsFile = vsFile
        self.vbFile = vbFile
        # self.cpFile = cpFile
        self.npzFile = npzFile
        self.pId = pId
        if lbs is None:
            self.lbs = LBS.tolist()
            self.lvs = LVS
        else:
            self.lbs = lbs
            self.lvs = [TLDIC[lb] for lb in self.lbs]
        self.scrsDic = dfDic(list)
    def ls2dic(self, ls): return ls2dic_(
        self.lvs, ls)  # - 🗒️转🔠{lbs:ls}

    def lg_(self, txt, t): return log_(txt, (self.tim0, t), sec=.1)

    # def ttSegCT(self, ctVol, isCpu=True):  # - 🧮: 标记CT
    #     import TotalSegmentator as ttSeg

    #     def seg2lbVol__(seg, mNam=''):
    #         lbVol = SNOD("vtkMRMLLabelMapVolumeNode", mNam)
    #         segLg = slicer.modules.segmentations.logic()
    #         segLg.ExportAllSegmentsToLabelmapNode(seg, lbVol)
    #         SCEN.RemoveNode(seg)
    #         return lbVol
    #     lgc = ttSeg.TotalSegmentatorLogic()
    #     vtSeg = SNOD("vtkMRMLSegmentationNode", 'vts')
    #     lgc.process(ctVol, vtSeg, cpu=isCpu, fast=isCpu,
    #                 task='total', subset=_rs)  # 分割[TODO]:CPU设置
    #     vts_ = seg2lbVol__(vtSeg, 'vts_')
    #     SCEN.RemoveNode(vtSeg)
    #     vbSeg = SNOD("vtkMRMLSegmentationNode", 'vbs')
    #     lgc.process(ctVol, vbSeg, cpu=isCpu, fast=isCpu,
    #                 task='vertebrae_body')
    #     vbs_ = seg2lbVol__(vbSeg, 'vbs_')
    #     SCEN.RemoveNode(vbSeg)
    #     return vts_, vbs_

    # tag 1_loadVol: 导入分割🗞并💃化

    def modVols(self):                           # 🧮: 导入分割🗞并💃化
        lbs = [self.lbs[0]-1] + self.lbs
        lvs = [TLDIC[lb] for lb in lbs]
        ctVd = readIsoCT(self.ctFile, 'ct_', 0, 1)
        vsVd = readIsoCT(self.vsFile, 'vs0_')
        vbVd = readIsoCT(self.vbFile, 'vb0_')

        vsVol = vsVd.vol
        vssArr = vsVd.arr
        scArr = np.where(vssArr != 79, 0, 28)
        sc_ = arr2vol(vsVol, scArr, 'sc_')
        if self.vtFile != '':
            vtVd = readIsoCT(self.vtFile, 'vt0_')
            # vt_ = vtVd.vol
            vsArr = vtVd.arr
            # vtsArr = np.where(vsArr != 0, 1, 0)
            vsArr[vsArr < lbs[0]] = 0
            vsArr[vsArr > lbs[-1]] = 0
            vt_ = arr2vol(vtVd.vol, vsArr, 'vt_')
            vtsArr = np.where(vsArr != 0, 1, 0)
            vs_ = arr2vol(vt_, vtsArr, 'vs_')
            # self.vtsArr = vtsArr
            # self.vt_ = vt_

        else:
            vsArr = 51 - vssArr
            vsArr[vsArr < lbs[0]] = 0
            vsArr[vsArr > lbs[-1]] = 0
            vt_ = arr2vol(vsVol, vsArr, 'vt_')
            vtsArr = np.where(vsArr != 0, 1, 0)
            vs_ = arr2vol(vsVol, vtsArr, 'vs_')
        vbArr = vbVd.arr * vsArr
        vb_ = arr2vol(vbVd.vol, vbArr, 'vb_')
        ctArr_ = ctVd.arr
        ctArr = np.where(ctArr_ == 0, 1, ctArr_)
        ctArr = ctArr * vtsArr
        ct_ = arr2vol(ctVd.vol, ctArr, 'ct_')
        self.vsPd = lVol2mpd(vs_, 'vsMod', exTyp='All')  # -脊柱💃
        self.vsPd.CreateDefaultDisplayNodes()
        self.vsPd.GetDisplayNode().SetOpacity(0.3)  # 设置透明度

        self.roi = pdBbx(self.vsPd, 'roi')[-1]
        self.vsVd = cropVol(vs_, self.roi, mNam='vs')
        self.scVd = cropVol(sc_, self.roi, mNam='sc')
        self.vtVd = cropVol(vt_, self.roi, mNam='vt')
#         self.vt0Vd = cropVol(vtVd.vol, self.roi, mNam='vt0')
        self.ctVd = cropVol(ct_, self.roi, mNam='ct')
        # self.vbCps = ndA(c2s_(self.vbCpc))+self.op
        self.vbVd = cropVol(vb_, self.roi, mNam='vb')
        self.vbsVd = cropVol(vbVd.vol, self.roi, mNam='vbs')
        self.vbPdc = self.vbVd.pd
        self.vbsPd = lVol2mpd(self.vbsVd.vol, exTyp='All')  # -椎体💃
        self.vbCps = ndA([pdCp(pds) for pds in c2s_(self.vbPdc)])
        ps2cFids(self.vbCps, 'vbCps', lvs)
        self.vbCpc = self.ls2dic(self.vbCps[1:])
        self.vtPdc = self.vtVd.pd
        # self.vtPsc = pd2Vps(self.vtVd)  # -脊椎💃
        # mvId = np.argmax(dsts)  # 最大质心距(一般为L3)
        # dstI = dsts[mvId]/dsts[mvId-1]  # 最大质心距与次大质心距比
        # if dstI > 1.66:  # 🚦L3距大于L2距2倍(1.66)
        #     print(f'发现连接椎: {mvId=}')
        #     return
        # self.vsPd = self.vsVd.mod  # -脊柱💃
        # self.vsPd.SetName('opt_vsMod')  # ～～莫名
        # self.vsPd.CreateDefaultDisplayNodes()
        # self.vsPd.GetDisplayNode().SetOpacity(0.3)  # 设置透明度
        # self.vsOT = getObb(self.vsPd)  # 👤脊柱OBB
        # self.ctMat = r2iMat(self.ctVd.vol)
        sVbPs = (self.vbCps[:-1]
                 + self.vbCps[1:]) / 2  # 椎体间点集(n+1)
        sdrs, sdts = psDst(self.vbCps[:-1]
                           - self.vbCps[1:])  # 质心距s
        # sdrs = uNor(self.vbCps[1:]
        #             - self.vbCps[:-1])  # 椎体下方向及中点距离集(n+1)
        # scPds = vtkPlns(list(zip(*ndA(sVbPs, sdrs))), self.scVd.pd, '',1)
        # vtkPlns(list(zip(*ndA(self.vbCps[1:], sdrs))), mNam = 'vbCpln')
        self.sVbPc = self.ls2dic(sVbPs)  # - 椎体间点s
        self.sdrtc = self.ls2dic(sdrs)  # - 椎体上方向s
        self.sDstc = self.ls2dic(sdts)  # - 椎体间距s
        # self.scPdc = self.ls2dic(scPds)
        self.vsOT = getObt(self.vsPd)  # 👤脊柱OBB

        return

    # tag 2_oritVt: 单椎定向
    # @timOut(100)
    def oritVt(self, mNam=''):
        # 1_ 椎体
        miPn = (self.vbCp, -self.sdrt)
        self.vbCt = CtPj(self.vbPd,
                         self.sdrt, self.vbCp, clean=True,
                         mNam=sNam(mNam, 'ePs'),
                         )
        eSps, eIps, sDrt, vbIps, vbInGps = self.vbCt.edPs()
        self.sDrt = sDrt if sDrt[2] > 0 else -sDrt
        self.eSp, esDt, _ = ps_pn(eSps, (self.vbCp, self.sDrt))
        self.eIp, eiDt, _ = ps_pn(eIps, (self.vbCp, self.sDrt))
        eDt = esDt+eiDt
        # 将vtPd以(self.eSp, self.sDrt)切割, 并将上面的部分投影到该平面
        vtPd_, vtPd0 = vtkPlnCrop(self.vtPd,(self.eSp, self.sDrt),refP=self.vbCp,inPd=True,)
        self.vtPd_ = pdAndPds(vtPd_, vtPd0)
        # vtPd_ = dotCut(self.vtPd, (esp, -sDrt), dst=eDt)
        sCp_ = pdCp(vtkCut(self.scVd.pd, miPn))    

        # vtPd_ = pdAndPds(self.vtPd_, vbps,
        #                  mNam=sNam(mNam, 'vtPd0'))
        self.vbSc = CtPj(self.vtPd,
                         sDrt, self.vbCp,
                          sp = sCp_,
                         mNam=sNam(mNam, 'ePs'))
        try:
            scIps, sCt, sRps = self.vbSc.sCt()
        except Exception as e:
            raise ValueError(f"{self.lv}椎管轮廓计算失败: {str(e)}")
        vbCp = vbIps.mean(0)
        sCp = scIps.mean(0)
        self.aDrt = uNor(vbCp - sCp)
        rDrt = np.cross(self.sDrt, self.aDrt)
        self.rDrt = rDrt if rDrt[0] > 0 else -rDrt
        self.ras = ndA(self.rDrt, self.aDrt, self.sDrt)
        sRps_ = dotCut(sRps, (self.eSp, -self.sDrt), dst=eDt)
        # 3_ 切三角
        rlps, self.rlPs = mxNorPs_(sCt,               # 椎管极值点和投影还原点集
                                   self.rDrt,
                                   rjPs=sRps_)
        cpTri = ndA(self.rlPs).mean(0)                 # 椎管中心点
        self.cpTri = (cpTri
                      if abs((cpTri-self.eSp)@self.sDrt) > 2 else
                      cpTri-self.sDrt*2)
        sRad = norm(rlps[0]-rlps[1])/2
        rp, _, _, rdx = p2pLn(self.cpTri,
                              nor=self.rDrt,
                              plus=sRad)
        adx = p2pLn(self.cpTri, nor=self.aDrt)[-1]
        aCtPs = vbInGps[vbInGps[:, 1] > self.vbCp[1]]
        ap_ = dotPlnX(aCtPs, (self.cpTri, self.rDrt))
        self.hiTri = norm(ap_-self.cpTri)
        ap = adx(self.hiTri)
        p2pLn(self.cpTri, ap, mNam=sNam(mNam, 'apLn'))
        self.cTri = ndA(rp, rdx(-sRad), ap)
        pds2Mod(self.cTri, sNam(mNam, 'cuTri'), 0)
        # 切参数
        # vtkPln(self.ePln, mNam=sNam(mNam, 'ePln'))
        idx = sCt[:, 1] > sCp[1]
        self.sAps = [dotPlnX(sCt[idx], (sCp, self.rDrt)),
                     dotPlnX(sCt[~idx], (sCp, self.rDrt))]
        # p2pLn(self.sAps[0], self.sAps[1], mNam=sNam(mNam, 'sAp0'))
        sRad_ = sRad - 2
        self.sRps = ndA(rdx(sRad_), rdx(-sRad_))
        # p2pLn(self.sRps[0], self.sRps[1], mNam=sNam(mNam, 'sRp0'))
#         self.eSps = self.sRps + self.sDrt*abs((self.sRps[0]-eSp)@self.sDrt)
        # p2pLn(self.eSps[0], self.eSps[1], mNam=sNam(mNam, 'epLn'))
        return

    # tag 3_cutPd: 单椎椎弓根截取
    def cutPd(self, mNam=''):
        # 0. 初始化
        rp, lp, ap = self.cTri  # 提取左,右,前点。
        rlp = np.array([rp, lp])
        rlDr = rDr, lDr = uNor(ap - rlp)
        rlAdr = uNor(np.cross(rlDr, self.sDrt))
        apDst = np.linalg.norm(self.sAps[1] - self.sAps[0])
        rDrt = [self.rDrt, -self.rDrt]

        # self.rDrt, self.aDrt, self.sDrt = self.ras                                  # 提取方向向量
        # 计算左右截切方向, 三角斜边长度
        lrDt, lrPx = p2pLn(rp, lp)[2:]  # 三角底边长度, 底边方向λ
        # vtEpd = epCut(self.vtPd, (self.ep, sd), mNam=sNam(mNam, 'vtEpd'))
        self.cPds = []
        self.rtNors = []
        self.bNors = []
        self.ctCps = []
        self.vSums = []
        self.rtCps = []
        self.scrs = []
        self.scrDic = dfDic(list)
        # 遍历左右两侧进行切割
        # self.rtPs = [rayCast(lp, rp, self.vsOT, plus=10),
        # rayCast(rp, lp, self.vsOT, plus=10)]
        for i, s in enumerate('RL'):  # 左右🔄
            try:                                     
                self.cPd = vtkPlnCrop(self.vtPd,
                                    ((self.sRps[i], rDrt[i]),
                                    (self.sAps[0], rlDr[i]),
                                    (self.sAps[1], rlDr[i])),
                                   refP=rlp[i],
                                   mNam=sNam(mNam, s+'bPd'))          # ✂️椎弓根块
                
                self.rtP = self.cTri[i]
                ctCp = self.getScrew(self.ras,
                                     mNam='scr' + self.lv + s + '0',
                                     rl=s)  # 🔦椎弓根块MIC

                self.ctCps += [ctCp,]
                self.cPds += [self.cPd]  # self椎弓根块
                self.vSums += [self.vxF,]
                self.scrs += [self.scr,]
            except Exception as e:
                print(f"{self.lv}{s} 椎弓根切割失败: {str(e)}")
                # 记录失败但继续处理另一侧
                continue
        self.multiDrts(mNam)
        return

    # tag 4_multiDrts: 多方向计算

    def multiDrts(self, mNam=''):
        """计算多个方向
        🧮 计算多个方向
        🎁 rtTri: 三角形点集
            rtNors: 旋转方向
        """
        # ctCp = ndA(self.ctCps).mean(0)
        rtPs = rayCast(self.ctCps[0],
                       self.ctCps[1],
                       self.vsOT,
                       plus=0,
                       oneP=False)
        # assert len(rtPs) == 2, '多方向计算失败'
        self.rtCp = ndA(rtPs).mean(0)
        cpx = p2pLn(self.cpTri, self.rtCp)[-1]
        self.rTri = cpx(p=self.cTri)
        self.ap = self.rTri[2]
        if mNam != '':
            pds2Mod(self.rTri, mNam+'rtTri', 0)
        tAgl = p3Angle(*self.rTri[::-1])
        self.tAgl = tAgl()
        # uAgl = self.tAgl/8
        # self.ap_ = self.ap+self.aDrt*self.hiTri
        # self.rNors_ = lambda n, a_rp=self.ap_-self.rtP: uNor(rdrgsRot(
        #             a_rp, self.sDrt*np.sign(a_rp[0]),
        #             np.radians(n * uAgl)))
        self.scrsDic[self.lv] = lsDic(dict(
            ras=self.ras,
            cpdR=pd2Dic(self.cPds[0]),
            cpdL=pd2Dic(self.cPds[1]),
            rtTri=self.rTri),
            self.scrsDic[self.lv])
        return

    # tag 5_getScrew: 椎弓根螺钉生成
    def getScrew(self, nor, mNam='', spLn=None, rl=''):  # 🧮: 生成螺钉 getScrew 螺钉生成
        ctData = CtPj(self.cPd, nor,
                      self.rtP,
                      mNam=sNam(mNam, 'ctPj')
                      )                       # 🔦椎弓根块MIC
        cp, rad_, cArr, ctPs = ctData.mic()
        if nor.ndim == 2:
            nor = nor[1]
        if rad_ < 1.5 and spLn is None:
            self.vxF = 0
            self.scr = None
            return cp
        self.rad = max(1., (rad_ - .5)//0.25*0.25)
        self.rad = self.rad if self.rad < 6.0 else 6.0
        cArr = psRoll_(cArr, self.rtP)
        cArr0 = cp + self.rad * uNor(cArr-cp)
        try:
            p0, p1 = rayCast(cp-nor*60, cp+nor*60,
                             mPd=self.vsOT,
                             oneP=False)[[0, -1]]
        except:
            print(f'{mNam}射线未穿透')
            return None
        # addFid(p0, 'p0', rad_*2)
        if spLn is not None and not dotPlnX(p1, spLn, eqX=None, isIn=True):
            return None
        px = p2pLn(p0, p1,
                   # mNam=sNam(mNam, 'p2p'),
                   #    gDia=self.rad-2
                   )[-1]
        ps0 = cArr0-nor*40
        p1s = cArr0+nor*60
        secPs = []
        for p0_, p1_ in zip(ps0, p1s):
            secPs += [rayCast(p1_, p0_, mPd=self.vsOT, oneP=False)[[0, -1]],]
        sP1, sP0 = ndA(secPs)[:, 0], ndA(secPs)[:, 1]
        pj0 = np.dot(sP0, nor)
        pj1 = np.dot(sP1, -nor)
        sp0 = sP0[np.argmax(pj0)]
        sp1 = sP1[np.argmax(pj1)]
        self.p0, dtx = lnXpln((sp0, nor), p0, p1)
        lg_ = dtx(sp1)-dtx()
        self.lg = lg_ * 5//5
        if self.lg < 30 and spLn is None:
            self.vxF = 0
            self.scr = None
            return cp
        if self.lg > 70:
            self.lg = 70
        self.p1 = px(self.lg, self.p0)
        self.scr = lambda nam='': (
            p2pLn(p0, p1, dia=self.rad *
                              2, mNam=nam),
            addFid(p0, sNam(nam, 'p0'),
                   self.rad*2))
        sGps_ = thrGrid(self.p0, self.p1, rad=self.rad)
        # sGps = nx3ps_(self.px(rgN_(self.lg), cArr0_[:, None]))
        sGps = nx3ps_(sGps_)
        vksCT = ras2vks(sGps, self.ctVd.vol, lb=0)[0].flatten()
        vk0Ids = np.where(vksCT == 0)[0]
        gps0 = sGps[vk0Ids]
        # ps2cFids(gps0, sNam(mNam, 'gps0'))
        # self.vxF = vksCT.mean()*self.lg*self.rad*self.rad
        self.vxF = 2 * np.pi * self.rad * np.sum(vksCT)
        # self.FscNor =
        self.scrDic[rl] = lsDic(dict(
            nor=nor, cp=cp, p0=p0, p1=p1,
            sP0=self.p0, sP1=self.p1, sRad=self.rad,
            rad0=rad_, lg0=lg_, sLg=self.lg, cArr=cArr, ctPs=ctPs,
            vks=vksCT, sGps=sGps_,
            vxSum=self.vxF, gps_0=gps0),
            self.scrDic[rl])
        return cp

    # tag 6_optScrew: 优化螺钉路径
    def optScrew(self, s, inScr=False):
        """优化螺钉路径
        🧮 函数: 计算最优螺钉路径
        🔱 参数:
            s: 左/右侧 ('R'/'L')
            inScr: 是否插入螺钉模型
        🎁 返回:
            最优螺钉名称
        """
        # 初始化
        ii = RL.index(s)
        ii_ = 1-2*ii
        self.cPd = self.cPds[ii]
        ctCp = self.ctCps[ii]
        ctCp_ = ctCp  # 保存初始接触点
        self.rtP = self.rTri[ii]

        # 常量定义
        uAgl = self.tAgl/8
        mxAgls = 8

        # 计算旋转向量
        # a_rv = self.rTri[2] + self.aDrt * self.hiTri - self.rtP

        # 定义旋转法线函数
        def rNor_(n):
            return uNor(rdrgsRot(self.aDrt, self.sDrt*ii_, np.radians(n * uAgl)))

        # 初始化存储结构
        vSum = [self.vxF,]  # 使用列表存储体素和，预留额外位置给前向路径
        scrNam = {0: f'scr{self.lv}{s}0'}  # 使用列表存储螺钉名称
        sNor = uNor(self.rtP-self.rtCp)  # 基准方向

        # 计算不同角度的螺钉
        for jj in range(1, mxAgls):
            log_("")
            scrNam[jj] = f'scr{self.lv}{s}{jj}'
            try:
                ctCp = self.getScrew(
                    rNor_(jj),
                    spLn=(self.rtCp, sNor),
                    mNam=scrNam[jj],
                    rl=s
                )

                if ctCp is None:
                    if jj > 4:
                        ctCp = ctCp_
                        continue  # 跳过无效结果
                    raise ValueError("射线未穿透")

                ctCp_ = ctCp
                vSum.append(self.vxF)  # 只有成功时才添加

            except Exception as e:
                print(f'{self.lv}{s}{jj}计算失败: {e}')
                continue

        # 添加前向路径
        jj += 1
        scrNam[jj] = f'scr{self.lv}{s}{jj}'
        try:
            ctCp = self.getScrew(
                uNor(self.rTri[2] - ctCp),
                mNam=scrNam[jj],
                spLn=(self.rtCp, sNor),
                rl=s
            )
            if ctCp is not None:
                vSum.append(self.vxF)  # 只有成功时才添加
        except Exception as e:
            print(f'{self.lv}{s}{jj}计算失败: {e}')

        # 找出最优螺钉
        # 安全地获取最优索引
        vId = np.argmax(ndA(vSum))
        print(f'{vId=}')

        # 确保索引有效
        if 0 <= vId < len(scrNam) and scrNam[vId]:
            optNamv = scrNam[vId]
        else:
            # 如果索引无效，找到第一个有效的螺钉名称
            for idx, name in enumerate(scrNam):
                if name and vSum[idx] > 0:
                    vId = idx
                    optNamv = name
                    break
            else:
                print(f"警告: 无法为{self.lv}{s}创建最优螺钉，没有有效的螺钉名称")
                return None

        opNamv = f'optv_{optNamv}'

        # 安全地创建螺钉模型
        # if (s in self.scrDic and
        #     'p0' in self.scrDic[s] and
        #     'p1' in self.scrDic[s] and
        #     'sRad' in self.scrDic[s] and
        #     0 <= vId < len(self.scrDic[s]['p0'])):

        self.scr = p2pLn(
            self.scrDic[s]['p0'][vId],
            self.scrDic[s]['p1'][vId],
            dia=self.scrDic[s]['sRad'][vId]*2,
            mNam=opNamv
        )

        addFid(
            self.scrDic[s]['p0'][vId],
            sNam(opNamv, 'p0'),
            self.scrDic[s]['sRad'][vId]*2
        )

        if inScr:
            self.insertScr(optNamv, mNam=opNamv)

        self.scrDic[s]['optv'] = vId
        # else:
        #     print(f"警告: 无法为{self.lv}{s}创建最优螺钉，数据不完整")

        return optNamv

    def loadScrPds(self):
        pth = rPth+'screwMods/'  # 🔩路径
        self.moDic = {}
        for f in os.listdir(pth):
            if f.endswith('.vtk'):
                self.moDic[str(f[1:-9])] = readVtk(pth+f)
        self.head = self.moDic['Head']
        return

    # tag 7_insertScr: 插入螺钉

    def insertScr(self, optNam, mNam='screw'):
        p0 = self.scrsDic[optNam]['p0'][0]
        p1 = self.scrsDic[optNam]['p1'][0]
        # px = p2pLn(p0, p1)[-1]
        dia = self.scrsDic[optNam]['sDia'][0]
        lg = self.scrsDic[optNam]['lg'][0]
        sMd = self.moDic[str(dia)]
        sMd = getNod(sMd, optNam + str(dia))
        sMd = pdTf(sMd, rotZ=180, delMpd=True)
        cMd = vtkPln(
            (ndA([0, lg, 0]), -NY),
            sMd, cPlns=True,
        )
        hMd = pdTf(self.head, rotZ=180, goY=-lg, delMpd=True)
        scrMd = pdAndPds((cMd, hMd))
        pdTf(scrMd, p1, p0, delMpd=True,
             mNam=mNam+'_D:'+str(dia)+'_L:'+str(lg))
        # pdTf(cMd, p1, p0, delMpd=True,
        #      mNam=optNam+'bd_D:'+str(dia)+'_L:'+str(lg))

    # tag 00_pppLogic: 大🔄

    def pppLogic(self, mNam='', inScr=False):

        id = self.pId
        tim = self.lg_(id + '计时开始', self.tim0)
        if self.vsFile == '':
            self.vsFile, self.vbFile = self.ttSegCT()
        self.modVols()
        tim = self.lg_('🗞-->💃完成', tim)
        if inScr:
            self.loadScrPds()
            tim = self.lg_('加载螺钉pd完成', tim)
        self.ras = OP
        self.rasc = {}
        # scrNam = dfDic(list)
        for il, lv in enumerate(self.lvs):
            self.scrDic = dfDic(list)
            # 节段名(20: L1...)
            self.lv = lv
            tim = log_(self.lv+': 开始计算', (self.tim0, tim))
            # 👤 椎体pd
            self.vbPd = self.vbPdc[lv]
            self.vtPd = self.vtPdc[lv]                              # 👤 脊椎pd
            self.vtObt = getObt(self.vtPd)  # 👤 脊椎OBB
            # self.scPd = self.scPdc[lv]
            self.sdrt = self.sdrtc[lv] if il == 0 else self.sDrt
            self.adrt = ndA(0,1,0) if il==0 else self.aDrt
            self.sdst = self.sDstc[lv]
            # 👤 椎体中心
            self.vbCp = self.vbCpc[lv]
            # 👤 椎间点
            self.vbSp = self.sVbPc[lv]
            # self.vbPs = self.vbVd.ps[lv]

            try:
                self.oritVt(sNam(mNam, 'oritVt'))
                tim = self.lg_('单椎定向完成', tim)
                self.cutPd(sNam(mNam, 'cutPd'))
                tim = self.lg_('椎弓根提取完成', tim)
            except Exception as e:
                print(f'由于{e}，{self.lv}计算失败')
                self.scrsDic[self.lv] = {}
                continue
            for s in RL:  # 🔄 左右
                self.rtP = self.rTri[:2][RL.index(s)]
                log_(id+self.lv+s+': 开始筛选')
                self.optScrew(s, inScr)
                tim = self.lg_('  '+self.lv+s+': 完成', tim)
                self.scrsDic[self.lv] = lsDic({s: self.scrDic[s]},
                                              self.scrsDic[self.lv])
            # if i == 2:
            #     break
#         nodsDisp()
#         nodsDisp('opt*', 1)
        # np.savez(self.npzFile, **self.scrsDic)
        tim = self.lg_('全部完成: ', tim)
        return


# You need to load the extension first
# %load_ext viztracer
# import importlib
# import pppUtil
# importlib.reload(pppUtil)  # 确保这行在修改后执行
# from pppUtil import *       # 重新导入所有内容
# with VizTracer(output_file="opt074.json", min_duration=800) as tracer:
# # %load_ext viztracer
# # %%viztracer
# # Your code after
# # 大循环修改为
# sDic = {}
# pKeys = list(fDic.keys())
# for i, k in enumerate(pKeys):
#     try:
#         fdic = fDic[k]
#         # 添加资源清理
#         slicer.mrmlScene.Clear(0)

#         self = PediclePuncturePlanner(
#             fdic['ct'], '', fdic['vs'], fdic['vb'],
#             pId=k,
#             # lbs=list(np.arange(20, 26))  # 明确指定处理范围
#         )
#         print(f'开始{k}计算')
#         self.pppLogic()

#         # 添加中间结果保存
#         # np.savez(fdic['npz'], **self.scrsDic)
#         sDic[k] = self.scrsDic

#         # 强制垃圾回收
#         del self
#         import gc
#         gc.collect()

#     except Exception as e:
#         print(f'{k}计算失败: {str(e)}')
#         continue
# #     if i > 2: break
# # # 最终保存
# np.savez(r'allCases0401x.npz', **sDic)
# 小循环
pId = 'v096'
lbs = list(np.arange(15, 17))  # T8-L5
fdic = fDic[pId]
self = PediclePuncturePlanner(
    fdic['ct'], '',
    fdic['vs'], fdic['vb'],

    pId=pId,
    # lbs=lbs
)
self.modVols()
# self.pppLogic()

# # 大循环
# # lbs = list(np.arange(23, 25))  # T8-L5
# sDic = {}
# pKeys = list(fDic.keys())
# pks = pKeys[:]
# for k in pks[:1]:
#     fdic = fDic[k]
# #     if os.path.exists(fdic['npz']):
# #         continue
#     self = PediclePuncturePlanner(
#         fdic['ct'], '', fdic['vs'], fdic['vb'], k) #, lbs)
#     print(f'开始{k}计算')
#     try:
#         self.pppLogic('tt')
#     except:
#         print(f'{k}计算失败')  # 2024年11月27日21:48:14 097, 112,145,127
#         continue
# #     np.savez(fdic['npz'], **self.scrsDic)
#     sDic[k] = self.scrsDic
# np.savez('all0420', **sDic)