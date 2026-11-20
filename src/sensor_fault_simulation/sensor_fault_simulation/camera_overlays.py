#!/usr/bin/env python3
"""Apply indoor camera-only fault overlays to a BGR OpenCV image.

CLI examples:
  python3 apply_camera_fault.py --input frame.png --fault tape --output tape.png
  python3 apply_camera_fault.py --input frame.png --fault water --output water.png

ROS use: instantiate CameraFaultOverlay(width, height) once. Then call
  out_bgr = fault_overlay.apply(in_bgr, 'tape')
or
  out_bgr = fault_overlay.apply(in_bgr, 'water')
before publishing the RGB image to Depth Anything. LiDAR stays unchanged.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

ASSETS=Path(__file__).resolve().parent / 'camera_fault_assets'
# x, y, rx, ry, rotation in the 1280x720 reference asset.
DROPS=[
  (255,426,33,50,-9),(390,184,14,20,5),(529,471,25,36,17),
  (650,290,40,30,-13),(732,430,24,57,22),(810,140,13,21,0),
  (930,285,60,78,-12),(1035,438,27,60,19),(1170,533,35,52,-6),
  (1120,175,20,29,0),(598,105,18,29,0),(1235,380,26,58,0),
  (806,530,13,20,0),(486,596,9,14,0),(186,280,13,17,0),
  (997,625,9,14,0),(755,340,13,19,0),(865,633,9,14,0),
  (121,615,7,10,0),(1073,104,16,25,0)
]

class CameraFaultOverlay:
    def __init__(self, width=1280, height=720, assets_dir=None):
        self.w,self.h=int(width),int(height)
        a_dir=Path(assets_dir) if assets_dir is not None else ASSETS
        def read_rgba(file_name):
            im=cv2.imread(str(a_dir/file_name),cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim!=3 or im.shape[2]!=4:
                raise FileNotFoundError('Expected RGBA PNG: '+str(a_dir/file_name))
            return cv2.resize(im,(self.w,self.h),interpolation=cv2.INTER_AREA)
        self.tape=read_rgba('opaque_industrial_tape_camera_overlay.png')
        self.water=read_rgba('water_splashed_camera_lens_overlay.png')
        yy,xx=np.mgrid[0:self.h,0:self.w].astype(np.float32)
        xr=xx*1280.0/self.w
        yr=yy*720.0/self.h
        dx=np.zeros((self.h,self.w),np.float32)
        dy=np.zeros((self.h,self.w),np.float32)
        mask=np.zeros((self.h,self.w),np.float32)
        for x,y,rx,ry,angle in DROPS:
            ca=np.cos(np.deg2rad(angle)); sa=np.sin(np.deg2rad(angle))
            ux=((xr-x)*ca+(yr-y)*sa)/rx
            uy=(-(xr-x)*sa+(yr-y)*ca)/ry
            r2=ux*ux+uy*uy
            area=(r2<1.0).astype(np.float32)
            strength=np.maximum(1.0-r2,0.0)**.8*area
            dx+=(x-xr)*.21*strength*self.w/1280.0
            dy+=(y-yr)*.21*strength*self.h/720.0
            mask=np.maximum(mask,np.minimum(1.0,area*.55+strength*.35))
        xp=(xr-965)/225.0
        yp=(yr-308)/255.0
        film=np.exp(-1.15*(xp*xp+yp*yp)).astype(np.float32)
        film*=.95+.17*np.sin(yp*13+xp*8)**2
        xp2=(xr-670)/260.0; yp2=(yr-410)/205.0
        film=np.maximum(film,.82*np.exp(-1.85*(xp2*xp2+yp2*yp2)))
        mask=np.clip(np.maximum(mask,film),0.0,.92)
        self.map_x=(xx+dx).astype(np.float32)
        self.map_y=(yy+dy).astype(np.float32)
        self.mask=cv2.GaussianBlur(mask,(0,0),sigmaX=max(0.7,2.1*self.w/1280.0))[:,:,None]
        # Water is intentionally optically blurred: compute its warped/blurred
        # view at half resolution on HD frames to limit ROS camera throughput
        # overhead, then blend it with the original full-resolution scene.
        self.low_w = min(self.w, 640)
        self.low_h = max(1, int(round(self.h*self.low_w/self.w)))
        if self.low_w < self.w:
            target = (self.low_w, self.low_h)
            self.low_map_x = cv2.resize(self.map_x, target, interpolation=cv2.INTER_LINEAR) * (self.low_w/self.w)
            self.low_map_y = cv2.resize(self.map_y, target, interpolation=cv2.INTER_LINEAR) * (self.low_h/self.h)
        else:
            self.low_map_x = self.map_x
            self.low_map_y = self.map_y

    @staticmethod
    def alpha_blend(frame, overlay):
        a=overlay[:,:,3:4].astype(np.float32)/255.0
        return np.clip(frame.astype(np.float32)*(1-a)+overlay[:,:,:3].astype(np.float32)*a,0,255).astype(np.uint8)

    def apply(self, frame_bgr, fault):
        if frame_bgr.shape[:2]!=(self.h,self.w) or frame_bgr.shape[2]!=3:
            raise ValueError('Expected %sx%s BGR image'%(self.w,self.h))
        if fault in ('none','off'):
            return frame_bgr.copy()
        if fault in ('tape','obstruction','opaque'):
            return self.alpha_blend(frame_bgr,self.tape)
        if fault in ('water','splash'):
            if self.low_w < self.w:
                small = cv2.resize(frame_bgr,(self.low_w,self.low_h),interpolation=cv2.INTER_AREA)
            else:
                small = frame_bgr
            warped=cv2.remap(small,self.low_map_x,self.low_map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            sigma=max(.7,9.3*self.low_w/1280.0)
            blurred=cv2.GaussianBlur(warped,(0,0),sigmaX=sigma)
            if self.low_w < self.w:
                blurred=cv2.resize(blurred,(self.w,self.h),interpolation=cv2.INTER_LINEAR)
            res=np.clip(frame_bgr.astype(np.float32)*(1-self.mask)+blurred.astype(np.float32)*self.mask,0,255).astype(np.uint8)
            return self.alpha_blend(res,self.water)
        raise ValueError('fault must be one of none, tape, water')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--fault',choices=['none','tape','water'],required=True)
    a=p.parse_args()
    im=cv2.imread(a.input,cv2.IMREAD_COLOR)
    if im is None:
        p.error('Cannot read --input '+a.input)
    h,w=im.shape[:2]
    fn=CameraFaultOverlay(w,h)
    out=fn.apply(im,a.fault)
    if not cv2.imwrite(a.output,out):
        p.error('Cannot write --output '+a.output)
    print('Wrote',a.output)
