"""USB 相机内参标定：Tkinter 界面、OpenCV 计算，不连接机器人。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
PATTERN = (6, 8)
SIZE = (1280, 720)
CAMERA = 'GENERAL WEBCAM'
SQUARE_MM = 25.0
cv2.setNumThreads(2)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def write_image(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, frame)
    if not ok:
        raise RuntimeError('图像编码失败')
    data.tofile(str(path))


def read_image(path):
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'无法读取 {path}')
    return frame


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
    return corners.astype(np.float32) if ok else None


def board_points(square_mm):
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    return obj * (square_mm / 1000.)


def solve(corners, size, square_mm):
    obj = board_points(square_mm)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj.copy() for _ in corners], corners, tuple(size), None, None)
    if not np.isfinite(K).all() or not np.isfinite(dist).all() or not np.isfinite(rms):
        raise ValueError('标定结果无效，请增加不同位置和倾角的清晰照片')
    errors = []
    for c, r, t in zip(corners, rvecs, tvecs):
        p, _ = cv2.projectPoints(obj, r, t, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((c.reshape(-1, 2)-p.reshape(-1, 2))**2, axis=1)))))
    centers = np.array([c.reshape(-1, 2).mean(0) / size for c in corners])
    warnings = []
    if rms > 1.0:
        warnings.append('重投影RMS超过1像素，请检查模糊、纸张弯曲和逐张误差。')
    if np.any(np.ptp(centers, axis=0) < .2):
        warnings.append('棋盘格中心位置变化较小，建议补拍画面不同边缘位置。')
    return dict(camera_matrix=K.tolist(), dist_coeffs=dist.reshape(-1).tolist(),
                rms_px=float(rms), per_view_rms_px=errors,
                board_rvecs_in_camera=[r.reshape(-1).tolist() for r in rvecs],
                board_tvecs_in_camera_m=[t.reshape(-1).tolist() for t in tvecs],
                center_span_fraction=np.ptp(centers, axis=0).tolist(), warnings=warnings)


def calibrate_session(folder):
    folder = Path(folder)
    meta = json.loads((folder/'session.json').read_text(encoding='utf-8'))
    if tuple(meta['pattern_inner_corners']) != PATTERN:
        raise ValueError('采集目录的棋盘格规格与程序不一致')
    size = tuple(meta['image_size'])
    corners, names, rejected = [], [], []
    for path in sorted(folder.glob('frame_*.png')):
        im = read_image(path)
        if (im.shape[1], im.shape[0]) != size:
            raise ValueError(f'图像分辨率不一致：{path.name}')
        c = detect(im)
        if c is None:
            rejected.append(path.name)
        else:
            corners.append(c)
            names.append(path.name)
    if len(corners) < 15:
        raise ValueError(f'至少需要15张完整检测到棋盘格的照片，目前有{len(corners)}张')
    result = solve(corners, size, meta['square_size_mm'])
    result.update(meta)
    result.update(calibrated_at=datetime.now().isoformat(), opencv_version=cv2.__version__,
                  images_used=names, rejected_images=rejected,
                  error_definition='sqrt(mean(du^2 + dv^2)), pixels',
                  note='重投影误差是拟合误差，不能单独证明标定准确；位姿平移单位为米。')
    write_json(folder/'intrinsics.json', result)
    rows = ['filename,rms_px'] + [f'{n},{e:.6f}' for n,e in zip(names,result['per_view_rms_px'])]
    (folder/'reprojection_errors.csv').write_text('\n'.join(rows)+'\n',encoding='utf-8-sig')
    im = read_image(folder/names[0])
    undist = cv2.undistort(im,np.array(result['camera_matrix']),np.array(result['dist_coeffs']))
    write_image(folder/'original_vs_undistorted.jpg',np.hstack([im,undist]))
    return result


class Camera:
    """按设备名称采集原生1280x720画面；读线程持续排空管道，只保留最新帧。"""
    def __init__(self, folder):
        exe = shutil.which('ffmpeg')
        if not exe:
            raise RuntimeError('没有找到FFmpeg，请检查环境')
        self.lock = threading.Lock()
        self.latest = None
        self.stop = threading.Event()
        self.frames = 0
        self.log = (folder/'camera.log').open('wb')
        cmd = [exe,'-hide_banner','-loglevel','warning','-f','dshow','-rtbufsize','64M',
               '-video_size',f'{SIZE[0]}x{SIZE[1]}','-framerate','30','-vcodec','mjpeg',
               '-i',f'video={CAMERA}','-an','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
        try:
            self.process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=self.log,
                stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:
            self.log.close()
            raise
        self.thread = threading.Thread(target=self._read,daemon=True)
        self.thread.start()

    def _read(self):
        count = SIZE[0]*SIZE[1]*3
        while not self.stop.is_set():
            data = bytearray()
            while len(data)<count and not self.stop.is_set():
                block = self.process.stdout.read(count-len(data))
                if not block:
                    return
                data.extend(block)
            if len(data)!=count:
                return
            frame = np.frombuffer(data,np.uint8).reshape(SIZE[1],SIZE[0],3).copy()
            with self.lock:
                self.frames += 1
                self.latest = (time.monotonic(), frame)

    def sample(self):
        with self.lock:
            return self.latest

    def close(self):
        if self.stop.is_set():
            return
        self.stop.set()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.thread.join(timeout=2)
        self.process.stdout.close()
        self.log.close()


class App:
    def __init__(self, root, square_mm, resume=None):
        self.root = root
        root.title('第一个任务：USB 相机内参标定')
        root.geometry('1120x850')
        root.minsize(800,640)
        recovered = []
        if resume is not None:
            self.folder = Path(resume).resolve()
            meta = json.loads((self.folder/'session.json').read_text(encoding='utf-8'))
            if (meta['camera_name'] != CAMERA or tuple(meta['image_size']) != SIZE
                    or tuple(meta['pattern_inner_corners']) != PATTERN
                    or meta['square_size_mm'] != square_mm):
                raise ValueError('旧会话的相机、分辨率或棋盘格参数不匹配，不能混合采集')
            paths = sorted(self.folder.glob('frame_*.png'))
            for path in paths:
                im = read_image(path)
                c = detect(im)
                if (im.shape[1], im.shape[0]) != SIZE or c is None:
                    raise ValueError(f'旧照片无法完整检测或分辨率不匹配：{path.name}')
                recovered.append(c)
            self.next_number = max([int(p.stem.split('_')[1]) for p in paths], default=0) + 1
        else:
            self.folder = HERE/'data'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.folder.mkdir(parents=True)
            write_json(self.folder/'session.json',dict(camera_name=CAMERA,image_size=list(SIZE),
                pattern_inner_corners=list(PATTERN),square_size_mm=square_mm,
                created_at=datetime.now().isoformat(),focus_note='采集和使用时保持焦距、分辨率、缩放一致'))
            self.next_number = 1
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.future_kind = None
        self.detected = None
        self.saved = recovered
        self.closed = False
        self.last_state = 0
        self.started = time.monotonic()
        self.preview_size = None
        self.calibrated_count = 0
        if (self.folder/'intrinsics.json').exists():
            old_result = json.loads((self.folder/'intrinsics.json').read_text(encoding='utf-8'))
            self.calibrated_count = len(old_result.get('images_used', []))
        self.camera = None
        style = ttk.Style(root)
        style.configure('TButton',font=('Microsoft YaHei UI',11),padding=8)
        style.configure('TLabel',font=('Microsoft YaHei UI',11))
        main = ttk.Frame(root,padding=14)
        main.pack(fill='both',expand=True)
        ttk.Label(main,text='USB 相机内参标定',font=('Microsoft YaHei UI',19,'bold')).pack(anchor='w')
        ttk.Label(main,text=f'{CAMERA}  ·  1280×720  ·  6×8 内角点  ·  单格 {square_mm:g} mm').pack(anchor='w',pady=(4,8))
        # A Label adopts every PhotoImage's natural size. Resizing the next frame
        # from that requested size creates a feedback loop that grows the window.
        # Canvas image items do not participate in geometry requests.
        self.video = tk.Canvas(main,bg='#18212e',highlightthickness=0,width=960,height=540)
        self.video.pack(fill='both',expand=True)
        self.video_image = self.video.create_image(480,270,anchor='center')
        self.video_message = self.video.create_text(480,270,text='正在打开 USB 相机……',
                                                    fill='white',font=('Microsoft YaHei UI',11))
        self.status = tk.StringVar(value='等待画面：请关闭 Windows「相机」等占用摄像头的软件。')
        ttk.Label(main,textvariable=self.status,wraplength=1050).pack(anchor='w',pady=(10,3))
        self.count = tk.StringVar(value=f'已保存 {len(self.saved)} 张 / 至少15张，建议20～25张')
        ttk.Label(main,textvariable=self.count).pack(anchor='w')
        self.tip = tk.StringVar(value='让整块棋盘格完整入镜，停稳后按空格或点击保存。只采集相机，不控制机械臂。')
        ttk.Label(main,textvariable=self.tip,wraplength=1050,foreground='#155e75').pack(anchor='w',pady=(4,8))
        buttons = ttk.Frame(main)
        buttons.pack(fill='x')
        self.save_btn = ttk.Button(buttons,text='保存照片（空格）',command=self.save)
        self.save_btn.pack(side='left',padx=(0,8))
        self.cal_btn = ttk.Button(buttons,text='计算内参（C）',command=self.calibrate)
        self.cal_btn.pack(side='left',padx=(0,8))
        ttk.Button(buttons,text='重新连接相机',command=self.reconnect).pack(side='left',padx=(0,8))
        ttk.Button(buttons,text='打开照片文件夹',command=self.open_folder).pack(side='left',padx=(0,8))
        ttk.Button(buttons,text='退出并释放相机',command=self.close).pack(side='right')
        ttk.Label(main,text='拍摄建议：中央、四周、远近、左右/上下倾斜；每次都停稳、完整、清晰。',foreground='#555555').pack(anchor='w',pady=(8,0))
        root.bind('<space>',lambda event:self.save())
        root.bind('<KeyPress-c>',lambda event:self.calibrate())
        root.bind('<KeyPress-C>',lambda event:self.calibrate())
        root.bind('<Escape>',lambda event:self.close())
        root.protocol('WM_DELETE_WINDOW',self.close)
        try:
            self.camera = Camera(self.folder)
        except Exception:
            self.pool.shutdown(wait=False,cancel_futures=True)
            raise
        self.tick()

    def reconnect(self):
        if self.future_kind == 'calibrate':
            self.tip.set('请等待标定计算完成后再重新连接。')
            return
        try:
            if self.camera is not None:
                self.camera.close()
            self.detected = None
            self.camera = Camera(self.folder)
            self.started = time.monotonic()
            self.tip.set('正在重新连接USB相机，请确保其他相机应用已关闭。')
        except Exception as exc:
            self.tip.set(f'重新连接失败：{exc}')

    def open_folder(self):
        import os
        os.startfile(str(self.folder))

    def save(self):
        if self.future_kind == 'calibrate':
            return
        item = self.detected
        if item is None or item[2] is None or time.monotonic()-item[0]>.8:
            self.tip.set('暂未保存：需要检测到全部48个角点，请完整入镜并停稳。')
            return
        stamp,frame,corners = item
        for old in self.saved:
            distance = min(float(np.sqrt(np.mean(np.sum((corners-old)**2,axis=2)))),
                           float(np.sqrt(np.mean(np.sum((corners-old[::-1])**2,axis=2)))))
            if distance<8:
                self.tip.set('暂未保存：与已有照片太相似，请改变位置、远近或倾斜角度。')
                return
        number = self.next_number
        try:
            write_image(self.folder/f'frame_{number:03d}.png',frame)
            shown = frame.copy()
            cv2.drawChessboardCorners(shown,PATTERN,corners,True)
            write_image(self.folder/'corners'/f'corners_{number:03d}.jpg',shown)
        except Exception as exc:
            messagebox.showerror('保存失败',str(exc),parent=self.root)
            return
        self.saved.append(corners.copy())
        self.next_number += 1
        self.count.set(f'已保存 {len(self.saved)} 张 / 至少15张，建议20～25张')
        suffix = '；已有标定结果需重新计算以包含新照片' if self.calibrated_count else ''
        self.tip.set(f'第{number}张已保存。请换一个位置或倾角，再停稳拍摄{suffix}。')

    def calibrate(self):
        if self.future_kind == 'calibrate':
            return
        if len(self.saved)<15:
            self.tip.set(f'还需要至少 {15-len(self.saved)} 张照片；建议总共20～25张。')
            return
        # The one-worker queue finishes the pending detection before calibration.
        self.future = self.pool.submit(calibrate_session,self.folder)
        self.future_kind = 'calibrate'
        self.save_btn.state(['disabled'])
        self.cal_btn.state(['disabled'])
        self.tip.set('正在后台计算内参，请稍候。相机预览继续显示。')

    def tick(self):
        if self.closed:
            return
        now = time.monotonic()
        latest = self.camera.sample()
        if self.future is not None and self.future.done():
            kind = self.future_kind
            try:
                result = self.future.result()
                if kind == 'detect':
                    self.detected = result
                else:
                    self.calibrated_count = len(result['images_used'])
                    warning = ' '.join(result['warnings'])
                    self.tip.set(f"结果已保存：RMS {result['rms_px']:.3f} 像素。{warning}")
                    messagebox.showinfo('标定结果已保存',
                        f"使用{self.calibrated_count}张照片\n重投影RMS：{result['rms_px']:.4f}像素\n"
                        f"{warning}\n\n结果：{self.folder/'intrinsics.json'}\n低误差仍需结合照片质量与去畸变效果检查。",parent=self.root)
            except Exception as exc:
                self.tip.set(f'处理失败：{exc}')
            self.future,self.future_kind = None,None
            self.save_btn.state(['!disabled'])
            self.cal_btn.state(['!disabled'])
        fresh = latest is not None and now-latest[0]<2
        if fresh:
            stamp,frame = latest
            if self.future is None:
                self.future = self.pool.submit(lambda s,f:(s,f,detect(f)),stamp,frame)
                self.future_kind = 'detect'
            # Show the exact detected frame when fresh, so saved image matches preview.
            found = self.detected is not None and self.detected[2] is not None and now-self.detected[0]<.8
            if found and self.future_kind!='calibrate':
                shown = self.detected[1].copy()
                cv2.drawChessboardCorners(shown,PATTERN,self.detected[2],True)
                self.status.set('已找到全部 48 个角点 ✓  棋盘格停稳后即可保存。')
            else:
                shown = frame
                self.status.set('相机已连接；请让完整棋盘格入镜并停稳，等待彩色角点。' if self.future_kind!='calibrate' else '正在计算内参，实时相机画面正常。')
            w,h = max(2,self.video.winfo_width()),max(2,self.video.winfo_height())
            picture = Image.fromarray(cv2.cvtColor(shown,cv2.COLOR_BGR2RGB))
            picture.thumbnail((w,h),Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(picture)
            self.preview_size = (w,h)
            self.video.coords(self.video_image,w//2,h//2)
            self.video.itemconfigure(self.video_image,image=self.photo)
            self.video.itemconfigure(self.video_message,state='hidden')
        else:
            found = False
            if now-self.started>12 or self.camera.process.poll() is not None:
                self.status.set('相机暂无画面：请关闭 Windows「相机」，然后点击“重新连接相机”。')
            self.save_btn.state(['disabled'])
        if fresh and self.future_kind!='calibrate':
            self.save_btn.state(['!disabled'])
        if now-self.last_state>1:
            self.last_state=now
            write_json(HERE/'app_state.json',dict(status='running',pid=__import__('os').getpid(),
                updated_at=datetime.now().isoformat(),camera_frames=self.camera.frames,
                camera_fresh=fresh,board_found=found,saved_count=len(self.saved),
                calibrated_count=self.calibrated_count,session=str(self.folder)))
        self.root.after(40,self.tick)

    def close(self):
        if self.closed:
            return
        self.closed=True
        if self.camera is not None:
            self.camera.close()
        self.pool.shutdown(wait=False,cancel_futures=True)
        write_json(HERE/'app_state.json',dict(status='closed',updated_at=datetime.now().isoformat(),session=str(self.folder),saved_count=len(self.saved)))
        self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--square-mm',type=float,default=SQUARE_MM)
    parser.add_argument('--resume',type=Path,help='继续已有采集会话，保留原照片并追加')
    parser.add_argument('--calibrate',type=Path,help='从已有采集目录离线计算，不打开相机')
    parser.add_argument('--probe',action='store_true',help='只读取USB相机并检测角点，保存一张诊断图后释放')
    parser.add_argument('--smoke-test',action='store_true',help='创建并关闭Tk窗口，不连接相机')
    args=parser.parse_args()
    if not np.isfinite(args.square_mm) or args.square_mm<=0:
        parser.error('格子边长必须是正数')
    if args.calibrate:
        print(json.dumps(calibrate_session(args.calibrate.resolve()),ensure_ascii=False,indent=2))
        return
    if args.smoke_test:
        root=tk.Tk();root.withdraw()
        im=ImageTk.PhotoImage(Image.new('RGB',(32,32)),master=root)
        ttk.Label(root,image=im,text='预览测试').pack();root.update();root.destroy()
        print('Tk/Pillow preview: PASS')
        return
    if args.probe:
        folder=HERE/'diagnostics';folder.mkdir(exist_ok=True)
        camera=Camera(folder)
        try:
            deadline=time.monotonic()+12
            while camera.sample() is None and time.monotonic()<deadline:
                time.sleep(.1)
            if camera.sample() is None:
                raise RuntimeError('相机没有画面，请检查diagnostics/camera.log')
            time.sleep(.5)
            frame=camera.sample()[1]
            c=detect(frame)
            write_image(folder/'probe.jpg',frame)
            if c is not None:
                overlay=frame.copy();cv2.drawChessboardCorners(overlay,PATTERN,c,True)
                write_image(folder/'probe_corners.jpg',overlay)
            print(json.dumps(dict(camera=CAMERA,size=SIZE,board_found=c is not None,corners=0 if c is None else len(c)),ensure_ascii=False))
        finally:
            camera.close()
        return
    root=tk.Tk()
    app=None
    try:
        app=App(root,args.square_mm,args.resume)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror('标定程序出错',str(exc),parent=root)
        raise
    finally:
        if app is not None and not app.closed:
            app.close()


if __name__=='__main__':
    main()
