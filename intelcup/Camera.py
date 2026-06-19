import glob
import os


def _ensure_gui_display() -> None:
    """Cursor/SSH 终端常无 DISPLAY，从 Wayland 会话补齐 XWayland 环境。"""
    uid = os.getuid()
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
    if os.path.isdir(runtime):
        os.environ.setdefault("XDG_RUNTIME_DIR", runtime)

    if not os.environ.get("DISPLAY") and os.path.isdir("/tmp/.X11-unix"):
        os.environ.setdefault("DISPLAY", ":0")

    if not os.environ.get("XAUTHORITY"):
        candidates = sorted(glob.glob(f"{runtime}/.mutter-Xwaylandauth.*"))
        if candidates:
            os.environ["XAUTHORITY"] = candidates[-1]

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")


_ensure_gui_display()

import cv2

#ubuntu版本
class Camera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.zoom = 0
        self.cap = None

        self.WIDTH = 3840
        self.HEIGHT = 2160
        self.FPS = 30

    def get_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print("摄像头打开失败")
                return False

            # 很多 USB 摄像头在 4K 下需要 MJPG，否则可能打不开高分辨率
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.FPS)

            self.set_zoom(self.zoom)

        return True

    def set_zoom(self, zoom):
        if self.cap is None or not self.cap.isOpened():
            print("Camera is not opened")
            return False

        self.zoom = zoom
        success = self.cap.set(cv2.CAP_PROP_ZOOM, zoom)

        if not success:
            print("Failed to set zoom")

        return success

    def get_frame(self):
        if not self.get_camera():
            return None

        ret, frame = self.cap.read()

        if not ret:
            print("Failed to get frame")
            return None

        return frame

    def show(self, preview_w: int = 1280, preview_h: int = 720):
        if not self.get_camera():
            return

        window = "Camera 实时预览 (按 q 退出)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, preview_w, preview_h)
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"实时预览已启动: {real_w}x{real_h} -> 窗口 {preview_w}x{preview_h}")
        print("若看不到窗口，请 Alt+Tab 切换到「Camera 实时预览」")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("读取失败")
                break

            h, w = frame.shape[:2]
            scale = min(preview_w / w, preview_h / h)

            frame_show = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(window, frame_show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("f"):
                try:
                    top = cv2.getWindowProperty(window, cv2.WND_PROP_TOPMOST)
                    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 0 if top else 1)
                except cv2.error:
                    pass

        self.release()


    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="USB 摄像头实时预览")
    parser.add_argument("--id", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--width", type=int, default=1920, help="采集宽度，默认 1920")
    parser.add_argument("--height", type=int, default=1080, help="采集高度，默认 1080")
    args = parser.parse_args()

    camera = Camera(camera_id=args.id)
    camera.WIDTH = args.width
    camera.HEIGHT = args.height

    if not camera.get_camera():
        raise SystemExit(1)

    real_w = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = camera.cap.get(cv2.CAP_PROP_FPS)
    print(f"实际分辨率: {real_w} x {real_h}")
    print(f"实际FPS: {real_fps}")

    camera.show()