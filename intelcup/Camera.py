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

    def show(self):
        if not self.get_camera():
            return

        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

        preview_w, preview_h = 1280, 720

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
                interpolation=cv2.INTER_AREA
            )

            cv2.imshow("Camera", frame_show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        self.release()


    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = Camera(camera_id=0)
    camera.get_frame()
    # 打印实际生效的分辨率
    real_w = camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    real_fps = camera.cap.get(cv2.CAP_PROP_FPS)
    print(f"实际分辨率: {int(real_w)} x {int(real_h)}")
    print(f"实际FPS: {real_fps}")

    frame = camera.get_frame()
    if frame is not None:
        print("图像shape(高,宽,通道)：", frame.shape)

    camera.show()