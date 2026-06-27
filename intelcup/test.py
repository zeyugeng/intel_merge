from Camera import Camera
from ultralytics import YOLO
import cv2

def main():
    # 1. 初始化摄像头
    cam = Camera(camera_id=0)
    cam.WIDTH = 1920
    cam.HEIGHT = 1080
    cam.FPS = 30

    if not cam.get_camera():
        print("摄像头打开失败，程序退出")
        return

    # 2. 加载YOLOv8模型
    model = YOLO("yolo26n.pt")  # 可替换 yolov8s.pt / yolov8m.pt

    window_name = "YOLOv8 Detection (q退出)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("开始推理画面，按下 q 关闭窗口")

    while True:
        # 读取一帧图像
        frame = cam.get_frame()
        if frame is None:
            break

        # 3. YOLO推理 + 绘制检测框
        results = model(frame, verbose=False)
        det_frame = results[0].plot()

        # 显示画面
        cv2.imshow(window_name, det_frame)

        # 按键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 释放资源
    cam.release()

if __name__ == "__main__":
    main()