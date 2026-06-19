import math
from ultralytics import YOLO
import cv2
import time

from Camera import Camera
from MicrophoneArray import MicrophoneArray, SoundSource,SoundSourceFrame
from Ptz import PTZ
from SoundPredict import SoundPredict

def status_1():
    while True:
        sound_frame = mic.get_latest()

        if MicrophoneArray.is_silent_frame(sound_frame):
            time.sleep(0.02)
            continue

        # 遍历当前 frame，取出第一个非静音声源
        source = None
        for src in sound_frame.sources:
            if src.activity > 0.001:
                source = src
                break

        if source is None:
            continue

        x = source.x
        y = source.y
        z = source.z
        pan_angle = math.degrees(math.atan2(x, z))
        horizontal_distance = math.sqrt(x * x + z * z)
        tilt_angle = math.degrees(math.atan2(y, horizontal_distance))
        print(f"检测到声源: id={source.id}, activity={source.activity:.3f}, "
              f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
        print(f"云台转向: pan={pan_angle:.2f}, tilt={tilt_angle:.2f}")
        ptz.move_angle(pan_angle, tilt_angle)
        break

def status_2():
    model = YOLO("yolo26n.pt")#加载模型
    # 比例系数，越大云台追踪越激进
    pan_k = 25.0
    tilt_k = 18.0

    # 误差死区，避免鸟已经接近中心时云台来回抖动
    dead_zone = 0.05

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # 假设画面中只有一个鸟；如果有多个框，取置信度最高的
        best_box = max(boxes, key=lambda box: float(box.conf[0]))

        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        conf = float(best_box.conf[0])

        bird_cx = (x1 + x2) / 2
        bird_cy = (y1 + y2) / 2

        frame_cx = w / 2
        frame_cy = h / 2

        # 归一化偏移，范围大约是 -1 ~ 1
        error_x = (bird_cx - frame_cx) / frame_cx
        error_y = (bird_cy - frame_cy) / frame_cy

        pan_angle, tilt_angle = ptz.get_current_angle()

        if abs(error_x) > dead_zone:
            pan_angle += error_x * pan_k

        if abs(error_y) > dead_zone:
            # 图像 y 轴向下，所以目标在画面下方时，tilt 应该减小
            tilt_angle -= error_y * tilt_k

        pan_angle = max(-90, min(90, pan_angle))
        tilt_angle = max(-90, min(90, tilt_angle))

        print(
            f"鸟位置: cx={bird_cx:.1f}, cy={bird_cy:.1f}, conf={conf:.2f}, "
            f"error_x={error_x:.3f}, error_y={error_y:.3f}"
        )
        print(f"云台跟踪: pan={pan_angle:.2f}, tilt={tilt_angle:.2f}")

        ptz.move_angle(pan_angle, tilt_angle, t=200)

        # 显示检测框，方便调试
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )
        cv2.circle(
            frame,
            (int(bird_cx), int(bird_cy)),
            5,
            (0, 0, 255),
            -1
        )
        cv2.circle(
            frame,
            (int(frame_cx), int(frame_cy)),
            5,
            (255, 0, 0),
            -1
        )

        preview_w = 1280
        scale = preview_w / w
        preview = cv2.resize(frame, (preview_w, int(h * scale)))

        cv2.imshow("tracking", preview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("tracking")





#初始化
camera = Camera()
camera.get_camera()
print("摄像头就绪")
mic = MicrophoneArray(host="0.0.0.0", port=5000)
mic.connect(wait=True)
print("麦克风阵列就绪")
ptz = PTZ()
print("云台就绪")
#predictor=SoundPredict(species_list="species_list.txt")
#predictor.load_model()
#print("birdnet模型就绪")

print("全部设备就绪")


while True:
    status_1()
    input("测试功能1")



