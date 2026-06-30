import cv2
from ultralytics import YOLO
# 假设Camera是你封装的自定义摄像头类，保留原有导入
from Camera import Camera

# 加载YOLO模型
model = YOLO("yolo26n.pt")
# 初始化摄像头
camera = Camera()
camera.set_zoom(1)


# 实时循环检测
while True:
    # 获取画面帧
    frame = camera.get_frame()
    # 判断帧是否有效（防止摄像头断开黑屏报错）
    if frame is None:
        print("未读取到摄像头画面，退出程序")
        break
    
    h, w = frame.shape[:2]
    scale = 0.4

    frame_show = cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    
    # YOLO推理
    result = model(frame_show)
    # 绘制检测框、标签
    draw_img = result[0].plot()
    
    # 窗口显示
    cv2.imshow("YOLO Real-time Detect", draw_img)
    
    # 按键监听：按 q / ESC 退出窗口
    # waitKey返回按键ASCII码，&0xFF兼容不同系统
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# 循环结束后释放资源
camera.release()
cv2.destroyAllWindows()