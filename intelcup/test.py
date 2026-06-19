import cv2
import sys

# 1. 打印OpenCV版本
print("OpenCV 版本:", cv2.__version__)
print("Python 版本:", sys.version)

# 2. 读取图片测试（替换为本地一张图片路径）
img_path = "test.jpg"
img = cv2.imread(img_path)

if img is None:
    print("图片读取失败，请检查路径！")
else:
    print("图片尺寸：宽={},高={},通道数={}".format(img.shape[1], img.shape[0], img.shape[2]))
    # 弹出窗口显示图片，按任意键关闭
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 3. 摄像头实时测试（有摄像头设备可用）
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("摄像头打开成功，按 q 退出窗口")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera", frame)
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("未检测到摄像头，跳过摄像头测试")

#../build/bin/odaslive -c myArray.cfg
