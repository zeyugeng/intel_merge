from Camera import Camera
from MicrophoneArray import MicrophoneArray, SoundSource,SoundSourceFrame
from Ptz import PTZ
from SoundPredict import SoundPredict
import cv2

import time

def testcamera():
    camera = Camera()
    camera.get_camera()
    frame1=camera.get_frame()
    cv2.imshow("Camera", frame1)
    camera.set_zoom(0)
    cv2.waitKey(0)
    camera.show()

def testptz(default_time=1000):
    ptz = PTZ()
    ptz.move_angle(90,0)
    input()
    ptz.move_angle(45,0)
    input()
    ptz.move_angle(-90,0)
    input()
    ptz.move_angle(0,45)
    input()
    ptz.move_angle(0,-60)
    input()
    ptz.move_angle(45,-45)
    input()
    ptz.center()

def testmic():
    mic = MicrophoneArray(host="0.0.0.0", port=5000)
    try:
        mic.connect(wait=False)

        while True:
            frame = mic.get_latest()

            if frame is not None:
                print("timeStamp:", frame.time_stamp)

                for src in frame.sources:
                    print(
                        f"id={src.id}, "
                        f"x={src.x:.3f}, y={src.y:.3f}, z={src.z:.3f}, "
                        f"activity={src.activity:.3f}"
                    )

                print("-" * 40)

            time.sleep(0.05)

    finally:
        mic.close()
    

def testsoundpredic():
    predictor = SoundPredict()
    predictor.load_model()
    print(1)
    result1 = predictor.predict("soundscape.wav")
    print(result1.to_structured_array())
    result2 = predictor.predict("soundscape.wav")
    print(result2.to_structured_array())
    predictor.close()

if __name__ == '__main__':
    testptz()
    
