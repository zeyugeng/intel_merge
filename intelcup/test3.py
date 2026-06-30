import time

from MicrophoneArray import odasconnecter, sssprocess, sstprocess


if __name__ == "__main__":
    odas = odasconnecter()
    sst = sstprocess()
    sss = sssprocess()

    sst.connect(wait=False)
    sss.start()
    time.sleep(0.2)
    odas.open_odas()

    last_save = time.time()

    try:
        while True:
            frame = sst.get_latest()
            if frame is not None:
                print("timeStamp:", frame.time_stamp)
                for src in frame.sources:
                    print(
                        f"id={src.id}, x={src.x:.3f}, y={src.y:.3f}, "
                        f"z={src.z:.3f}, activity={src.activity:.3f}"
                    )
                print("-" * 40)

            if time.time() - last_save >= 5:
                sss.save_last_3s_wav()
                last_save = time.time()

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("程序退出")
    finally:
        sst.close()
        sss.close()
        odas.close_odas()
