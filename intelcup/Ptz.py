import time
import serial


class PTZ:
    def __init__(self,
        port="/dev/ttyUSB1",
        baud=115200,
        pan_id=1,
        tilt_id=2,
        pan_cal=(975, 2270),     # -90, +90
        tilt_cal=(815, 2150),    # -90, +90
        pan_invert=False,
        tilt_invert=False,
        default_time=1000,):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(1)

        self.pan_id = pan_id
        self.tilt_id = tilt_id
        self.default_time = default_time

        self.pan_n90, self.pan_p90 = pan_cal
        self.tilt_n90, self.tilt_p90 = tilt_cal

        self.pan_0 = (self.pan_n90 + self.pan_p90) / 2
        self.tilt_0 = (self.tilt_n90 + self.tilt_p90) / 2

        self.pan_invert = pan_invert
        self.tilt_invert = tilt_invert

        self.pan_pwm = round(self.pan_0)
        self.tilt_pwm = round(self.tilt_0)

    @staticmethod
    def _clamp(x, low, high):
        return max(low, min(high, x))

    @staticmethod
    def _make_cmd(servo_id, pwm, t):
        return f"#{servo_id:03d}P{int(pwm):04d}T{int(t)}!"

    def _angle_to_pwm(self, angle, n90, zero, p90, invert=False):
        if invert:
            angle = -angle

        angle = self._clamp(angle, -90.0, 90.0)

        if angle >= 0:
            pwm = zero + angle / 90.0 * (p90 - zero)
        else:
            pwm = zero + angle / 90.0 * (zero - n90)

        return round(pwm)

    def _pwm_to_angle(self, pwm, n90, zero, p90, invert=False):
        pwm = float(pwm)

        if pwm >= zero:
            angle = (pwm - zero) / (p90 - zero) * 90.0
        else:
            angle = (pwm - zero) / (zero - n90) * 90.0

        angle = self._clamp(angle, -90.0, 90.0)

        return -angle if invert else angle

    def move_pwm(self, pan_pwm, tilt_pwm, t=None):
        """
        直接用 PWM 控制云台
        """
        if t is None:
            t = self.default_time

        pan_pwm = round(
            self._clamp(
                pan_pwm,
                min(self.pan_n90, self.pan_p90),
                max(self.pan_n90, self.pan_p90),
            )
        )

        tilt_pwm = round(
            self._clamp(
                tilt_pwm,
                min(self.tilt_n90, self.tilt_p90),
                max(self.tilt_n90, self.tilt_p90),
            )
        )

        cmd = (
            self._make_cmd(self.pan_id, pan_pwm, t)
            + self._make_cmd(self.tilt_id, tilt_pwm, t)
        )

        #print("SEND:", cmd)
        self.ser.write(cmd.encode("ascii"))

        self.pan_pwm = pan_pwm
        self.tilt_pwm = tilt_pwm

        time.sleep(t / 1000 + 0.2)

    def move_angle(self, pan_angle, tilt_angle, t=None):
        """
        用角度控制云台

        pan_angle:  -90 ~ +90
        tilt_angle: -90 ~ +90
        """
        pan_pwm = self._angle_to_pwm(
            pan_angle,
            self.pan_n90,
            self.pan_0,
            self.pan_p90,
            self.pan_invert,
        )

        tilt_pwm = self._angle_to_pwm(
            tilt_angle,
            self.tilt_n90,
            self.tilt_0,
            self.tilt_p90,
            self.tilt_invert,
        )

        self.move_pwm(pan_pwm, tilt_pwm, t)

    def get_current_angle(self):
        """
        返回当前记录的目标角度，不是舵机真实反馈角度
        """
        pan_angle = self._pwm_to_angle(
            self.pan_pwm,
            self.pan_n90,
            self.pan_0,
            self.pan_p90,
            self.pan_invert,
        )

        tilt_angle = self._pwm_to_angle(
            self.tilt_pwm,
            self.tilt_n90,
            self.tilt_0,
            self.tilt_p90,
            self.tilt_invert,
        )

        return round(pan_angle, 2), round(tilt_angle, 2)

    def center(self, t=None):
        """
        回到中心位置
        """
        self.move_angle(0, 0, t)

    def close(self):
        self.ser.close()