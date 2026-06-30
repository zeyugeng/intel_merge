import time
import serial

# =========================
# Basic configuration
# =========================
PORT = "/dev/ttyUSB0"
BAUD = 115200

PAN_ID = 1  # horizontal servo ID
TILT_ID = 2  # vertical servo ID

# =========================
# Calibration results
# =========================
# Pan calibration:
# -90 deg -> P0975
#   0 deg -> middle value, estimated by (-90 PWM and +90 PWM)
# +90 deg -> P2270
PAN_N90_PWM = 975
PAN_0_PWM = (975 + 2270) / 2
PAN_P90_PWM = 2270

# Tilt calibration:
# -90 deg -> P0815
#   0 deg -> middle value, estimated by (-90 PWM and +90 PWM)
# +90 deg -> P2150
TILT_N90_PWM = 815
TILT_0_PWM = (815 + 2150) / 2
TILT_P90_PWM = 2150

# Angle limits
PAN_MIN_ANGLE = -90.0
PAN_MAX_ANGLE = 90.0

TILT_MIN_ANGLE = -90.0
TILT_MAX_ANGLE = 90.0

# If direction is reversed, change False to True
PAN_INVERT = False
TILT_INVERT = False

# Movement time
DEFAULT_TIME = 1000


def clamp(x, low, high):
    return max(low, min(high, x))


def angle_to_pwm(angle, n90_pwm, zero_pwm, p90_pwm, invert=False):
    """
    Convert angle to PWM using 3-point calibration:
        -90 deg -> n90_pwm
          0 deg -> zero_pwm
        +90 deg -> p90_pwm
    """
    if invert:
        angle = -angle

    angle = clamp(angle, -90.0, 90.0)

    if angle >= 0:
        pwm = zero_pwm + angle / 90.0 * (p90_pwm - zero_pwm)
    else:
        pwm = zero_pwm + angle / 90.0 * (zero_pwm - n90_pwm)

    return int(round(pwm))


def pwm_to_angle(pwm, n90_pwm, zero_pwm, p90_pwm, invert=False):
    """
    Convert PWM back to angle using 3-point calibration.
    """
    pwm = float(pwm)

    if pwm >= zero_pwm:
        angle = (pwm - zero_pwm) / (p90_pwm - zero_pwm) * 90.0
    else:
        angle = (pwm - zero_pwm) / (zero_pwm - n90_pwm) * 90.0

    angle = clamp(angle, -90.0, 90.0)

    if invert:
        angle = -angle

    return angle


def make_cmd(servo_id, pwm, t):
    return f"#{servo_id:03d}P{int(pwm):04d}T{int(t)}!"


class PanTilt:
    def __init__(self, port=PORT, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(1)

        self.pan_pwm = int(round(PAN_0_PWM))
        self.tilt_pwm = int(round(TILT_0_PWM))

    def move_pwm(self, pan_pwm, tilt_pwm, t=DEFAULT_TIME):
        """
        Move by raw PWM values.
        """
        pan_pwm = int(
            round(
                clamp(
                    pan_pwm,
                    min(PAN_N90_PWM, PAN_P90_PWM),
                    max(PAN_N90_PWM, PAN_P90_PWM),
                )
            )
        )
        tilt_pwm = int(
            round(
                clamp(
                    tilt_pwm,
                    min(TILT_N90_PWM, TILT_P90_PWM),
                    max(TILT_N90_PWM, TILT_P90_PWM),
                )
            )
        )

        cmd = make_cmd(PAN_ID, pan_pwm, t) + make_cmd(TILT_ID, tilt_pwm, t)
        print("SEND:", cmd)

        self.ser.write(cmd.encode("ascii"))

        self.pan_pwm = pan_pwm
        self.tilt_pwm = tilt_pwm

        time.sleep(t / 1000 + 0.2)

    def move_angle(self, pan_angle, tilt_angle, t=DEFAULT_TIME):
        """
        Move by angle.

        pan_angle:
            -90 to +90 deg

        tilt_angle:
            -90 to +90 deg
        """
        pan_pwm = angle_to_pwm(
            pan_angle, PAN_N90_PWM, PAN_0_PWM, PAN_P90_PWM, PAN_INVERT
        )

        tilt_pwm = angle_to_pwm(
            tilt_angle, TILT_N90_PWM, TILT_0_PWM, TILT_P90_PWM, TILT_INVERT
        )

        self.move_pwm(pan_pwm, tilt_pwm, t)

    def get_current_angle(self):
        """
        Return current recorded target angle.
        This is not servo hardware feedback.
        """
        pan_angle = pwm_to_angle(
            self.pan_pwm, PAN_N90_PWM, PAN_0_PWM, PAN_P90_PWM, PAN_INVERT
        )

        tilt_angle = pwm_to_angle(
            self.tilt_pwm, TILT_N90_PWM, TILT_0_PWM, TILT_P90_PWM, TILT_INVERT
        )

        return round(pan_angle, 2), round(tilt_angle, 2)

    def center(self, t=DEFAULT_TIME):
        self.move_angle(0, 0, t)

    def close(self):
        self.ser.close()


def print_help():
    print("Commands:")
    print("  m pan tilt [time]          move by angle")
    print("                            example: m 30 -20 800")
    print("  p pan_pwm tilt_pwm [time]  move by PWM")
    print("                            example: p 1800 1300 800")
    print("  c                          center, move to 0 deg, 0 deg")
    print("  g                          get current recorded angle")
    print("  h                          help")
    print("  q                          quit")
    print()


def main():
    pt = PanTilt(PORT, BAUD)

    print("Pan-tilt controller started.")
    print(
        f"Pan calibration:  -90 -> {PAN_N90_PWM}, 0 -> {PAN_0_PWM:.1f}, +90 -> {PAN_P90_PWM}"
    )
    print(
        f"Tilt calibration: -90 -> {TILT_N90_PWM}, 0 -> {TILT_0_PWM:.1f}, +90 -> {TILT_P90_PWM}"
    )
    print_help()

    try:
        while True:
            s = input(">>> ").strip()

            if not s:
                continue

            parts = s.split()
            cmd = parts[0].lower()

            try:
                if cmd == "q":
                    break

                elif cmd == "h":
                    print_help()

                elif cmd == "c":
                    pt.center(DEFAULT_TIME)
                    print("Current angle:", pt.get_current_angle())

                elif cmd == "g":
                    print("Current angle:", pt.get_current_angle())

                elif cmd == "m":
                    pan_angle = float(parts[1])
                    tilt_angle = float(parts[2])
                    t = int(parts[3]) if len(parts) >= 4 else DEFAULT_TIME

                    pt.move_angle(pan_angle, tilt_angle, t)
                    print("Current angle:", pt.get_current_angle())

                elif cmd == "p":
                    pan_pwm = int(parts[1])
                    tilt_pwm = int(parts[2])
                    t = int(parts[3]) if len(parts) >= 4 else DEFAULT_TIME

                    pt.move_pwm(pan_pwm, tilt_pwm, t)
                    print("Current angle:", pt.get_current_angle())

                else:
                    print("Unknown command. Type h for help.")

            except IndexError:
                print("Command format error. Type h for help.")
            except ValueError:
                print("Value format error. Use numbers, for example: m 30 -20 800")
            except Exception as e:
                print("Error:", e)

    finally:
        pt.center(DEFAULT_TIME)
        pt.close()
        print("Closed.")


if __name__ == "__main__":
    main()
