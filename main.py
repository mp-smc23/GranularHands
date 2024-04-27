import hand_detection as hd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1",
        help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=7400,
        help="The port the OSC server is listening on")
    args = parser.parse_args()

    # init the hand detection class
    hand_detector = hd.HandDetection(args.ip, args.port)
    hand_detector.run()

