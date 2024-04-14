import threading

import hand_detection as hd
import granular_effect as ge
import mapping as map

mapping = map.Mapping()
granular = ge.GranularEffect(mapping, "audio/Sunrise.wav")
granular_thread = threading.Thread(target=granular.start)
granular_thread.start()

# init the hand detection class
# hand_detector = hd.HandDetection()
# hand_detector.run()

granular_thread.join()
