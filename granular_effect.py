import pyo
import time


s = pyo.Server(sr=48000, nchnls=2, buffersize=512, duplex=1).boot() 
s.start()
snd = pyo.SndTable("audio/Sunrise.wav")
env = pyo.HannTable()
pos = pyo.Phasor(freq=snd.getRate()*.25, mul=snd.getSize())
dur = pyo.Noise(mul=.001, add=.1)
g = pyo.Granulator(snd, env, [1, 1.001], pos, dur, 32, mul=.1).out()

try:
    while True:
        pass
except KeyboardInterrupt:
    print("Interrupted by user")
    s.stop()

import pyo
import time

class GranularEffect:
    def __init__(self, audio_file):
        self.s = pyo.Server(sr=48000, nchnls=2, buffersize=512, duplex=1).boot() 
        self.s.start()
        self.snd = pyo.SndTable(audio_file)
        self.env = pyo.HannTable()
        self.phasor = pyo.Phasor(freq=self.snd.getRate()*.25, mul=self.snd.getSize())
        self.grain = pyo.Granulator(self.snd, 
                                    env=self.env, 
                                    pitch=1, 
                                    pos=self.phasor, 
                                    dur=0.1, 
                                    grains=32, 
                                    basedur=0.1).out()

    def start(self):
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Interrupted by user")
            self.s.stop()

    def update_parameters(self, env=pyo.HannTable(), pitch=1, duration=0.1, basedur=0.1, density=8):
        self.grain.setEnv(env)
        self.grain.setPitch(pitch)
        self.grain.setDur(duration)
        self.grain.setBaseDur(basedur)
        self.grain.setGrains(density)



    

