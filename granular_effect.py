import pyo
import time
import mapping as map

class GranularEffect:
    def __init__(self, mapping, audio_file):
        self.s = pyo.Server(sr=44100, nchnls=2).boot() 
        self.s.start()
        self.snd = pyo.SndTable(audio_file)
        self.env = pyo.HannTable()
        self.phasor = pyo.Phasor(freq=self.snd.getRate()*.25, mul=self.snd.getSize())
        self.grain = pyo.Granulator(self.snd, 
                                    env=self.env, 
                                    pitch=1, 
                                    pos=self.phasor, 
                                    dur=0.1, 
                                    grains=1, 
                                    basedur=0.1)
        
        self.mapping = mapping

    def start(self):
        self.grain.out()
        try:
            while True:
                self.update_parameters()
                pass
        except KeyboardInterrupt:
            print("Interrupted by user")
            self.s.stop()

    def update_parameters(self):
        # window_kaiser = scipy.signal.windows.tukey(M, 0.5)
        # env = pyo.SndTable(window_kaiser)
        # self.grain.setEnv(env)
        self.grain.setPitch(self.mapping.pitch)
        self.grain.setDur(self.mapping.dur)
        self.grain.setBaseDur(self.mapping.basedur)
        self.grain.setGrains(self.mapping.grains)



    

