import scipy

class Mapping:
    def __init__(self) -> None:
        self.grains = 1 
        self.basedur = 0.1
        self.dur = 0.1
        self.env = 0
        self.pitch = 1

    def update_parameters(self, y_offset_left, y_offset_right, left_right_distance, left_angle, right_angle):
        self.grains = self.map_y_offset_left(y_offset_left)
        self.basedur = self.map_y_offset_right(y_offset_right)
        self.dur = self.map_left_right_distance(left_right_distance)
        self.env = self.map_left_angle(left_angle)
        self.pitch = self.map_right_angle(right_angle)

    def map_y_offset_left(self, param):
        # map param from (0,1) to (1,32)
        return 1 + param * 31
    
    def map_y_offset_right(self, param):
        # map param from (0,1) to (0.1,1)
        return 0.1 + param * 0.9
    
    def map_left_right_distance(self, param):
        # map param from (0,1) to (0.3, 0.01)
        return 0.3 - param * 0.29
    
    def map_left_angle(self, param):
        # map param from (-1,1) to (0.1, 1)
        return (param + 1 / 2) * 0.9 + 0.1
    
    def map_right_angle(self, param):
        # map param from (-1,1) to (0.5, 1.5)
        return (param + 1) / 2 + 0.5


