import numpy as np

class PathLengthCalculator():

    def __init__(self,freemocap_data:np.ndarray):
        self.freemocap_data = freemocap_data

    def slice_data(self, freemocap_data, num_frame_range):
        sliced_freemocap_data = freemocap_data[num_frame_range[0]:num_frame_range[-1],:]
        return sliced_freemocap_data

    def calculate_path_length(self, sliced_freemocap_data):
        path_length = 0
        for frame_number in range(1,len(sliced_freemocap_data)):
            path_length += self.calculate_distance(sliced_freemocap_data[frame_number-1],sliced_freemocap_data[frame_number])

        normalized_path_length = path_length/len(sliced_freemocap_data)
        return normalized_path_length

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

    def get_path_length(self,num_frame_range):
        self.sliced_freemocap_data = self.slice_data(self.freemocap_data,num_frame_range)
        self.path_length = self.calculate_path_length(self.sliced_freemocap_data)
        return self.path_length

    def calculate_velocity(self, num_frame_range):
        sliced_freemocap_data = self.slice_data(self.freemocap_data,num_frame_range)
        velocity_data = []

        for dimension in range(3):
            velocity_data.append(np.diff(sliced_freemocap_data[:,dimension]))
        # x_velocity_data = np.diff(sliced_freemocap_data[:,0])
        # y_velocity_data = np.diff(sliced_freemocap_data[:,1])

        return velocity_data 