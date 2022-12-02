from PyQt6.QtCore import Qt
from PyQt6 import QtGui
from PyQt6.QtWidgets import QMainWindow, QSlider, QGridLayout, QWidget,QLabel, QFileDialog,QPushButton, QFormLayout
from PyQt6.QtWidgets import QApplication  



import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from pathlib import Path
import numpy as np
import cv2

from skeleton_builder_v3 import mediapipe_indices,mediapipe_connections,build_skeleton


class VideoCapture(QWidget):
    def __init__(self, filename, parent):
        super(QWidget, self).__init__()
        self.cap = cv2.VideoCapture(str(filename))
        self.video_frame = QLabel()
        parent.layout.addWidget(self.video_frame)

    def show_first_frame(self):
        self.cap.set(2,0)
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)

        QtGui.QPixmap()
        pix = QtGui.QPixmap.fromImage(img)
        resizeImage = pix.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_frame.setPixmap(resizeImage)

        f=2


    def update_frame(self,frame_number):
        self.cap.set(2,frame_number)
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        img = QtGui.QImage(frame, 300, 300, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

class VideoDisplayWidget(QWidget):
    def __init__(self,parent):
        super(VideoDisplayWidget, self).__init__(parent)

        self.layout = QFormLayout(self)

        self.startButton = QPushButton('Start', parent)
        self.startButton.setFixedWidth(50)
        self.startButton.clicked.connect(parent.show_first_frame)
        self.pauseButton = QPushButton('Pause', parent)
        self.layout.addRow(self.startButton, self.pauseButton)

        self.setLayout(self.layout)




class Mpl3DPlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111,projection = '3d')
        super(Mpl3DPlotCanvas, self).__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QGridLayout()
        print('starting')
        self.session_folder_path = None
        #self.anthropometric_info_dataframe = build_anthropometric_dataframe(segments,joint_connections,segment_COM_lengths,segment_COM_lengths)
        self.folderOpenButton = QPushButton('Load a session folder',self)
        layout.addWidget(self.folderOpenButton,0,0)
        self.folderOpenButton.clicked.connect(self.open_folder_dialog)

        self.videoLoadButton = QPushButton('Load a video',self)
        layout.addWidget(self.videoLoadButton,0,1)
        self.videoLoadButton.clicked.connect(self.load_video)
        self.is_video_loaded = False

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMaximum(0)


        layout.addWidget(self.slider,1,0,1,1)
        self.label = QLabel(str(self.slider.value()))
        layout.addWidget(self.label,2,0)
        
        self.initialize_skeleton_plot()


        self.slider.valueChanged.connect(self.replot)
        layout.addWidget(self.fig,3,0)

        
        self.videoDisplayWidget = VideoDisplayWidget(self)
        layout.addWidget(self.videoDisplayWidget,3,1)

        widget = QWidget()

        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def open_folder_dialog(self):
        
        self.folder_diag = QFileDialog()
        self.session_folder_path  = QFileDialog.getExistingDirectory(None,"Choose a session")

        if self.session_folder_path:
            self.session_folder_path = Path(self.session_folder_path)

            
            #data_array_folder = 'output_data'
            data_array_folder = 'DataArrays'
            array_name = 'mediaPipeSkel_3d_origin_aligned.npy'
            #array_name = 'mediaPipeSkel_3d.npy'
            #array_name = 'mediaPipeSkel_3d_origin_aligned.npy'
            #array_name = 'mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
            
            skeleton_data_folder_path = self.session_folder_path / data_array_folder/array_name
            self.skel3d_data = np.load(skeleton_data_folder_path)

            self.mediapipe_skeleton = build_skeleton(self.skel3d_data,mediapipe_indices,mediapipe_connections)

            self.num_frames = self.skel3d_data.shape[0]
            self.reset_slider()
            self.reset_skeleton_3d_plot()


            
    def load_video(self):
        self.folder_diag = QFileDialog()
        self.video_path,filter  = QFileDialog.getOpenFileName(self, 'Open file', directory = str(self.session_folder_path))
        
        if self.video_path:
            self.capture = VideoCapture(self.video_path, self.videoDisplayWidget)

    def show_first_frame(self):
        self.capture.show_first_frame()


        f = 2

    def initialize_skeleton_plot(self):
        #self.skel_x,self.skel_y,self.skel_z = self.get_x_y_z_data()
        self.fig = Mpl3DPlotCanvas(self, width=5, height=4, dpi=100)
        self.ax = self.fig.figure.axes[0]

    def reset_skeleton_3d_plot(self):
        self.ax.cla()
        self.calculate_axes_means(self.skel3d_data)
        self.skel_x,self.skel_y,self.skel_z = self.get_x_y_z_data()
        self.plot_skel(self.skel_x,self.skel_y,self.skel_z)


    def reset_slider(self):
        self.slider_max = self.num_frames -1
        self.slider.setValue(0)
        self.slider.setMaximum(self.slider_max)

    def calculate_axes_means(self,skeleton_3d_data):
        self.mx_skel = np.nanmean(skeleton_3d_data[:,0:33,0])
        self.my_skel = np.nanmean(skeleton_3d_data[:,0:33,1])
        self.mz_skel = np.nanmean(skeleton_3d_data[:,0:33,2])
        self.skel_3d_range = 900

    def plot_skel(self,skel_x,skel_y,skel_z):
        self.ax.scatter(skel_x,skel_y,skel_z)
        self.plot_skeleton_bones()
        self.ax.set_xlim([self.mx_skel-self.skel_3d_range, self.mx_skel+self.skel_3d_range])
        self.ax.set_ylim([self.my_skel-self.skel_3d_range, self.my_skel+self.skel_3d_range])
        self.ax.set_zlim([self.mz_skel-self.skel_3d_range, self.mz_skel+self.skel_3d_range])

        self.fig.figure.canvas.draw_idle()

    def plot_skeleton_bones(self):
            frame = self.slider.value()
            this_frame_skeleton_data = self.mediapipe_skeleton[frame]
            for connection in this_frame_skeleton_data.keys():
                line_start_point = this_frame_skeleton_data[connection][0] 
                line_end_point = this_frame_skeleton_data[connection][1]
                
                bone_x,bone_y,bone_z = [line_start_point[0],line_end_point[0]],[line_start_point[1],line_end_point[1]],[line_start_point[2],line_end_point[2]] 

                self.ax.plot(bone_x,bone_y,bone_z)

    def get_x_y_z_data(self):
        skel_x = self.skel3d_data[self.slider.value(),:,0]
        skel_y = self.skel3d_data[self.slider.value(),:,1]
        skel_z = self.skel3d_data[self.slider.value(),:,2]

        return skel_x,skel_y,skel_z

    def replot(self):
        skel_x,skel_y,skel_z = self.get_x_y_z_data()
        self.ax.cla()
        self.plot_skel(skel_x,skel_y,skel_z)
        self.label.setText(str(self.slider.value()))


if __name__ == "__main__":


    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
        # logger.info(f"`main` exited with error code: {error_code}")
        # win.close()
        # if error_code != EXIT_CODE_REBOOT:
        #     logger.info(f"Exiting...")
        #     break
        # else:
        #     logger.info("`main` exited with the 'reboot' code, so let's reboot!")