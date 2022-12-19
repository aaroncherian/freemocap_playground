import cv2
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog

from pathlib import Path
class LoadVideo(QWidget):
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout (self._layout)

        self.videoLoadButton = QPushButton('Load a video',self)
        self.videoLoadButton.setEnabled(False)
        self._layout.addWidget(self.videoLoadButton)
        self.videoLoadButton.clicked.connect(self.load_video)

        self.video_is_loaded = False
    def set_session_folder_path(self,session_folder_path:Path):
        self.session_folder_path = session_folder_path

    def load_video(self):
        self.folder_diag = QFileDialog()
        self.video_path,filter  = QFileDialog.getOpenFileName(self, 'Open file', directory = str(self.session_folder_path))

        if self.video_path:
            self.vid_capture_object = cv2.VideoCapture(str(self.video_path))
            self.video_is_loaded = True

        
        f = 2
class VideoCapture(QWidget):
    def __init__(self):
        super().__init__()
        # self.cap = cv2.VideoCapture(str(filename))

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.video_loader = LoadVideo()
        self._layout.addWidget(self.video_loader)

        self.video_frame = QLabel()
        self._layout.addWidget(self.video_frame)
        # parent.layout.addWidget(self.video_frame)
    #
    def set_frame(self,frame_number:int):
        self.video_loader.vid_capture_object.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
        ret, frame = self.video_loader.vid_capture_object.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)

        QtGui.QPixmap()
        pix = QtGui.QPixmap.fromImage(img)
        resizeImage = pix.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_frame.setPixmap(resizeImage)

        f = 2

    # def show_frame(self, frame_number: int):
    #     self.cap.set(2,frame_number)
    #     ret, frame = self.cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)
    #
    #     QtGui.QPixmap()
    #     pix = QtGui.QPixmap.fromImage(img)
    #     resizeImage = pix.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
    #     self.video_frame.setPixmap(resizeImage)
    #     f=2
