import cv2
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class VideoCapture(QWidget):
    def __init__(self):
        super().__init__()
        # self.cap = cv2.VideoCapture(str(filename))
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.video_frame = QLabel('LABEL')
        self._layout.addWidget(self.video_frame)
        # parent.layout.addWidget(self.video_frame)
    #
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
