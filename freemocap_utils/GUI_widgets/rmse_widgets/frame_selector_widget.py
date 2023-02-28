

from PyQt6.QtWidgets import QWidget,QPushButton,QVBoxLayout, QLineEdit,QFormLayout,QLabel
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import pyqtSignal

import numpy as np

class FrameSelectorWidget(QWidget):
    frame_intervals_updated_signal = pyqtSignal()
    def __init__(self, qualisys_data:np.ndarray, freemocap_data:np.ndarray):
        super().__init__()

        self.qualisys_data = qualisys_data
        self.freemocap_data = freemocap_data

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        text_width = 100
        self.starting_frame_line = QLineEdit()
        self.starting_frame_line.setValidator(QIntValidator())
        self.starting_frame_line.setFixedWidth(text_width)

        self.ending_frame_line = QLineEdit()
        self.ending_frame_line.setValidator(QIntValidator())
        self.ending_frame_line.setFixedWidth(text_width)


        frame_form = QFormLayout()
        frame_form.addRow(QLabel('Starting Frame'), self.starting_frame_line)
        frame_form.addRow(QLabel('Ending Frame'), self.ending_frame_line)

        self.starting_frame_line.setText(str(0))
        self.ending_frame_line.setText(str(freemocap_data.shape[0]))

        self._layout.addLayout(frame_form)

        self.submitButton = QPushButton('Update Plot')
        self.submitButton.pressed.connect(self.submit_frames)
        self._layout.addWidget(self.submitButton)

    def submit_frames(self):
        self.start_end_frames_new = [int(self.starting_frame_line.text()),int(self.ending_frame_line.text())]
        
        self._slice_data(self.start_end_frames_new)
        self.frame_intervals_updated_signal.emit()


    def _slice_data(self,start_end_frames_new:list):

        self.qualisys_data_sliced = self.qualisys_data[start_end_frames_new[0]:start_end_frames_new[1],:,:].copy()
        self.freemocap_data_sliced = self.freemocap_data[start_end_frames_new[0]:start_end_frames_new[1],:,:].copy()

    def get_sliced_qualisys_data(self):
        return self.qualisys_data_sliced

    def get_sliced_freemocap_data(self):
        return self.freemocap_data_sliced



    




