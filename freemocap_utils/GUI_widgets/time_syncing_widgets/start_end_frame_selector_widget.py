

from PyQt6.QtWidgets import QWidget,QPushButton,QVBoxLayout, QLineEdit,QFormLayout,QLabel
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import pyqtSignal

class FrameSelectorWidget(QWidget):
    frame_intervals_updated_signal = pyqtSignal()
    def __init__(self, freemocap_start_end_frames:list, qualisys_start_end_frames:list):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        text_width = 100
        self.freemocap_starting_frame_line = QLineEdit()
        self.freemocap_starting_frame_line.setValidator(QIntValidator())
        self.freemocap_starting_frame_line.setFixedWidth(text_width)

        self.freemocap_ending_frame_line = QLineEdit()
        self.freemocap_ending_frame_line.setValidator(QIntValidator())
        self.freemocap_ending_frame_line.setFixedWidth(text_width)

        self.qualisys_starting_frame_line = QLineEdit()
        self.qualisys_starting_frame_line.setValidator(QIntValidator())
        self.qualisys_starting_frame_line.setFixedWidth(text_width)

        self.qualisys_ending_frame_line = QLineEdit()
        self.qualisys_ending_frame_line.setValidator(QIntValidator())
        self.qualisys_ending_frame_line.setFixedWidth(text_width)

        frame_form = QFormLayout()
        frame_form.addRow(QLabel('FreeMoCap Starting Frame'), self.freemocap_starting_frame_line)
        frame_form.addRow(QLabel('FreeMoCap Ending Frame'), self.freemocap_ending_frame_line)

        frame_form.addRow(QLabel('Qualisys Starting Frame'), self.qualisys_starting_frame_line)
        frame_form.addRow(QLabel('Qualisys Ending Frame'), self.qualisys_ending_frame_line)

        self.set_start_and_end_frames(freemocap_start_end_frames=freemocap_start_end_frames, qualisys_start_end_frames=qualisys_start_end_frames)

        self._layout.addLayout(frame_form)

        self.submitButton = QPushButton('Update Plot')
        self.submitButton.pressed.connect(self.submit_frames)
        self._layout.addWidget(self.submitButton)

    def submit_frames(self):
        self.freemocap_start_end_frames_new = [int(self.freemocap_starting_frame_line.text()),int(self.freemocap_ending_frame_line.text())]
        self.qualisys_start_end_frames_new = [int(self.qualisys_starting_frame_line.text()),int(self.qualisys_ending_frame_line.text())]

        self.frame_intervals_updated_signal.emit()

    def set_start_and_end_frames(self, freemocap_start_end_frames, qualisys_start_end_frames):
        self.freemocap_starting_frame_line.setText(str(freemocap_start_end_frames[0]))
        self.freemocap_ending_frame_line.setText(str(freemocap_start_end_frames[1]))
        self.qualisys_starting_frame_line.setText(str(qualisys_start_end_frames[0]))
        self.qualisys_ending_frame_line.setText(str(qualisys_start_end_frames[1]))





