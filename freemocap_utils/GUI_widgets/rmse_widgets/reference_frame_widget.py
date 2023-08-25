from PyQt6.QtWidgets import QWidget,QPushButton,QHBoxLayout, QLineEdit,QFormLayout,QLabel
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import pyqtSignal

import numpy as np

class ReferenceFrameWidget(QWidget):
    reference_frame_updated_signal = pyqtSignal(object)
    reset_signal = pyqtSignal()
    def __init__(self):
        super().__init__()

        self._layout = QHBoxLayout()
        self.setLayout(self._layout)

        # self._layout.addWidget(QLabel('Reference Frame'))

        text_width = 100
        self.reference_frame_line = QLineEdit()
        self.reference_frame_line.setValidator(QIntValidator())
        self.reference_frame_line.setFixedWidth(text_width)
        self._layout.addWidget(self.reference_frame_line)

        self.submitButton = QPushButton('Set Reference Frame')
        self.submitButton.pressed.connect(self.set_reference_frame)
        self._layout.addWidget(self.submitButton)

        self.resetButton = QPushButton('Reset Reference Frame')
        self.resetButton.pressed.connect(self.reset_reference_frame)
        self._layout.addWidget(self.resetButton)

    def set_reference_frame(self):
        self.reference_frame = int(self.reference_frame_line.text())
        self.reference_frame_updated_signal.emit(self.reference_frame)
    
    def reset_reference_frame(self):
        self.reference_frame_line.setText('')
        self.reset_signal.emit()