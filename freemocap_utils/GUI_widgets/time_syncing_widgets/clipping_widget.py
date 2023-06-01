from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QFormLayout, QLabel
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import pyqtSignal



class ClipWidget(QWidget):
    data_clipped_signal = pyqtSignal()
    save_clipped_data_signal = pyqtSignal(object,object)
    def __init__(self, qualisys_3d_data, freemocap_3d_data, qualisys_com_data):
        super().__init__()

        self.qualisys_3d_data = qualisys_3d_data
        self.freemocap_3d_data = freemocap_3d_data
        self.qualisys_com_data = qualisys_com_data

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        text_width = 100
        self.lag_entry_line = QLineEdit()
        self.lag_entry_line.setValidator(QIntValidator())
        self.lag_entry_line.setFixedWidth(text_width)

        lag_form = QFormLayout()
        lag_form.addRow(QLabel('Lag:'), self.lag_entry_line)
        self._layout.addLayout(lag_form)

        self.clip_data_button = QPushButton('Clip Qualisys Data')
        self.clip_data_button.pressed.connect(self.clip_qualisys_data)
        self._layout.addWidget(self.clip_data_button)

        self.save_clipped_data_button = QPushButton('Save Clipped Qualisys Data')
        self.save_clipped_data_button.pressed.connect(self.save_clipped_data)
        self.save_clipped_data_button.setEnabled(False)
        self._layout.addWidget(self.save_clipped_data_button)


    def clip_qualisys_data(self):
        lag = int(self.lag_entry_line.text())

        qualisys_start_frame = lag
        qualisys_end_frame = self.freemocap_3d_data.shape[0] + lag

        self.clipped_qualisys_data = self.qualisys_3d_data[qualisys_start_frame:qualisys_end_frame,:,:]
        self.qualisys_start_end_frames_new = [qualisys_start_frame,qualisys_end_frame]
        self.freemocap_start_end_frames_new = [0, self.freemocap_3d_data.shape[0]]

        assert self.freemocap_3d_data.shape[0] == self.clipped_qualisys_data.shape[0], f'FreeMoCap and clipped Qualisys array lengths are not equal.'

        self.clipped_qualisys_com_data = self.qualisys_com_data[qualisys_start_frame:qualisys_end_frame,:]
        self.save_clipped_data_button.setEnabled(True)

        self.data_clipped_signal.emit()

    def save_clipped_data(self):

        self.save_clipped_data_signal.emit(self.clipped_qualisys_data,self.clipped_qualisys_com_data)


