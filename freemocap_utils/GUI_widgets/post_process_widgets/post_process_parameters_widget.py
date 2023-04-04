from PyQt6.QtWidgets import QWidget,QVBoxLayout, QComboBox, QLineEdit, QFormLayout, QPushButton, QLabel, QHBoxLayout, QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import pyqtSignal

class ParameterEntryWidget(QWidget):

    def __init__(self):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        text_width = 100

        parameter_line_edits = {'frame rate': None, 'butterworth_filter_order':None, 'butterworth_filter_cutoff':None}

        for key in parameter_line_edits:

            parameter_line_edits[key] = QLineEdit()
            parameter_line_edits[key].setValidator(QIntValidator())
            parameter_line_edits[key].setFixedWidth(text_width)


        self.original_data_framerate = QLineEdit()
        self.original_data_framerate.setValidator(QIntValidator())
        self.original_data_framerate.setFixedWidth(text_width)

        self.butterworth_filter_order = QLineEdit()
        self.butterworth_filter_order.setValidator(QIntValidator())
        self.butterworth_filter_order.setFixedWidth(text_width)

        self.butterworth_filter_cutoff = QLineEdit()
        self.butterworth_filter_cutoff.setValidator(QIntValidator())
        self.butterworth_filter_cutoff.setFixedWidth(text_width)

        frame_form = QFormLayout()
        frame_form.addRow(QLabel('Starting Frame'), self.starting_frame)
        frame_form.addRow(QLabel('Ending Frame'), self.ending_frame)