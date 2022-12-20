from PyQt6.QtWidgets import QWidget,QVBoxLayout, QComboBox, QLineEdit, QFormLayout, QPushButton, QLabel, QHBoxLayout, QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import pyqtSignal


class FrameMarker(QWidget):
    conditions_dict_updated_signal = pyqtSignal()
    def __init__(self):
        super().__init__()

        self._layout = QHBoxLayout()
        self.setLayout(self._layout)

        self.condition_widget_dictionary = {}
    
        conditions_layout = QVBoxLayout()
        self.conditions_box= QComboBox()

        self.conditions_box.addItems(['Eyes Open/Solid Ground', 'Eyes Closed/Solid Ground', 'Eyes Open/Foam', 'Eyes Closed/Foam'])
        
        conditions_layout.addWidget(self.conditions_box)

        text_width = 100
        self.starting_frame = QLineEdit()
        self.starting_frame.setValidator(QIntValidator())
        self.starting_frame.setFixedWidth(text_width)

        self.ending_frame = QLineEdit()
        self.ending_frame.setValidator(QIntValidator())
        self.ending_frame.setFixedWidth(text_width)

        frame_form = QFormLayout()
        frame_form.addRow(QLabel('Starting Frame'), self.starting_frame)
        frame_form.addRow(QLabel('Ending Frame'), self.ending_frame)

        conditions_layout.addLayout(frame_form)

        self.save_condition_button = QPushButton('Save Frame Interval For This Condition')
        self.save_condition_button.pressed.connect(self.save_conditions_to_table)
        conditions_layout.addWidget(self.save_condition_button)

        self._layout.addLayout(conditions_layout)

        self.saved_conditions_table = QTableWidget()
        self.saved_conditions_table.setRowCount(4)
        self.saved_conditions_table.setColumnCount(2)
        self._layout.addWidget(self.saved_conditions_table)

    def save_conditions_to_table(self):
        current_condition = self.conditions_box.currentText()
        start_frame = int(self.starting_frame.text())
        end_frame = int(self.ending_frame.text())

        self.condition_widget_dictionary[current_condition] = [start_frame,end_frame]
        self.saved_conditions_table.setVerticalHeaderLabels(list(self.condition_widget_dictionary.keys()))


        for row_count, condition in enumerate(self.condition_widget_dictionary.keys()):
            this_start_frame = self.condition_widget_dictionary[condition][0]
            this_end_frame = self.condition_widget_dictionary[condition][1]
            self.saved_conditions_table.setItem(row_count,0,QTableWidgetItem(str(this_start_frame)))
            self.saved_conditions_table.setItem(row_count,1,QTableWidgetItem(str(this_end_frame)))

        self.conditions_dict_updated_signal.emit()
    


        f = 2

    

