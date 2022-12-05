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
