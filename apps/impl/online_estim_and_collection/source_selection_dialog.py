
import logging

try:
    from PySide6.QtCore import Qt, QCoreApplication
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
    QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
except ImportError:
    pass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class SourceSelectDialog(QDialog):
    """Simple 2-button popup to choose data source for skeletons."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Choose Skeleton Source")
        if parent is not None:
            parent.destroyed.connect(self.close)
        self.selected = None  # "kinect" or "mmwave"

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Use Kinect skeletons or mmWave prediction?"))
        btns = QHBoxLayout()
        self.btn_kinect = QPushButton("Kinect", self)
        self.btn_mmwave = QPushButton("mmWave", self)
        btns.addWidget(self.btn_kinect)
        btns.addWidget(self.btn_mmwave)
        layout.addLayout(btns)

        self.btn_kinect.clicked.connect(self._choose_kinect)
        self.btn_mmwave.clicked.connect(self._choose_mmwave)

    def _choose_kinect(self):
        self.selected = "kinect"
        self.accept()

    def _choose_mmwave(self):
        self.selected = "mmwave"
        self.accept()