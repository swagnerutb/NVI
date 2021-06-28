import sys
import os
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import QObject, QRectF, Qt
from PySide2.QtWidgets import QMainWindow, QFileDialog, QWidget, QCheckBox, QVBoxLayout, QLineEdit, QLabel, QPushButton

# Check OS
import platform
if(platform.system() == 'Darwin'): #if running on mac
    os.environ['QT_MAC_WANTS_LAYER'] = '1'


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.setWindowTitle("Initialize vgosDBpy")

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)


        self.nameLabel = QLabel(self.centralWidget)
        self.line = QLineEdit(self.centralWidget)
        self.file_select_button = QPushButton(self.centralWidget)
        self.graph_check = QtWidgets.QCheckBox(self.centralWidget)
        self.run_prog = QPushButton(self.centralWidget)

        self.line.setGeometry(QtCore.QRect(140, 25, 300, 20))
        self.nameLabel.setGeometry(QtCore.QRect(20, 25, 100, 20))

        self.file_select_button.setGeometry(QtCore.QRect(10, 10, 100, 100))
        self.graph_check.setGeometry(QtCore.QRect(10, 10, 200, 200))
        self.run_prog.setGeometry(QtCore.QRect(10, 10, 200, 200))
        
        self.nameLabel.setText("Selected .wrp file:")
        self.file_select_button.setText("Load .wrp file")
        self.graph_check.setText("Run the graphical interface")
        self.run_prog.setText("Run")

        self.gridLayout.addWidget(self.nameLabel,0,0)
        self.gridLayout.addWidget(self.line,0,2)
        self.gridLayout.addWidget(self.file_select_button,1,0)
        self.gridLayout.addWidget(self.graph_check,1,2)
        self.gridLayout.addWidget(self.run_prog,3,0,1,4)

        self.graph_intface = False
        
        MainWindow.setCentralWidget(self.centralWidget)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        QMainWindow.__init__(self)

        # Initialize UI
        self.setupUi(self)
        self.file_select_button.clicked.connect(self._file_path)
        self.graph_check.setCheckState(QtCore.Qt.Unchecked)
        self.graph_check.stateChanged.connect(self._check_graph)
        self.run_prog.clicked.connect(self._run_vgosDB)

    def tr(self, text):
        return QObject.tr(self, text)

    def _file_path(self):
        path = os.environ.get('VGOSDB_DIR')
        path_to_wrp, _ = QFileDialog.getOpenFileName(self, self.tr("Load .wrp file"), self.tr(path), self.tr("Wrap files (*.wrp)"))
        
        if(path_to_wrp != ""):
            self.line.setText(path_to_wrp)
    
    def _check_graph(self):
        self.graph_intface = (False == self.graph_intface)

    def _run_vgosDB(self):
        cmd = "python -m vgosDBpy "+self.line.text()
        cmd = "python3 -m vgosDBpy "+self.line.text()
        if(self.graph_intface == True):
            cmd = cmd+" -g"
        
        app.quit()

        os.system(cmd)


if __name__ == "__main__":
    path = os.environ.get('VGOSDB_DIR')

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())