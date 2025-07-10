
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui import  ui_ImgFileDlg



class ImageFileDlg(QDialog, ui_ImgFileDlg.Ui_ImgFileDlg):
    def __init__(self, parent=None):
        super(ImageFileDlg, self).__init__(parent)
        self.setupUi(self)
        
        self.okBtn.clicked.connect(self.OKGo)
        self.cancelBtn.clicked.connect(self.cancelGo)
        self.browseBtn.clicked.connect(self.getDir)
        
        self.typeBox.addItems(['.jpg', '.png', '.tif'])
        self.typeBox.setCurrentIndex(0)
        
        self.__appSettings = QSettings('afsc.noaa.gov', 'ImgFileDlg')
        self.__lBase = self.__appSettings.value('left_base', '')
        self.__rBase = self.__appSettings.value('right_base', '')
        self.__calName = self.__appSettings.value('cal_name', '')
        self.__dataDir= self.__appSettings.value('datadir', '')
        self.__rowNum= self.__appSettings.value('row_num', '')
        self.__colNum= self.__appSettings.value('col_num', '')
        self.__scale= self.__appSettings.value('scale', '')
        self.lBaseEdit.setText(self.__lBase)
        self.rBaseEdit.setText(self.__rBase)
        self.nameEdit.setText(self.__calName)
        self.pathEdit.setText(self.__dataDir)
        self.rowNumEdit.setText(self.__rowNum)
        self.colNumEdit.setText(self.__colNum)
        self.scaleEdit.setText(self.__scale)


    def getDir(self):
        dirDlg = QFileDialog(self)
        dirName = dirDlg.getExistingDirectory(self, 'Select Calibration Image Directory', self.__dataDir,
                                              QFileDialog.ShowDirsOnly)
        self.pathEdit.setText(dirName)

    def OKGo(self):
        # do some checks
        self.__appSettings.setValue('winposition', self.pos())
        self.__appSettings.setValue('winsize', self.size())
        self.__appSettings.setValue('left_base', self.lBaseEdit.text())
        self.__appSettings.setValue('right_base', self.rBaseEdit.text())
        self.__appSettings.setValue('cal_name', self.nameEdit.text())
        self.__appSettings.setValue('datadir', self.pathEdit.text())
        self.__appSettings.setValue('row_num', self.rowNumEdit.text())
        self.__appSettings.setValue('col_num', self.colNumEdit.text())
        self.__appSettings.setValue('scale', self.scaleEdit.text())


        self.accept()

    def cancelGo(self):
        self.reject()


