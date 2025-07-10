
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui import  ui_infoDialog
import numpy as np

class InfoDlg(QDialog, ui_infoDialog.Ui_infoDialog):

    def __init__(self, resetWindowPosition=False, parent=None):

        #  set up the UI
        super(InfoDlg, self).__init__(parent)
        self.setupUi(self)


        #  set a few porperties of the SimpleTextEdit object that we can't set in designer
        self.statusText.setMaximumBlockCount(1000)
        self.statusText.setWordWrapMode(QTextOption.NoWrap)
        
        self.saveBtn.clicked.connect(self.close)
        self.__appSettings = QSettings('afsc.noaa.gov', 'infodlg')
        size = self.__appSettings.value('winsize', QSize(940,646))
        position = self.__appSettings.value('winposition', QPoint(10,10))
        if not resetWindowPosition:
            #  make sure the program is on the screen
            position, size = self.checkWindowLocation(position, size)

            #  now move and resize the window
            self.move(position)
            self.resize(size)

    def closeEvent(self, event=None):
        self.__appSettings.setValue('winposition', self.pos())
        self.__appSettings.setValue('winsize', self.size())
        self.close()

    def updateLog(self, text, color):
        if type(text)==type(np.array([])):
            for line in text:
                t=str(line)
                logText = '<text style="color:' + color +'">' + str(t[1:-1])
                self.statusText.appendHtml(logText)
        else:
            logText = '<text style="color:' + color +'">' + str(text)
            self.statusText.appendHtml(logText)

        #  ensure that the window is scrolled to see the new line(s) of text.
        self.statusText.verticalScrollBar().setValue(self.statusText.verticalScrollBar().maximum())



    def checkWindowLocation(self, position, size):
        '''
        checkWindowLocation accepts a window position (QPoint) and size (QSize)
        and returns a potentially new position and size if the window is currently
        positioned off the screen.
        '''

        #  determine the current virtual screen size
        screenObj = QGuiApplication.primaryScreen()
        screenGeometry = screenObj.availableVirtualGeometry()

        #  assume the new and old positions are the same
        newPosition = position
        newSize = size

        #  check if the upper left corner of the window is off the screen
        if position.x() < screenGeometry.x():
            newPosition.setX(10)
        if position.x() >= screenGeometry.x():
            newPosition.setX(10)
        #  check if the window title bar is off the top of the screen
        if position.y() < screenGeometry.y():
            newPosition.setY(10)
        #  check if the window title bar is off the bottom of the screen
        #  Subtract 50 pixels from the position to ensure that
        #  the title bar is clear of the taskbar
        if (position.y() - 50) >= screenGeometry.y():
            newPosition.setY(10)

        #  now make sure the lower right (resize handle) is on the screen
        if (newPosition.x() + newSize.width()) > screenGeometry.width():
            newSize.setWidth(screenGeometry.width() - newPosition.x() - 5)
        if (newPosition.y() + newSize.height()) > screenGeometry.height():
            newSize.setWidth(screenGeometry.height() - newPosition.y() - 5)

        return [newPosition, newSize]


