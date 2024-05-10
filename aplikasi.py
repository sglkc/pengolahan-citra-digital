import os
import sys
import atexit
import shutil
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from I8_I11 import I8_I11 as Aplikasi

# Hapus folder __pycache__ karena import modul
def remove_pycache():
    current_directory = os.getcwd()
    pycache_path = os.path.join(current_directory, '__pycache__')

    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)

# Jalankan aplikasi
def main():
    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)

    atexit.register(remove_pycache)
    app = QApplication(sys.argv)
    window = Aplikasi()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
