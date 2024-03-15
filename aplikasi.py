import atexit
import os
import shutil
import sys
from H1_H3 import H1_H3 as Aplikasi
from PyQt5.QtWidgets import QApplication

# Hapus folder __pycache__ karena import modul
def remove_pycache():
    current_directory = os.getcwd()
    pycache_path = os.path.join(current_directory, '__pycache__')

    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)

# Jalankan aplikasi
def main():
    atexit.register(remove_pycache)
    app = QApplication(sys.argv)
    window = Aplikasi()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
