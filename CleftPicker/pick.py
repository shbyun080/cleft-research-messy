import cv2

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QFileDialog, QScrollArea, QMessageBox
from PyQt5.QtCore import QUrl, QEventLoop, Qt
import pyqtgraph as pg

import backend


class ImagePlot(pg.GraphicsLayoutWidget):

    def __init__(self, db=None, image=None):
        super(ImagePlot, self).__init__()

        self.image = image
        self.image_file = None
        self.image_item = None

        self.orig_point = []
        self.db = db

        self.curr_point = -1
        self.last_points = []

        self.p1 = pg.PlotItem()
        self.addItem(self.p1)
        self.p1.vb.invertY(True)  # Images need inverted Y axis

        # Use ScatterPlotItem to draw points
        self.scatterItem = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 255, 255)
        )

        self.scatterItem.setZValue(2)  # Ensure scatterPlotItem is always at top
        self.points = []
        for l in landmark_labels:
            self.points.append([l, []])

        self.p1.addItem(self.scatterItem)

        direction.setText("Select Landmark")

    def set_image(self, image):

        self.p1.clear()
        self.p1.addItem(self.scatterItem)

        self.image_item = pg.ImageItem(image)
        self.image_item.setOpts(axisOrder='row-major')
        self.p1.addItem(self.image_item)

        self.image = image

    def next_image(self, image, image_file=None):
        self.set_image(image)

        self.points = []
        for l in landmark_labels:
            self.points.append([l, []])
        self.scatterItem.setData(pos=[])

        self.image_file = image_file

        # TODO fetch data from database and load self.orig_point
        db_points = backend.fetch_data(self.db, image_file)
        self.orig_point = []
        if db_points is not None:
            for l in landmark_labels:
                self.orig_point.append([l, []])
            for i in range(0, len(landmark_labels)):
                if db_points[2 * i] is not None:
                    self.orig_point[i][1] = [db_points[2 * i],
                                             db_points[2 * i + 1]]
                    landmark_buttons[landmark_map_reversed[i]].setStyleSheet("background-color : green")

        # initialize points if needed
        if len(self.orig_point) != 0:
            self.points = self.orig_point.copy()
            self.scatterItem.setData(pos=[i[1] for i in self.points if len(i[1]) != 0])

    def proceed(self):
        global proceed_ok

        # FULL POINTS REQUIRED mod
        # for l, p in self.points:
        #     if len(p)==0:
        #         if self.curr_point != -1:
        #             direction.setText("You have not marked all the points.\nSelect "+landmark_labels[self.curr_point])
        #         else:
        #             direction.setText("You have not marked all the points.")
        #         return

        record_data(self.db, [[0, 0] if len(i[1]) == 0 else i[1] for i in self.points], self.image_file)
        proceed_ok = True

    def add_data(self, x, y):

        self.points[landmark_map[self.curr_point]][1] = [x, y]
        self.scatterItem.setData(pos=[i[1] for i in self.points if len(i[1]) != 0])

        t = QLabel(landmark_labels[self.curr_point] + ": " + str(x) + ", " + str(y))
        text_layout.addWidget(t)

        landmark_buttons[self.curr_point].setStyleSheet("background-color : green")

        self.last_points.append(self.curr_point)
        self.curr_point = -1

        for l, p in self.points:
            if len(p) == 0:
                direction.setText("Select Landmark")
                return
        direction.setText("Press Confirm")

    def mousePressEvent(self, event):
        point = self.p1.vb.mapSceneToView(event.pos())  # get the point clicked
        # Get pixel position of the mouse click
        x, y = point.x(), point.y()

        if event.button() == Qt.LeftButton and self.curr_point != -1:
            self.add_data(x, y)

        super().mousePressEvent(event)

    def landmark_select(self, i):
        self.curr_point = i
        landmark_buttons[i].setStyleSheet("background-color : yellow")
        direction.setText("Select " + new_landmark_labels[i])


def record_data(db, points, image_file):
    try:
        backend.insert_data(db, image_file, [i for s in points for i in s])
    except Exception:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText('Backend Error: Cannot connect to AWS')
        msg.setWindowTitle("Error")
        msg.exec_()
        record_data_csv(points, image_file)


def record_data_csv(points, image_file):
    a = open("cleft_landmarks.csv", "a")
    a.write(image_file + "," + ",".join([str(i) for s in points for i in s]) + "\n")


def clear_layout(layout):
    for i in reversed(range(layout.count())):
        widgetToRemove = layout.itemAt(i).widget()
        # remove it from the layout list
        layout.removeWidget(widgetToRemove)
        # remove it from the gui
        widgetToRemove.setParent(None)


def add_buttons(layout):
    global landmark_buttons
    landmark_buttons = []
    for i in range(len(new_landmark_labels)):
        landmark_buttons.append(QPushButton(new_landmark_labels[i]))
        landmark_buttons[-1].clicked.connect(button_click(i))
        landmark_buttons[-1].setStyleSheet("background-color : red")
        layout.addWidget(landmark_buttons[-1], i % ((len(landmark_labels) + 1) // 2), 2 * i // len(landmark_labels))


def button_click(i):
    return lambda: imageplot_main.landmark_select(i)


def full_reset():
    imageplot_main.points = []
    for l in landmark_labels:
        imageplot_main.points.append([l, []])
    for i in range(len(landmark_buttons)):
        landmark_buttons[i].setStyleSheet("background-color : red")
    imageplot_main.scatterItem.setData(pos=[])
    imageplot_main.curr_point = -1
    clear_layout(text_layout)
    direction.setText("Select Landmark")


# revert to database original
def reset():
    imageplot_main.points = []
    if len(imageplot_main.orig_point) != 0:
        imageplot_main.points = imageplot_main.orig_point.copy()
        for i in range(len(imageplot_main.points)):
            if len(imageplot_main.points[i]) != 0:
                landmark_buttons[landmark_map_reversed[i]].setStyleSheet("background-color : green")
            else:
                landmark_buttons[landmark_map_reversed[i]].setStyleSheet("background-color : red")
    else:
        for l in landmark_labels:
            imageplot_main.points.append([l, []])
        for i in range(len(landmark_buttons)):
            landmark_buttons[i].setStyleSheet("background-color : red")

    imageplot_main.scatterItem.setData(pos=[i[1] for i in imageplot_main.points if len(i[1]) != 0])
    imageplot_main.curr_point = -1
    clear_layout(text_layout)
    direction.setText("Select Landmark")


def undo():
    if len(imageplot_main.last_points) == 0:
        return
    imageplot_main.points[landmark_map[imageplot_main.last_points[-1]]][1] = []
    landmark_buttons[imageplot_main.last_points[-1]].setStyleSheet("background-color : red")
    imageplot_main.curr_point = -1
    del imageplot_main.last_points[-1]
    imageplot_main.scatterItem.setData(pos=[i[1] for i in imageplot_main.points if len(i[1]) != 0])
    direction.setText("Select Landmark")


def skip():
    full_reset()
    global loop
    loop.quit()


def confirm():
    global proceed_ok
    imageplot_main.proceed()
    if proceed_ok:
        full_reset()
        global loop
        loop.quit()


def main():
    global direction
    global text_layout
    global imageplot_main

    global loop

    global proceed_ok

    db = backend.get_db()

    # Main Window
    main_win = QWidget()
    main_layout = QHBoxLayout()
    main_win.setLayout(main_layout)
    win.setCentralWidget(main_win)
    win.setWindowTitle("Cleft Marker")

    # Left window
    central_win = QWidget()
    central_layout = QVBoxLayout()
    central_win.setLayout(central_layout)
    main_layout.addWidget(central_win)

    # Pane for instruction
    direction_win = QWidget()
    direction_layout = QVBoxLayout()
    direction_win.setLayout(direction_layout)
    central_layout.addWidget(direction_win)

    # Instruction
    direction = QLabel()
    direction_layout.addWidget(direction)

    # Pane for image viewing & selecting
    image_win = QWidget()
    image_layout = QHBoxLayout()
    image_win.setLayout(image_layout)
    central_layout.addWidget(image_win)

    image_win.setFixedHeight(500)

    # Two image panes for image pane
    imageplot_main = ImagePlot(db)
    image_layout.addWidget(imageplot_main)

    # Text space for recorded points for image
    text_win = QWidget()
    text_layout = QVBoxLayout()
    text_win.setLayout(text_layout)

    scroll = QScrollArea()
    scroll.setWidget(text_win)
    scroll.setWidgetResizable(True)
    central_layout.addWidget(scroll)

    # Buttons space
    button_win = QWidget()
    button_layout = QHBoxLayout()
    button_win.setLayout(button_layout)
    central_layout.addWidget(button_win)

    # Button for full clearing
    reset_button = QPushButton("Reset All")
    reset_button.clicked.connect(full_reset)
    button_layout.addWidget(reset_button)

    # Button for reverting back
    revert_button = QPushButton("Revert to Original")
    revert_button.clicked.connect(reset)
    button_layout.addWidget(revert_button)

    # Button for undoing
    undo_button = QPushButton("Undo")
    undo_button.clicked.connect(undo)
    button_layout.addWidget(undo_button)

    # Button for undoing
    skip_button = QPushButton("Skip")
    skip_button.clicked.connect(skip)
    button_layout.addWidget(skip_button)

    # Button for completing
    confirm_button = QPushButton("Confirm")
    confirm_button.clicked.connect(confirm)
    central_layout.addWidget(confirm_button)

    # RIGHT WINDOW
    right_win = QWidget()
    right_layout = QGridLayout()
    right_win.setLayout(right_layout)
    main_layout.addWidget(right_win)

    add_buttons(right_layout)

    win.show()

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("How to Start")
    msg.setInformativeText('General Usage\n\tSelect image file(s) from your computer\n\tSelect landmark to select\n\tMark desired points\n\tPress confirm to save\nManipulating Image\n\tHold left click to pan\n\tHold right click to skew\n\tScroll mouse wheel to zoom')
    msg.setWindowTitle("Introduction")
    msg.exec_()

    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(central_win, "QFileDialog.getOpenFileNames()", "", "Images (*.png *.jpg)", options=options)

    if not files:
        app.exit(app.exec_())
        return

    for img_path in files:
        proceed_ok = False

        url = QUrl.fromLocalFile(img_path)
        filename = url.fileName()

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imageplot_main.next_image(img, filename)

        loop = QEventLoop()
        loop.exec()

    app.quit()


if __name__ == "__main__":
    rootDir = '../../data/Unilateral/'

    landmark_labels = ['Face: Top Left Corner', 'Face: Bottom Right Corner',
                       "c'ala", "sn", "c'cphs", "nc'ch", "nc'ala",
                       "nc'chps", "c'c", "c'sbal", "nc'sbal", "c'ch",
                       "nc'c", "prn", "ls", "mchpi", "c'rl",
                       "sto", "c'nt1", "c'nt2", "lcphi", "m'rl", "c'cphi"]

    new_landmark_labels = ['Face: Top Left Corner', 'Face: Bottom Right Corner',
                           "c'ala", "c'sbal", "c'c", "c'cphs", "sn", "nc'cphs",
                           "nc'c", "nc'sbal", "nc'ala", "prn", "nc'ch", "mchpi",
                           "ls", "lcphi", "m'rl", "sto", "c'ch", "c'cphi", "c'rl",
                           "c'nt1", "c'nt2"]

    landmark_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 9,
        4: 8,
        5: 4,
        6: 3,
        7: 7,
        8: 12,
        9: 10,
        10: 6,
        11: 13,
        12: 5,
        13: 15,
        14: 14,
        15: 20,
        16: 21,
        17: 17,
        18: 11,
        19: 22,
        20: 16,
        21: 18,
        22: 19
    }

    landmark_map_reversed = {
        0: 0,
        1: 1,
        2: 2,
        3: 6,
        4: 5,
        5: 12,
        6: 10,
        7: 7,
        8: 4,
        9: 3,
        10: 9,
        11: 18,
        12: 8,
        13: 11,
        14: 14,
        15: 13,
        16: 20,
        17: 17,
        18: 21,
        19: 22,
        20: 15,
        21: 16,
        22: 19
    }

    app = QApplication([])
    win = QMainWindow()

    main()
