import torch
import matplotlib.colors
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import spacenavigator
import time
import sys


class LatentVariablesTour:
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.setup_spacemouse()
        self.cnt = 0
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.update)

    def setup_spacemouse(self):
        n = len(spacenavigator.list_devices())
        i = 0
        print("Please keep using your SpaceMouse...")
        for _ in range(n * 20):
            success = spacenavigator.open(DeviceNumber=i)
            if success:
                time.sleep(0.1)
                state = spacenavigator.read()
                spacenavigator.close()
                if any((state.x, state.y, state.z)) or i == 4:
                    break
            i = (i + 1) % n
        else:
            sys.quit()
        print(f"device number = {i}")
        spacenavigator.open(DeviceNumber=i)

    def read_spmouse(self):
        state = spacenavigator.read()
        return np.array([state.x, state.y, state.z, state.roll, state.pitch, state.yaw])

    def setup_app_window(self):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow()
        self.layoutgb = QtGui.QGridLayout()
        self.win.setLayout(self.layoutgb)

    def plot_model_data(self, is_show_grid=False):
        colors = np.ones((10, 4))
        colors[:, :3] = (
            np.array(
                [
                    [int(i[1:3], 16), int(i[3:5], 16), int(i[5:7], 16)]
                    for i in matplotlib.colors.TABLEAU_COLORS.values()
                ]
            )
            / 255
        )

        self.w = gl.GLViewWidget(rotationMethod="quaternion")
        self.w.setCameraPosition(distance=20)

        if is_show_grid:
            self.g = gl.GLGridItem()
            self.w.addItem(self.g)
        for i in self.dataset.class_to_idx.values():
            with torch.no_grad():
                xyz = (
                    self.model.enc(
                        self.dataset.data[self.dataset.targets == i]
                        .reshape(-1, 1, 28, 28)[:5000]
                        .to(torch.float32)
                        .to(self.device)
                        / 255
                    )
                    .to("cpu")
                    .detach()
                    .numpy()
                )
            xyz *= 100
            size = np.full(len(xyz), 3)
            spi = gl.GLScatterPlotItem(pos=xyz, color=tuple(colors[i % 10]), size=size)
            self.w.addItem(spi)

        self.layoutgb.addWidget(self.w, 0, 0)

    def plot_origin(self):
        self.pos = np.zeros(3)
        size = np.full(1, 10)
        self.sp = gl.GLScatterPlotItem(pos=self.pos, color=(1, 1, 1, 1), size=size)
        self.w.addItem(self.sp)

        self.p1 = pg.PlotWidget()
        self.p1.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        self.layoutgb.addWidget(self.p1, 0, 1)

    def set_layout_balance(self):
        self.layoutgb.setColumnStretch(0, 8)
        self.layoutgb.setColumnStretch(1, 5)

    def update(self):
        rot = self.w.cameraParams()["rotation"]
        cos, sin = np.cos, np.sin
        mc = self.read_spmouse()
        a = mc[3] / 50 * np.pi
        rot = QtGui.QQuaternion(cos(a), 0, 0, -sin(a)) * rot
        a = mc[4] / 50 * np.pi
        rot = QtGui.QQuaternion(cos(a), -sin(a), 0, 0) * rot
        a = mc[5] / 50 * np.pi
        rot = QtGui.QQuaternion(cos(a), 0, -sin(a), 0) * rot

        self.pos += (lambda x: np.array([x.x(), x.y(), x.z()]))(
            rot.conjugate().rotatedVector(QtGui.QVector3D(mc[0], mc[2], -mc[1]))
        )
        self.w.setCameraParams(center=QtGui.QVector3D(*self.pos), rotation=rot)
        self.sp.setData(pos=self.pos)

        if self.cnt % 10 == 9:
            with torch.no_grad():
                im = (
                    self.model.dec(
                        torch.from_numpy(self.pos[None] / 100).float().to(self.device)
                    )
                    .cpu()
                    .detach()
                    .numpy()[0, 0]
                )
            im = (im - im.min()) / (im.max() - im.min())
            self.img.setImage(im[::-1, :].transpose(1, 0))
        self.cnt += 1

    def run(self):
        self.setup_app_window()
        self.plot_model_data()
        self.plot_origin()
        self.set_layout_balance()
        self.t.start(10)
        self.app.exec()
