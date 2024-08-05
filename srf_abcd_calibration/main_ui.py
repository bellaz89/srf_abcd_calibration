# SRF ABCD calibration program for superconducting accelerating cavity's
# LLRF system
# Copyright (C) 2024 Andrea Bellandi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import tkinter
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import os

class App(tkinter.Frame):
    def __init__(self, root, station_picker):

        self.station_picker = station_picker
        self.station = tkinter.StringVar(root)
        self.station.set(self.station_picker.pick_first())

        self.root = root

        param_frame = ttk.Frame(root, padding="3 3 12 12")
        param_frame.columnconfigure(tuple(range(6)), weight=1)

        result_frame = ttk.Frame(root, padding="3 3 12 12")
        result_frame.columnconfigure(tuple(range(0, 8, 2)), weight=1)
        result_frame.columnconfigure(tuple(range(1, 8, 2)), weight=3)

        self.fig_plots, axes = plt.subplots(nrows=2, ncols=2,
                                            sharex=True, figsize=(12, 8))
        ((self.ax_IQ, self.ax_det), (self.ax_FR, self.ax_bw)) = axes

        self.toolbar_frame = tkinter.Frame(root)
        self.toolbar_frame.pack(side=tkinter.TOP, fill=tkinter.X)

        self.canvas = FigureCanvasTkAgg(self.fig_plots, self.toolbar_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        ui = ttk.Label(root, text="Parameters:")
        ui.pack()

        param_frame.pack(expand=1, fill="both", padx=(3, 10))

        ui = ttk.Label(root, text="Results:")
        ui.pack()

        result_frame.pack(expand=1, fill="both", padx=(3, 10))

        param_frame['borderwidth'] = 2
        param_frame['relief'] = 'groove'

        result_frame['borderwidth'] = 2

        result_frame['relief'] = 'groove'

        ui = ttk.Label(param_frame, text="F0(Hz):")
        ui.grid(column=0, row=1, sticky=tkinter.E)

        self.f0 = ttk.Label(result_frame, text="")
        self.f0.grid(row=1, column=1, sticky=tkinter.W)

        ui = ttk.Label(param_frame, text="Fs(Hz):")
        ui.grid(column=2, row=1, sticky=tkinter.E)

        self.fs = ttk.Label(param_frame, text="")
        self.fs.grid(row=1, column=1, sticky=tkinter.W)

        ui = ttk.Label(param_frame, text="Station:")
        ui.grid(column=4, row=1, sticky=tkinter.E)

        ui = ttk.OptionMenu(param_frame, self.station, None, *self.station_picker.station_names(),
                            command=lambda x: self.clear())
        ui.grid(column=5, row=1, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="QL:")
        ui.grid(row=1, column=2, sticky=tkinter.E)

        self.QL = ttk.Label(result_frame, text="")
        self.QL.grid(row=1, column=3, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="Half bandwidth(Hz):")
        ui.grid(row=1, column=4, sticky=tkinter.E)

        self.hbw = ttk.Label(result_frame, text="")
        self.hbw.grid(row=1, column=5, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="x:")
        ui.grid(row=2, column=2, sticky=tkinter.E)

        self.x = ttk.Label(result_frame, text="")
        self.x.grid(row=2, column=3, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="y:")
        ui.grid(row=2, column=4, sticky=tkinter.E)

        self.y = ttk.Label(result_frame, text="")
        self.y.grid(row=2, column=5, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="a:")
        ui.grid(row=3, column=0, sticky=tkinter.E)

        self.a = ttk.Label(result_frame, text="")
        self.a.grid(row=3, column=1, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="b:")
        ui.grid(row=3, column=2, sticky=tkinter.E)

        self.b = ttk.Label(result_frame, text="")
        self.b.grid(row=3, column=3, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="c:")
        ui.grid(row=3, column=4, sticky=tkinter.E)

        self.c = ttk.Label(result_frame, text="")
        self.c.grid(row=3, column=5, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="d:")
        ui.grid(row=3, column=6, sticky=tkinter.E)

        self.d = ttk.Label(result_frame, text="")
        self.d.grid(row=3, column=7, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="Initial detuning(Hz):")
        ui.grid(row=4, column=0, sticky=tkinter.E)

        self.det0 = ttk.Label(result_frame, text="")
        self.det0.grid(row=4, column=1, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="LFD(Hz/(MV/m)²):")
        ui.grid(row=4, column=0, sticky=tkinter.E)

        self.det0 = ttk.Label(result_frame, text="")
        self.det0.grid(row=4, column=1, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="Detuning slope(Hz/s):")
        ui.grid(row=4, column=0, sticky=tkinter.E)

        self.slope = ttk.Label(result_frame, text="")
        self.slope.grid(row=4, column=1, sticky=tkinter.W)

        footer = ttk.Frame(self.root, padding="3 3 12 12")
        footer.pack(side="right")

        ui = ttk.Button(footer, text='Close', command=self._quit)
        ui.pack(anchor="s", side="right", padx=(3, 10))

        ui = ttk.Button(footer, text='Calibrate', command=self.calibrate)
        ui.pack(anchor="s", side="right", padx=(3, 10))

        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind('<Escape>', lambda e: self._quit())
        self.root.bind_all('<Control-c>', lambda e: self._quit())

    def clear(self):
        self.ax_IQ.clear()
        self.ax_det.clear()
        self.ax_FR.clear()
        self.ax_bw.clear()

        self.ax_IQ.set_xlabel("Time (s)")
        self.ax_det.set_xlabel("Time (s)")
        self.ax_FR.set_xlabel("Time (s)")
        self.ax_bw.set_xlabel("Time (s)")

        self.ax_IQ.set_ylabel("Gradient (MV/m)")
        self.ax_det.set_ylabel("Detuning (Hz)")
        self.ax_FR.set_ylabel("Gradient (MV/m)")
        self.ax_bw.set_ylabel("Bandwidth (Hz)")

        self.ax_bw.set_ylim(0, 3000)
        self.ax_det.set_ylim(-1500, 1500)

        self.ax_IQ.set_title("I&Q")
        self.ax_det.set_title("Detuning")
        self.ax_FR.set_title("Amplitudes")
        self.ax_bw.set_title("Bandwidth")

        self.QL["text"] = ""
        self.hbw["text"] = ""
        self.x["text"] = ""
        self.y["text"] = ""
        self.a["text"] = ""
        self.b["text"] = ""
        self.c["text"] = ""
        self.d["text"] = ""

        self.f0["text"] = ""
        self.fs["text"] = ""
        self.det0["text"] = ""
        self.klfd["text"] = ""
        self.slope["text"] = ""

        self.fig_plots.tight_layout()
        self.canvas.draw()

    def calibrate(self):
        station = self.station.get()
        self.clear()
        station.calibrate()

        result = self.get_ui_data()

        f0 = result["f0"]
        fs = result["fs"]
        det0 = result["det0"]
        klfd = result["klfd"]
        slope = result["slope"]
        xy = result["xy"]
        QL = result["QL"]
        hbw = result["hbw"]
        abcd = result["abcd"]

        self.QL["text"] = "{:.5e}".format(QL)
        self.hbw["text"] = "{:.3e}".format(hbw)
        self.x["text"] = "{:.5f}".format(xy[0])
        self.y["text"] = "{:.5f}".format(xy[1])
        self.a["text"] = "{:.5f}".format(abcd[0])
        self.b["text"] = "{:.5f}".format(abcd[1])
        self.c["text"] = "{:.5f}".format(abcd[2])
        self.d["text"] = "{:.5f}".format(abcd[3])

        self.f0["text"] = "{:.3e}".format(f0)
        self.fs["text"] = "{:.3e}".format(fs)
        self.det0["text"] = "{:.3e}".format(hbw)
        self.klfd["text"] = "{:.3e}".format(klfd)
        self.slope["text"] = "{:.3e}".format(slope)

        self.ax_det.plot(result["time_trace"], result["detXY"], label="XY")
        self.ax_det.plot(result["time_trace"], result["detABCD"], label="ABCD")
        self.ax_det.plot(result["time_trace"], result["detEST"], label="Model")

        self.ax_bw.plot(result["time_trace"], result["hbwXY"], label="XY")
        self.ax_bw.plot(result["time_trace"], result["hbwABCD"], label="ABCD")
        self.ax_bw.plot(result["time_trace"], result["hbwDECAY"], "--", label="Decay")

        self.ax_IQ.plot(result["time_trace"], result["probe_I"], label=" Probe I")
        self.ax_IQ.plot(result["time_trace"], result["probe_Q"], label=" Probe Q")
        self.ax_IQ.plot(result["time_trace"], result["VprobeXY_I"], label="VProbe(XY) I")
        self.ax_IQ.plot(result["time_trace"], result["VprobeXY_Q"], label="VProbe(XY) Q")
        self.ax_IQ.plot(result["time_trace"], result["VprobeABCD_I"], label="VProbe(ABCD) I")
        self.ax_IQ.plot(result["time_trace"], result["VprobeABCD_Q"], label="VProbe(ABCD) Q")

        self.ax_FR.plot(result["time_trace"], result["probe_amp"], label="Probe")
        self.ax_FR.plot(result["time_trace"], result["vforwXY_amp"], label="Forward(XY)")
        self.ax_FR.plot(result["time_trace"], result["vreflXY_amp"], label="Reflected(XY)")
        self.ax_FR.plot(result["time_trace"], result["vforwABCD_amp"], label="Forward(ABCD)")
        self.ax_FR.plot(result["time_trace"], result["vreflABCD_amp"], label="Reflected(detABCD)")

        self.ax_det.legend(fontsize=8)
        self.ax_bw.legend(fontsize=8)
        self.ax_IQ.legend(fontsize=8)
        self.ax_FR.legend(fontsize=8)

        self.fig_plots.tight_layout()
        self.canvas.draw()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

def main(picker):
    root = tkinter.Tk()
    root.minsize(600, 600)
    root.title("RF calibration")
    app = App(root, station_picker)
    app.root.mainloop()





