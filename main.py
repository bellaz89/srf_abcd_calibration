import tkinter
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt
from calibrate_abcd import calculate_abcd

import numpy as np

class App(tkinter.Frame):
    F0 = 1.3e9
    DELAY_DECAY = 50
    FILLING_DELAY = 100
    PROBE_AMP_ADDR = "XFEL.RF/LLRF.CONTROLLER/{}.{}.{}/PROBE.AMPL"
    PROBE_PHA_ADDR = "XFEL.RF/LLRF.CONTROLLER/{}.{}.{}/PROBE.PHASE"
    FORWARD_AMP_ADDR = "XFEL.RF/LLRF.CONTROLLER/{}.{}.{}/VFORW.AMPL"
    FORWARD_PHA_ADDR = "XFEL.RF/LLRF.CONTROLLER/{}.{}.{}/VFORW.PHASE"
    REFLECTED_AMP_ADDR = "XFEL.RF/LLRF.CONTROLLER/{}.{}.{}/VREFL.AMPL"
    REFLECTED_PHA_ADDR = "XFEL.RF/LLRF.CONTROLLER/{}.{}.{}/VREFL.PHASE"
    DELAY_ADDR = "XFEL.RF/LLRF.CONTROLLER/CTRL.{}/PULSE_DELAY"
    FILLING_ADDR = "XFEL.RF/LLRF.CONTROLLER/CTRL.{}/PULSE_FILLING"
    FLATTOP_ADDR = "XFEL.RF/LLRF.CONTROLLER/CTRL.{}/PULSE_FLATTOP"
    THRESHOLD = 0.70
    FLATTEN_S = 50e-6
    
    STATIONS = ["A2.L1", "A3.L2", "A4.L2", "A5.L2"]
    STATIONS += ["A{}.L3".format(i) for i in range(6, 26)]
    
    MODULES = ["M1", "M2", "M3", "M4"]
    
    CAVITIES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

    def __init__(self, root):

        self.station = tkinter.StringVar(root)
        self.station.set(self.STATIONS[0])

        self.module = tkinter.StringVar(root)
        self.module.set(self.MODULES[0])

        self.cavity = tkinter.StringVar(root)
        self.cavity.set(self.CAVITIES[0])

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

        ui = ttk.Label(param_frame, text="Station:")
        ui.grid(column=0, row=1, sticky=tkinter.E)

        ui = ttk.OptionMenu(param_frame, self.station, *self.STATIONS, 
                            command=lambda x: self.clear())
        ui.grid(column=1, row=1, sticky=tkinter.W)

        ui = ttk.Label(param_frame, text="Module:")
        ui.grid(column=2, row=1, sticky=tkinter.E)

        ui = ttk.OptionMenu(param_frame, self.module, *self.MODULES, 
                            command=lambda x: self.clear())
        ui.grid(column=3, row=1, sticky=tkinter.W)

        ui = ttk.Label(param_frame, text="Cavity:")
        ui.grid(column=4, row=1, sticky=tkinter.E)

        ui = ttk.OptionMenu(param_frame, self.cavity, *self.CAVITIES, 
                            command=lambda x: self.clear())
        ui.grid(column=5, row=1, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="QL:")
        ui.grid(row=1, column=2, sticky=tkinter.E)

        self.QL = ttk.Label(result_frame, text="")
        self.QL.grid(row=1, column=3, sticky=tkinter.W)

        ui = ttk.Label(result_frame, text="bw:")
        ui.grid(row=1, column=4, sticky=tkinter.E)

        self.bw = ttk.Label(result_frame, text="")
        self.bw.grid(row=1, column=5, sticky=tkinter.W)

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

        self.ax_IQ.set_ylabel("(MV/m)")
        self.ax_det.set_ylabel("(Hz)")
        self.ax_FR.set_ylabel("(MV/m)")
        self.ax_bw.set_ylabel("(Hz)")

        self.ax_bw.set_ylim(0, 500)
        self.ax_det.set_ylim(-500, 500)

        self.ax_IQ.set_title("I&Q")
        self.ax_det.set_title("Detuning")
        self.ax_FR.set_title("Amplitudes")
        self.ax_bw.set_title("Bandwidth") 

        self.QL["text"] = ""
        self.bw["text"] = ""
        self.x["text"] = ""
        self.y["text"] = ""
        self.a["text"] = ""
        self.b["text"] = ""
        self.c["text"] = ""
        self.d["text"] = ""

        self.fig_plots.tight_layout()
        self.canvas.draw()

    def calibrate(self):
        station = self.station.get()
        module = self.module.get()
        cavity = self.cavity.get()

        self.clear()
        result = None
        ATTEMPTS = 5
        attempt = ATTEMPTS

        while attempt != 0:
            try:
                result = calculate_abcd(self.F0, 
                                        self.DELAY_DECAY, 
                                        self.FILLING_DELAY, 
                                        self.PROBE_AMP_ADDR.format(cavity, module, station), 
                                        self.PROBE_PHA_ADDR.format(cavity, module, station), 
                                        self.FORWARD_AMP_ADDR.format(cavity, module, station), 
                                        self.FORWARD_PHA_ADDR.format(cavity, module, station), 
                                        self.REFLECTED_AMP_ADDR.format(cavity, module, station), 
                                        self.REFLECTED_PHA_ADDR.format(cavity, module, station), 
                                        self.DELAY_ADDR.format(station), 
                                        self.FILLING_ADDR.format(station), 
                                        self.FLATTOP_ADDR.format(station), 
                                        self.THRESHOLD, 
                                        self.FLATTEN_S)
            except:
                attempt -= 1

        if not result:
            print("Failed to calibrate the signal after", ATTEMPTS, "attempts")
        else:

            QL = result["QL"]
            bw = result["bw"]
            xy = result["xy"]
            abcd = result["abcd"]

            self.QL["text"] = "{:.5e}".format(QL)
            self.bw["text"] = "{:.1f} Hz".format(bw)
            self.x["text"] = "{:.5f}".format(xy[0])
            self.y["text"] = "{:.5f}".format(xy[1])
            self.a["text"] = "{:.5f}".format(abcd[0])
            self.b["text"] = "{:.5f}".format(abcd[1])
            self.c["text"] = "{:.5f}".format(abcd[2])
            self.d["text"] = "{:.5f}".format(abcd[3])

            result["time_trace"] *= 1e-6

            self.ax_det.plot(result["time_trace"], result["detuning_xy"], label="Detuning(xy)")
            self.ax_det.plot(result["time_trace"], result["detuning_abcd"], label="Detuning(abcd)")
            self.ax_bw.plot(result["time_trace"], result["bandwidth_xy"], label="Bandwidth(xy)")
            self.ax_bw.plot(result["time_trace"], result["bandwidth_abcd"], label="Bandwidth(abcd)")
            self.ax_bw.plot(result["time_trace"], 
                            np.ones_like(result["time_trace"]) * bw, "--", label="Bandwidth(decay)")
            
            self.ax_IQ.plot(result["time_trace"], np.real(result["probe"]), label=" Probe I")
            #self.ax_IQ.plot(result["time_trace"], np.real(result["vforw_xy"] + result["vrefl_xy"]),
            #                                                           label="VProbe(xy) I")
            self.ax_IQ.plot(result["time_trace"], np.real(result["vforw_abcd"] + result["vrefl_abcd"]),
                                                                       label="VProbe(abcd) I")
            self.ax_IQ.plot(result["time_trace"], np.imag(result["probe"]), label=" Probe Q")
            #self.ax_IQ.plot(result["time_trace"], np.imag(result["vforw_xy"] + result["vrefl_xy"]),
            #                                                           label="VProbe(xy) Q")
            self.ax_IQ.plot(result["time_trace"], np.imag(result["vforw_abcd"] + result["vrefl_abcd"]),
                                                                       label="VProbe(abcd) Q")
            

            self.ax_FR.plot(result["time_trace"], np.abs(result["probe"]), label="Probe") 
            self.ax_FR.plot(result["time_trace"], np.abs(result["vforw_xy"]), label="Forward(xy)") 
            self.ax_FR.plot(result["time_trace"], np.abs(result["vrefl_xy"]), label="Reflected(xy)") 
            self.ax_FR.plot(result["time_trace"], np.abs(result["vforw_abcd"]), label="Forward(abcd)") 
            self.ax_FR.plot(result["time_trace"], np.abs(result["vrefl_abcd"]), label="Reflected(abcd)") 

            self.ax_det.legend(fontsize=8)
            self.ax_bw.legend(fontsize=8)
            self.ax_IQ.legend(fontsize=8)
            self.ax_FR.legend(fontsize=8)

        self.fig_plots.tight_layout()
        self.canvas.draw()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

root = tkinter.Tk()
root.minsize(600, 600)
root.title("RF calibration")
myapp = App(root)
myapp.root.mainloop()
