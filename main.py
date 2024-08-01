import tkinter
from tkinter import ttk

import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt
from calibrate_abcd import calculate_abcd, C2AP, AP2C, get_flattop_range_idx
import pydoocs as pd
import numpy as np
import os

class App(tkinter.Frame):
    F0 = 1.3e9
    DELAY_DECAY = 50
    FILLING_DELAY = 100
    PROBE_AMP_ADDR = "CMTB.RF/LLRF.CONTROLLER/PROBE.SCAV.CMTB/AMPL"
    PROBE_PHA_ADDR = "CMTB.RF/LLRF.CONTROLLER/PROBE.SCAV.CMTB/PHASE"
    VFORW_AMP_ADDR = "CMTB.RF/LLRF.CONTROLLER/FORWARD.SCAV.CMTB/AMPL"
    VFORW_PHA_ADDR = "CMTB.RF/LLRF.CONTROLLER/FORWARD.SCAV.CMTB/PHASE"
    VREFL_AMP_ADDR = "CMTB.RF/LLRF.CONTROLLER/REFLECTED.SCAV.CMTB/AMPL"
    VREFL_PHA_ADDR = "CMTB.RF/LLRF.CONTROLLER/REFLECTED.SCAV.CMTB/PHASE"

    PROBE_AMP_CAL_ADDR = "CMTB.RF/LLRF.CONTROLLER/PROBE.SCAV.CMTB/CAL_SCA"
    PROBE_PHA_CAL_ADDR = "CMTB.RF/LLRF.CONTROLLER/PROBE.SCAV.CMTB/CAL_ROT"
    VFORW_AMP_CAL_ADDR = "CMTB.RF/LLRF.CONTROLLER/FORWARD.SCAV.CMTB/CAL_SCA"
    VFORW_PHA_CAL_ADDR = "CMTB.RF/LLRF.CONTROLLER/FORWARD.SCAV.CMTB/CAL_ROT"
    VREFL_AMP_CAL_ADDR = "CMTB.RF/LLRF.CONTROLLER/REFLECTED.SCAV.CMTB/CAL_SCA"
    VREFL_PHA_CAL_ADDR = "CMTB.RF/LLRF.CONTROLLER/REFLECTED.SCAV.CMTB/CAL_ROT"

    DELAY_ADDR = "CMTB.RF/LLRF.CONTROLLER/CTRL.SCAV.CMTB/PULSE_DELAY"
    FILLING_ADDR = "CMTB.RF/LLRF.CONTROLLER/CTRL.SCAV.CMTB/PULSE_FILLING"
    FLATTOP_ADDR = "CMTB.RF/LLRF.CONTROLLER/CTRL.SCAV.CMTB/PULSE_FLATTOP"

    AI_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/A.I"
    AQ_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/A.Q"
    BI_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/B.I"
    BQ_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/B.Q"
    CI_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/C.I"
    CQ_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/C.Q"
    DI_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/D.I"
    DQ_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/D.Q"
    HALF_BANDWIDTH_ADDR = "CMTB.RF/LLRF.CAVITY_ESTIMATOR/SCAV.ESTIMATOR/HALF_BANDWIDTH"
    QL_ADDR = "CMTB.RF/LLRF.CONTROLLER/CONFIG.SCAV.CMTB/QL"


    THRESHOLD = 0.70
    FLATTEN_S = 50e-6


    VPM_ADDR = "TTF.RF/GPIB/C{}.MTS.PROBE/CH{}.VALUE.VPM"
    PM_N = { "C1" : ("12", "0"), 
             "C2" : ("12", "1"), 
             "C3" : ("34", "0"), 
             "C4" : ("34", "1"), 
             "C5" : ("56", "0"), 
             "C6" : ("56", "1"), 
             "C7" : ("78", "0"),
             "C8" : ("78", "1")}
    
    CAVITIES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

    def __init__(self, root):

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

        ui = ttk.Label(param_frame, text="Cavity:")
        ui.grid(column=4, row=1, sticky=tkinter.E)

        ui = ttk.OptionMenu(param_frame, self.cavity, None, *self.CAVITIES, 
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

        self.ax_bw.set_ylim(0, 3000)
        self.ax_det.set_ylim(-1500, 1500)

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
        cavity = self.cavity.get()

        self.clear()
        result = None
        ATTEMPTS = 5
        attempt = ATTEMPTS

        mvpm = pd.read(self.VPM_ADDR.format(*self.PM_N[cavity]))["data"]*1e-6
        probe_amp = pd.read(self.PROBE_AMP_ADDR)["data"]
        time_trace = probe_amp[:, 0]
        probe_amp = probe_amp[:, 1]
        probe_pha = pd.read(self.PROBE_PHA_ADDR)["data"][:, 1]

        (start, stop) = get_flattop_range_idx(time_trace, 0, 0, 
                                              self.DELAY_ADDR, 
                                              self.FILLING_ADDR,
                                              self.FLATTOP_ADDR)


        corr_amp = mvpm / np.max(probe_amp)   
        corr_pha = np.mean(probe_pha[start:stop])

        corr = AP2C(corr_amp, -corr_pha)

        cal_amp = pd.read(self.PROBE_AMP_CAL_ADDR)["data"]
        cal_pha = pd.read(self.PROBE_PHA_CAL_ADDR)["data"]

        print(start, stop)
        print(corr_pha, cal_pha)

        (cal_amp, cal_pha) = C2AP(corr * AP2C(cal_amp, cal_pha))

        pd.write(self.PROBE_AMP_CAL_ADDR, cal_amp)
        pd.write(self.PROBE_PHA_CAL_ADDR, cal_pha)

        print(cal_pha)

        result = calculate_abcd(self.F0, 
                                self.DELAY_DECAY, 
                                self.FILLING_DELAY, 
                                self.PROBE_AMP_ADDR, 
                                self.PROBE_PHA_ADDR, 
                                self.VFORW_AMP_ADDR, 
                                self.VFORW_PHA_ADDR, 
                                self.VREFL_AMP_ADDR, 
                                self.VREFL_PHA_ADDR, 
                                self.DELAY_ADDR, 
                                self.FILLING_ADDR, 
                                self.FLATTOP_ADDR, 
                                self.THRESHOLD, 
                                self.FLATTEN_S)

        if not result:
            print("Failed to calibrate the signal after", ATTEMPTS, "attempts")
        else:

            result["cavity"] = cavity
            result["timestamp"] = time.time()

            np.savez(str(result["timestamp"]) + "." + 
                     cavity + ".npz",
                     **result)


            xy = result["xy"]

            cal_amp = pd.read(self.VFORW_AMP_CAL_ADDR)["data"]
            cal_pha = pd.read(self.VFORW_PHA_CAL_ADDR)["data"]

            (cal_amp, cal_pha) = C2AP(xy[0] * AP2C(cal_amp, cal_pha))

            pd.write(self.VFORW_AMP_CAL_ADDR, cal_amp)
            pd.write(self.VFORW_PHA_CAL_ADDR, cal_pha)

            cal_amp = pd.read(self.VREFL_AMP_CAL_ADDR)["data"]
            cal_pha = pd.read(self.VREFL_PHA_CAL_ADDR)["data"]

            (cal_amp, cal_pha) = C2AP(xy[1] * AP2C(cal_amp, cal_pha))

            pd.write(self.VREFL_AMP_CAL_ADDR, cal_amp)
            pd.write(self.VREFL_PHA_CAL_ADDR, cal_pha)

            QL = result["QL"]

            pd.write(self.QL_ADDR, QL)

            bw = result["bw"]
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
            self.ax_IQ.plot(result["time_trace"], np.real(result["vforw_xy"] + result["vrefl_xy"]),
                                                                       label="VProbe(xy) I")
            self.ax_IQ.plot(result["time_trace"], np.real(result["vforw_abcd"] + result["vrefl_abcd"]),
                                                                       label="VProbe(abcd) I")
            self.ax_IQ.plot(result["time_trace"], np.imag(result["probe"]), label=" Probe Q")
            self.ax_IQ.plot(result["time_trace"], np.imag(result["vforw_xy"] + result["vrefl_xy"]),
                                                                       label="VProbe(xy) Q")
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

            #ssh_command = ("ssh cmtbcpullscav /usr/bin/python3 " + 
            #              "/home/bellandi/Public/cmtb-scav-set-qldet/main.py " + 
            #              "{} {} {} {} {}".format(abcd[0], abcd[1], abcd[2], abcd[3],
            #              0.5*np.pi*bw))

            pd.write(self.AI_ADDR, np.real(abcd[0]))
            pd.write(self.AQ_ADDR, np.imag(abcd[0]))
            pd.write(self.BI_ADDR, np.real(abcd[1]))
            pd.write(self.BQ_ADDR, np.imag(abcd[1]))
            pd.write(self.CI_ADDR, np.real(abcd[2]))
            pd.write(self.CQ_ADDR, np.imag(abcd[2]))
            pd.write(self.DI_ADDR, np.real(abcd[3]))
            pd.write(self.DQ_ADDR, np.imag(abcd[3]))
            pd.write(self.HALF_BANDWIDTH_ADDR, 0.5*bw) 

            #print(ssh_command)
            #os.system(ssh_command)

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
