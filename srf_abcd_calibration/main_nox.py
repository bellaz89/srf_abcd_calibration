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

def main(station_picker, name, verbose=False):
    station = station_picker.pick_station(name)
    station.calibrate()

    if verbose:
        result = station.get_ui_data()

        f0 = result["f0"]
        fs = result["fs"]
        det0 = result["det0"]
        klfd = result["klfd"]
        slope = result["slope"]
        xy = result["xy"]
        QL = result["QL"]
        hbw = result["hbw_decay"]
        abcd = result["abcd"]
        abcd_tot = result["abcd_tot"]
        peak_amp = result["peak_amp"]

        print("F0(Hz): {:.3e}".format(f0))
        print("Fs(Hz): {:.3e}".format(fs))
        print("Peak amplitude(MV): {:.3e}".format(peak_amp))
        print("Half bandwidth(Hz): {:.3e}".format(hbw))
        print("LFD(Hz/MV²): {:.3e}".format(klfd))
        print("Detuning slope(Hz/s) {:.3e}".format(slope))
        print("QL: {:.5e}".format(QL))
        print("Half bandwidth(Hz): {:.3e}".format(hbw))
        print("x: {:.5f} y: {:.5f}".format(*xy))
        print("a: {:.5f} b: {:.5f} c: {:.5f} d: {:.5f}".format(*abcd))
        print("a_tot: {:.5f} b_tot: {:.5f} c_tot: {:.5f} d_tot: {:.5f}".format(*abcd_tot))


