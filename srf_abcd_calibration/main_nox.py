

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
        hbw = result["hbw"]
        abcd = result["abcd"]
        peak_amp = result["peak_amp"]

        print("F0(Hz): {:.3e}".format(f0))
        print("Fs(Hz): {:.3e}".format(fs))
        print("Peak amplitude(MV/m): {:.3e}".format(peak_amp))
        print("Half bandwidth(Hz): {:.3e}".format(hbw))
        print("LFD(Hz/(MV/m)²): {:.3e}".format(klfd))
        print("Detuning slope(Hz/s) {:.3e}".format(slope))
        print("QL: {:.5e}".format(QL))
        print("Half bandwidth(Hz): {:.3e}".format(hbw))
        print("x: {:.5f} y: {:.5f}".format(*xy))
        print("a: {:.5f} b: {:.5f} c: {:.5f} d: {:.5f}".format(*abcd))


