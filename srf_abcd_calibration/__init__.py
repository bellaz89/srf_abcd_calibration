"""srf_abcd_calibration

Usage:
    srf_abcd_calibration [--group=<gp>] [--stations=<sts>] [--conf=<cf>] [--nox=<st>] [--verbose] [--list-available] [--list-types] [--bootstrap-conf]
    srf_abcd_calibration (-h | --help)

Options:
    -h --help         Show this screen
    --group=<gp>      Select all stations that belong to a particular group
    --stations=<sts>  Select all stations using a comma separated list of names
    --conf=<cf>       Use an alternate .toml configuration
    --nox=<st>        Run the calibration script on a station in text mode
    --verbose         Print the calibration results (text mode only)
    --list-available  Print all available station names and return
    --list-types      Print all types of stations that can be created and return
    --bootstrap-conf  Generates a file named config_user.toml with example configuration and return

"""
from docopt import docopt
from importlib_resources import files
from station_picker import StationPicker
from station import STATION_TYPES
import main_ui
import main_nox


def main():
    arguments = docopt(__doc__)
    print(arguments)
    station_picker = StationPicker(arguments["--conf"],
                                   arguments["--group"],
                                   arguments["--stations"])

    if arguments["--list-available"]:
        print("Available stations:")
        for name in station_picker.station_names():
            print("  " + name)

    elif arguments["--list-types"]:
        print("Available station types:")
        for name, Station in STATION_TYPES.items():
            if Station.loadable():
                print("  " + name + "  " + "(Loadable)")
            else:
                print("  " + name + "  " + "(Not loadable)")

    elif arguments["--bootstrap-conf"]:
        conf_text = files('srf_abcd_calibration').joinpath('config_template.toml').read_text()
        with open("config_user.toml", w) as f:
            f.write(conf_text)

    elif arguments["--nox"]:
        main_nox.main(station_picker,
                      arguments["--nox"],
                      arguments["--verbose"])
    else:
        main_ui.main(station_picker)


if __name__ == "__main__":
    main()
