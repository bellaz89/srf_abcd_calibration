from station_desydoocs import DesyDoocsSCAVStation, DesyDoocsMCAVStation
from importlib_resources import files
import tomli

STATION_TYPES = {"DesyDoocsSCAV" : DesyDoocsSCAVStation,
                 "DesyDoocsMCAV" : DesyDoocsMCAVStation}

# Class to pick one station
class StationPicker(object):
    def __init__(self, toml_path=None, group=None, unique_items=None):
        self.group = group
        self.unique_items = unique_items

        if unique_items:
            self.unique_items = set(unique_items.split(","))

        if toml_path:
            with open(toml_path, "r") as f:
                raw_conf = tomli.load(f)
        else:
            conf_text = files('srf_abcd_calibration').joinpath('config.toml').read_text()
            raw_conf = tomli.loads(conf_text)

        self.group_params = dict()

        if "group" in raw_conf:
            self.group_params = raw_conf["group"]
            del raw_conf["group"]

        self.stations = dict()

        for name, conf in raw_conf.items():
            conf_ext = dict()

            for group in conf.get("groups", "").split(","):
                if group in self.group_params:
                    conf_ext.update(self.group_params[group])

            conf_ext.update(conf)
            Station = STATION_TYPES[conf["type"]]

            if Station.loadable():
                station = Station(name, conf_ext)
                stations[station.name] = station

        if self.group:
            for name in list(self.stations):
                if self.group not in self.stations[name].groups:
                    del self.stations[name]


        if self.unique_items:
            names_set = set(self.stations)
            names_set.intersection_update(self.unique_items)
            self.stations = {name: conf
                             for name, conf
                             in self.stations
                             if name in names_set}

    def pick_first(self):
        return list(self.stations.values)[0]

    def station_names(self):
        return list(self.stations)

    def pick_station(self, name):
        return self.stations[name]

