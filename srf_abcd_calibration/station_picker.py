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

from .station_desydoocs import DesyDoocsSCAVStation, DesyDoocsMCAVStation
from .station_chimeratk import ChimeraTKStation
from .station_dummy import DummyStation
from importlib_resources import files
import tomli

STATION_TYPES = {"DesyDoocsSCAV" : DesyDoocsSCAVStation,
                 "DesyDoocsMCAV" : DesyDoocsMCAVStation,
                 "ChimeraTK" : ChimeraTKStation,
                 "Dummy" : DummyStation}

# Class to pick one station
class StationPicker(object):
    def __init__(self, toml_path=None):
        if toml_path:
            with open(toml_path, "rb") as f:
                raw_conf = tomli.load(f)
        else:
            conf_text = files('srf_abcd_calibration').joinpath('config.toml').read_text()
            raw_conf = tomli.loads(conf_text)

        self.stations = dict()

        for name, conf in raw_conf.items():
            Station = STATION_TYPES[conf["type"]]

            if Station.loadable():
                station = Station(name, conf)
                self.stations[station.name] = station

    def pick_first(self):
        return list(self.stations)[0]

    def station_names(self):
        return list(self.stations)

    def pick_station(self, name):
        return self.stations[name]

