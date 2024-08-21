Introduction
============

This program facilitates the forward and reflected RF channels calibration of superconducting accelerating systems. See [the related publication][1].

The program works with different particle accelerator control systems ([Doocs][2], [Epics][3], [OPC-UA][4]) interfacing with the [DESY's LLRF server][5]. However, it is also possible to add new control or LLRF systems.

This program refers to an unique combination of an SRF cavity and a LLRF system to be calibrated as a *station*.

Installation
============

```bash
  pip install git+https://gitlab.desy.de/msk-llrf/scripts/python/common/calibration-abcd-decay-based
```

Usage
=====

The package define a program that can be called from the shell. Below the program options
```
Usage:
    srf_abcd_calibration [--group=<gp>] [--stations=<sts>] [--conf=<cf>] [--nox=<st>] [--verbose] [--list-available] [--list-types] [--bootstrap-conf] [--dry-run] [--info-station=<st>]
    srf_abcd_calibration (-h | --help)

Options:
    -h --help           Show this screen
    --conf=<cf>         Use an alternate .toml configuration
    --nox=<st>          Run the calibration script on a station in text mode
    --verbose           Print the calibration results (text mode only)
    --list-available    Print all available station names and return
    --list-types        Print all types of stations that can be created and return
    --bootstrap-conf    Generates a file named config_user.toml with example configuration and return
```

When no options are passed, the program is executed in graphical mode loading all the systems defined in the default [.toml][6] configuration.

Program usage example
---------------------


### Use an user defined configuration file

```bash
  srf_abcd_calibration --conf=myconf.toml
```

### List available stations

```bash
  srf_abcd_calibration --list-available
```

### List available system types

```bash
  srf_abcd_calibration --list-types
```

This switch is useful to see if a particular type of LLRF system can be used. This usually (and should) depends on the available packages installed on the system.

- `DesyDoocsSCAV` and `DesyDoocsMCAV` are available when the package `pydoocs` is installed
- `DesyChimeraTKSCAV` is available when the package [`deviceaccess`][7] is installed.

### Run the program in non-graphical mode and print the results

```bash
  srf_abcd_calibration --nox=dummy_xfel --verbose
```

generates

```

```

Customize stations
==================

The program use the [TOML][6] configuration file format. A station is defined by defining a new table in the configuration file. The fragment

```toml

# ...

[MyLLRFStation]
type="Dummy"

# ...

```

defines a new station named `MyLLRFStation` with type `Dummy.

The common station properties are listed below

| Configuration parameter | Meaning                                     | Default |
| ----------------------- | ------------------------------------------- | ------- |
| name                    | Optional name that overrides the table name | None    |
| type                    | LLRF system type. See below                 | None    |


Station types
-------------

### DesyDoocsSCAV

| Configuration parameter | Meaning                                        | Default |
| ----------------------- | ---------------------------------------------- | ------- |
| address                 | SCAV LLRF control system (CTRL.) DOOCS address | None    |

### DesyDoocsMCAV


| Configuration parameter | Meaning                              | Default |
| ----------------------- | ------------------------------------ | ------- |
| address                 | MCAV LLRF cavity (.Cx) DOOCS address | None    |


### DesyChimeraTKSCAV

| Configuration parameter | Meaning                                    | Default |
| ----------------------- | ------------------------------------------ | ------- |
| dmap_path               | SCAV LLRF control system DMAP file path    | None    |
| probe_ch                | Probe RF channel                           | 1       |
| vforw_ch                | Forward RF channel                         | 0       |
| vrefl_ch                | Reflected RF channel                       | 2       |

### Dummy

A dummy station. Always available. Instead of interacting with a real control system, it calibrates a simulated LLRF system whose parameters are defined by the user.
Useful for debugging and presentation.

| Configuration parameter | Meaning                           | Default     |
| ----------------------  | --------------------------------- | ----------- |
| f0                      | Cavity frequency(Hz)              | 1.3e9       |
| fs                      | Sampling frequency(Hz)            | 10e6        |
| flattop_start           | Flattop start(s)                  | 1e-3        |
| decay_start             | Decay start(s)                    | 2e-3        |
| decay_stop              | Decay stop(s)                     | 3e-3        |
| vforw_flattop           | Forward max amplitude(MV)         | 7.5         |
| hbw                     | Half bandwidth(Hz)                | 65.0        |
| det0                    | Initial detuning                  | 225.0       |
| klfd                    | Lorentz force detuning (Hz/MV^2)  | -1.0        |
| slope                   | Detuning slope (Hz/s)             | 0.0         |
| amplifier_noise         | Amplifier noise STD (MV)          | 0.1e6       |
| adc_noise               | ADC noise STD (MV)                | 0.01        |
| probe_scaling           | Initial probe mismatch            | 0.83        |
| cross_coupling_a        | Cross coupling mismatch A         | "1.12+0.1j" |
| cross_coupling_b        | Cross coupling mismatch B         | "-0.1-0.3j" |
| cross_coupling_c        | Cross coupling mismatch C         | "0.0+0.2j"  |
| cross_coupling_d        | Cross coupling mismatch D         | "0.3+1.0j"  |

Example configuration
---------------------

**TODO**

Write a new station type
------------------------

**TODO**

Contacts
========

Andrea Bellandi (andrea.bellandi@desy.de)

[1]: https://www.sciencedirect.com/science/article/pii/S0168900224000986
[2]: https://doocs.desy.de/
[3]: https://epics-controls.org/
[4]: https://opcfoundation.org/
[5]: https://msk.desy.de/
[6]: https://toml.io/en/
[7]: https://github.com/ChimeraTK/DeviceAccess-PythonBindings
