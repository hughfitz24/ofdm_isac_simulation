# OFDM ISAC Simulation

This repository implements an ISAC system using OFDM symbols for both data communication and target sensing. 

The simulation generates OFDM symbols at a transmitter, and models these symbols transmitting over a wireless channel. Two seperate translations happen to the symbols:
- Part of the signal power is reflected off of objects in the wireless field. These reflections are processed at the transmitter to estimate the range and velocity of the objects. 
- The rest of the signal power goes to a downlink receiver that is displaced in the wireless field. Standard OFDM processing using an MMSE receiver is done at this receiver. 

This simulation seeks to show how the same OFDM symbols can be used to do both radar detection and data communication. It also seeks to show how these two things compete with one another, and serves as a basis for how an optimisation problem can be formulated for joint performance in these two competing tasks. 

---

# Getting Started

This project is set up using `uv`. To run the simulator:
```bash
uv run ofdm-radar-sim
```

# Configuration of Simulator
The simulator is configured using a TOML config file. Comments have been placed in this TOML file to help advise what the individual values mean. 

# High Level Model
This simulation performs some slight simplifications and assumptions worth clarifying. 

The wireless channel that the radar / data signals propagate through are modelled seperately. A goal for this simulator would be to have this be a cohesive simulation, such that the downlink channel paths actually come from the radar targets in the wireless field, however this is an extension of the simulation rather than a core feature. For now, we assume that they can be modelled seperately from one another despite being inherently related. 

