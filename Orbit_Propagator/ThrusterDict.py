# -*- coding: utf-8 -*-
"""
Thruster Dictionary
Create a dictionary with thruster parameters
"""
from dataclasses import dataclass
 
@dataclass
class thruster():
    Name : str                    # Name of thruster
    Type : str                    # Thruster type (i.e. PPT, hall-effect, etc.)
    Isp : float                   # Specific impulse [s]
    MaxThrust : float             # Maximum thrust [N]
    FuelMass : float              # Total mass of fuel [g]
    TotalMass : float             # Total mass of unit [g]
    ThrustPowerRatio : float      # Thrust to power ratio [N/W]
    
#Thrusters
Hypernova_NanoThrusterA = thruster( 
    Name = 'Hypernova_NanoThrusterA',
    Type = 'VAT',
    Isp = 500.0,
    MaxThrust = 50.0e-6,
    FuelMass = 3.0,
    TotalMass = 600.0,
    ThrustPowerRatio = 5.0e-6
    )

Enpropulsion_MicroR3_OneUnit = thruster( 
    Name = 'Enpropulsion_MicroR3',
    Type = 'FEEP',
    Isp = 1500.0,
    MaxThrust = 1.35e-3,
    FuelMass = 1300,
    TotalMass = 2600,
    ThrustPowerRatio = 11.25e-6
    )

Enpropulsion_NanoR3 = thruster( 
    Name = 'Enpropulsion_NanoR3',
    Type = 'FEEP',
    Isp = 2000.0,
    MaxThrust = 350e-6,
    FuelMass = 220,
    TotalMass = 1200,
    ThrustPowerRatio = 8.75e-6
    )

DawnCube_Bipropellant_CSPM_1U = thruster( 
    Name = 'DawnCube_Bipropellant_CSPM',
    Type = 'bipropellant',
    Isp = 285.0,
    MaxThrust = 1.0,
    FuelMass = 310,
    TotalMass = 1100,
    ThrustPowerRatio = 12.5
    )

DummyThruster = thruster( 
    Name = 'dummy',
    Type = 'dummy',
    Isp = 1000.0,
    MaxThrust = 100.0e-3,
    FuelMass = 100.0,
    TotalMass = 1000.0,
    ThrustPowerRatio = 10.0e-6
    )

