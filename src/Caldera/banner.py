"""
Caldera: Multimaterial heat transfer solver.
https://github.com/MarcelFerrari/Caldera

File: banner.py
Description: Banner to be printed at the start of the simulation.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import termcolor

banner = termcolor.colored(r"""
   (          (   (                     
   )\      )  )\  )\ )   (   (       )  
 (((_)  ( /( ((_)(()/(  ))\  )(   ( /(  
 )\___  )(_)) _   ((_))/((_)(()\  )(_)) 
((/ __|((_)_ | |  _| |(_))   ((_)((_)_  
 | (__ / _` || |/ _` |/ -_) | '_|/ _` | 
  \___|\__,_||_|\__,_|\___| |_|  \__,_|   (v0.1)
""", 'red', attrs=['bold'])
                               
def print_banner():
    print(banner)
    