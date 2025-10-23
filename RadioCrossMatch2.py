#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:40:55 2025

@author: jacksonzimmerman
"""

import matplotlib.pyplot as plt
import astropy.io.votable as v
import glob
import astropy.units as u
from astropy.coordinates import SkyCoord 
import re
import numpy as np

folder = "/Users/jacksonzimmerman/Downloads/lightcurves"
files = glob.glob(folder + "/*.xml")
flux_ary = []
sbid_ary = []
ra_ary = []
dec_ary = []
flux_err_ary = []
max_sep = 2.5 * u.arcsec

for file in files:
    try:
        votable = v.parse(file)
        table = votable.get_first_table().to_table()
        
        # Storing the SBID's for each observation because I couldn't figure out
        # how to get the dates and after some research sbids seemed like
        # the next best thing
        sbid_str = re.search(r"SB(\d+)", file)
        sbid = int(sbid_str.group(1))
        sbid_ary.append(sbid)
        
        ra  = np.asarray(table["col_ra_deg_cont"])
        dec = np.asarray(table["col_dec_deg_cont"])
        flux = np.asarray(table["col_flux_peak"])
        flux_err = np.asarray(table["col_flux_peak_err"])
        
        ra_ary.append(ra)
        dec_ary.append(dec)
        flux_ary.append(flux)
        flux_err_ary.append(flux_err)
            
    # One particular file was giving me the error 
    # "ValueError: 1:0: not well-formed (invalid token)" so I added this exception.
    except ValueError:
        print(f"Value Error Occured With File: {file}")
        continue
   
sbid_ary = np.asarray(sbid_ary, dtype=int) 
order = np.argsort(sbid_ary)

sorted_sbid = sbid_ary[order]
sorted_ra = [ra_ary[i] for i in order]
sorted_dec = [dec_ary[i] for i in order]
sorted_flux = [flux_ary[i] for i in order]
sorted_flux_err = [flux_err_ary[i] for i in order]

# Choose epoch and find fluxes (switch to user input later)
ra_chosen = sorted_ra[5]
dec_chosen = sorted_dec[5]

c1 = SkyCoord(ra_chosen, dec_chosen, unit="deg")
print(c1)
flux_final = np.full((len(sorted_sbid), len(c1)), np.nan, dtype=float)
flux_err_final = np.full((len(sorted_sbid), len(c1)), 0, dtype=float)

for i in range(len(sorted_sbid)):
    c2 = SkyCoord(sorted_ra[i], sorted_dec[i], unit="deg")
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    
    for j in range(len(c1)):
        if d2d[j] <= max_sep:
            flux_final[i][j] = sorted_flux[i][idx[j]]
            flux_err_final[i][j] = sorted_flux_err[i][idx[j]]
        else:
            flux_final[i][j] = 1.25
            flux_err_final[i][j] = 0.25

tidx = 6
y = flux_final[:, tidx]
err = flux_err_final[:, tidx]
uplim = (y == 1.25) & (err == 0.25)
detect = ~uplim

plt.errorbar(sorted_sbid[detect], y[detect], yerr=err[detect],fmt='o-', color='blue', label = 'detections')
plt.errorbar(sorted_sbid[uplim], y[uplim], yerr=err[uplim], uplims=True,fmt='o', color='red', label ='uplims')
plt.title(f'Light Curve of object at epoch 2 and target {tidx}')
plt.xlabel('SBID')
plt.ylabel('Flux (mJy)')
plt.legend()
plt.show()

# Check for interest 
i = 1
for _ in range(len(sorted_sbid)-1):
    df = y[i] - y[i-1]
    de = np.sqrt(err[i]**2 + err[i-1]**2)
    sig = np.abs(df) / de
    
    if sig > 2 and df > 0:
        print(f'Objects flux rose by {df} mJy from observation {sorted_sbid[i-1]} to {sorted_sbid[i]}')
    elif sig > 2 and df < 0:
        print(f'Objects flux decreased by {df} mJy from observation {sorted_sbid[i-1]} to {sorted_sbid[i]}')
    
    i += 1
    
