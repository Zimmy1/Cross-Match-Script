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
import requests
import tempfile
import json
import os

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

# Choose epoch and find fluxes
ra_chosen = sorted_ra[0]
dec_chosen = sorted_dec[0]

c1 = SkyCoord(ra_chosen, dec_chosen, unit="deg")
flux_final = np.full((len(sorted_sbid), len(c1)), 1.25, dtype=float)
flux_err_final = np.full((len(sorted_sbid), len(c1)), 0.25, dtype=float)
source_count = []

# Build light curves
# Switched to mask because the for loops were painfully slow
for i in range(len(sorted_sbid)):
    c2 = SkyCoord(sorted_ra[i], sorted_dec[i], unit="deg")
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    mask = d2d <= max_sep
    flux_final[i, mask] = sorted_flux[i][idx[mask]]
    flux_err_final[i, mask] = sorted_flux_err[i][idx[mask]]
    
    idx2, d2d2, _ = c2.match_to_catalog_sky(c1)
    new_mask = d2d2 > max_sep
    new_indices = np.where(new_mask)[0]
    n_indices = new_indices.size
    
    # Build lightcurves for sources not found in previous epochs
    if n_indices > 0:
        new_coords = c2[new_indices]
        print(f'{n_indices} new sources found in epoch {i+1}')
        
        new_ra  = np.append(c1.ra.deg,  new_coords.ra.deg)
        new_dec = np.append(c1.dec.deg, new_coords.dec.deg)
        c1 = SkyCoord(ra=new_ra * u.deg, dec=new_dec * u.deg, frame="icrs")
        
        new_flux = np.full((len(sorted_sbid), n_indices), 1.25, dtype=float)
        new_flux_err = np.full((len(sorted_sbid), n_indices), 0.25, dtype=float)
        new_flux[i, :] = sorted_flux[i][new_indices]
        new_flux_err[i, :] = sorted_flux_err[i][new_indices]
        
        flux_final = np.hstack([flux_final, new_flux])
        flux_err_final = np.hstack([flux_err_final, new_flux_err])

n_epoch, n_obj = flux_final.shape

# Impliment slack API
SLACK_TOKEN = ""
CHANNEL = ""

def slack_msg(text):
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SLACK_TOKEN}" }
    payload = {"channel": CHANNEL, "text": text}
    r = requests.post(url, headers=headers, json=payload)
    print(r.json())
    
def slack_png(fig, title="Light Curve"):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
    file_path = tmp.name
    file_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)

    headers_auth = {"Authorization": f"Bearer {SLACK_TOKEN}"}
    url1 = "https://slack.com/api/files.getUploadURLExternal"
    data1 = {"filename": filename,"length": file_size,}
    r1 = requests.post(url1, headers=headers_auth, data=data1)
    resp1 = r1.json()
    print("getUploadURLExternal:", resp1)
    if not resp1.get("ok"):
        print("Failed to get upload URL")
        return

    upload_url = resp1["upload_url"]
    file_id = resp1["file_id"]

    with open(file_path, "rb") as f:
        r2 = requests.post(upload_url, data=f)
    print("binary upload status:", r2.status_code, r2.text)

    url3 = "https://slack.com/api/files.completeUploadExternal"
    payload3 = {"channel_id": CHANNEL,"initial_comment": title,"files": [{"id": file_id, "title": title}],}
    headers_json = {"Authorization": f"Bearer {SLACK_TOKEN}","Content-Type": "application/json;charset=utf-8",}
    r3 = requests.post(url3, headers=headers_json, data=json.dumps(payload3))
    print("completeUploadExternal:", r3.json())


for tidx in range(n_obj):
    
    # Plot light curves
    y = flux_final[:, tidx]
    err = flux_err_final[:, tidx]
    uplim = (y == 1.25) & (err == 0.25)
    detect = ~uplim
    
    
    fig, ax = plt.subplots()
    ax.errorbar(sorted_sbid[detect], y[detect], yerr=err[detect],
                fmt='o-', label='detections')
    ax.errorbar(sorted_sbid[uplim], y[uplim], yerr=err[uplim],
                uplims=True, fmt='o', label='uplims')
    ax.set_title(f'Light Curve of Target at {c1[tidx]}')
    ax.set_xlabel('SBID')
    ax.set_ylabel('Flux (mJy)')
    ax.legend()
    plt.show()
    print("Uploading image for tidx =", tidx)
    slack_png(fig, title=f'Light Curve of Target at {c1[tidx]}')
    plt.close(fig)
    
    # Check for interest 
    i = 1
    for _ in range(len(sorted_sbid)-1):
        df = y[i] - y[i-1]
        de = np.sqrt(err[i]**2 + err[i-1]**2)
        sig = np.abs(df) / de
        
        if sig > 2 and df > 0:
            message = f'\nObject at {c1[tidx]} flux rose by {df:.3f} mJy from observation {sorted_sbid[i-1]} to {sorted_sbid[i]}'
            slack_msg(message)
        elif sig > 2 and df < 0:
            message = f'\nObject at {c1[tidx]} flux decreased by {df:.3f} mJy from observation {sorted_sbid[i-1]} to {sorted_sbid[i]}'
            slack_msg(message)       
        i += 1


