#!/usr/bin/env python
import sys
sys.path.append('/home/adam/Documents/pulsarsurveyscraper')
import os
import argparse
import json
import re
from astropy.coordinates import SkyCoord
from astropy import units as u
import pulsarsurveyscraper
import csv
import numpy as np
import chime_fn
import logging
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import scipy
def main():
    pulsarsurveyscraper.log.setLevel("WARNING")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fn", default='my_new_sources.csv', help="Clustering csv file")
    parser.add_argument("--radec_only",action="store_true", help="only match ra and dec, not dm")
    parser.add_argument("--dmtol", default=None, type=float, help="DM tolerance")
    parser.add_argument(
        "-r", "--radius", default=10, type=float, help="Search radius (degrees)"
    )
    pulsar_table = pulsarsurveyscraper.PulsarTable()

    args = parser.parse_args()
    radec_only = args.radec_only
    radius = args.radius
    chime_ft = chime_fn.load_clusters(args.fn)
    dm_tol = args.dmtol
    unique_survey = []
    matched_survey = []
    KSS_info = []
    print('cluster_num,pulsar_name,seperation,dm_range,num_events,mode_date,all_dates')
    for source in chime_ft:
        ra=float(source['ara'])
        dec=float(source['adec'])
        dm=float(source['adm'])
        coord = SkyCoord(ra, dec, unit="deg")
        #just need to set DM to None to not have a DM cut
        if radec_only:
            #to do: asymetric radius
            result = pulsar_table.search(
                    coord,
                    ra_tol = source['era'],
                    dec_tol = source['edec'],
                    DM=None,
                    DM_tolerance=source['edm'],
                    return_json=True,
                    deduplicate=False,
            )
        else:
            result = pulsar_table.search(
                    coord,
                    ra_tol=False,
                    radius=radius*u.deg,
                    DM=source['adm'],
                    DM_tolerance=dm_tol,
                    return_json=True,
                    deduplicate=False,
            )
        if result["nmatches"]==0:
            unique_survey.append(source)
            print(f"{source['cc']}|{source['ara']}|{source['adec']}|{source['adm']}")

        if result["nmatches"]>0:
#            result['cluster']=[source,float(ra),float(dec),dm]
            # need to output the search results too
            plt.scatter(result['searchra']['value'],result['searchdec']['value'],label=str(str(source['cc'])+'DM: '+str(source['adm'])))

            for i,key in enumerate(result.keys()):
                if radec_only:
                    im = 4
                else:
                    im = 6
                if i>im:
                    plt.scatter(result[key]['ra']['value'],result[key]['dec']['value'],label=result[key]['survey']['value']+' '+
                                key+' DM: '+str(result[key]['dm']['value'])+' Period: '+str(result[key]['period']['value']))
                    RA = result[key]['ra']['value']
                    Dec = result[key]['dec']['value']
                    DM = result[key]['dm']['value']

                    # print(f'Source: {key}, ra: {RA}, dec: {Dec}, DM: {DM}')
                    # Values to print out for KSS
                    d = result[key]['distance']['value']
                    c = source['cc']
                    edm = source['edm']
                    num_events = len(source['times'])
                    dates_str = list(t.strftime('%Y:%m:%d') for t in source['times'])
                    dates = list(t.date() for t in source['times'])
                    mcd = scipy.stats.mode(dates)
                    mcd_str = mcd.mode[0].strftime('%Y:%m:%d')
                    # KSS_info.append({'cluster':c,'sep':d,'dm_range':edm,'num_events':num_events,'most_common_date':mcd,'all_dt':source['times']})
                    print(f'{c},{key},{d},{edm},{num_events},{mcd_str},{dates_str}')
            ax = plt.gca()
            ax.add_patch(
                Ellipse(
                    (result['searchra']['value'],result['searchdec']['value']),
                    width=2*source['era'],
                    height=2*source['edec'],
                    facecolor="none",
                    edgecolor="r",
                    alpha=1,
                )
            )

            plt.legend()
            plt.xlabel('ra')
            plt.ylabel('dec')
            if radec_only:
                app = 'radec'
            else:
                app = 'radecdm'
            plt.savefig(str(str(source['cc'])+app+'.png'))
            plt.close()
            # matched_survey.append(result)

    np.save('KSS_info',KSS_info)
    np.save('new_sources',unique_survey)
    np.save('matched_surveys',matched_survey)

if __name__ == "__main__":
    main()
