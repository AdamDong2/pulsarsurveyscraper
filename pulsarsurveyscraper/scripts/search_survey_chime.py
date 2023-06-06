#!/usr/bin/env python
import sys
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

def plot_matches(result,source,ra,dec,dm):
    if result["nmatches"]>0:
        result['cluster']=[source,float(ra),float(dec),dm]
        import matplotlib.pyplot as plt
        plt.scatter(result['searchra']['value'],result['searchdec']['value'],label='search')
        plt.savefig(source+'.png')

def write_source_location(filename,ra,dec,dm):
    with open(filename,'a') as f:
        f.write("|ra|dec|dm|\n")
        f.write(f"|{ra}|{dec}|{dm}|\n")
def write_image_org(filename,imagename):
    with open(filename,'a') as f:
        f.write(f"[[file:{imagename}]]\n")
def write_title_org(filename,title,level=1):
    with open(filename,'a') as f:
        for i in range(level):
            f.write("*")
        f.write(' '+title+'\n')

def preamble(fn,org_file_name=None):
    #write title
    fpath_array = fn.split('/')
    if org_file_name:
        outf = org_file_name
    else:
        outf = fpath_array[-2]+'.org'

    with open(outf,'w') as f:
        f.write("#+TITLE: New Clustering Sources \n")
        f.write('#+LATEX_HEADER: \\usepackage[margin=0.1in]{geometry} \n')
    return outf

def write_table(result_10,org_file_name):
    with open(org_file_name,'a') as f:
        f.write("nearest neighbouring known sources \n")
        f.write("| Source name | Catalog | RA (deg) | Dec (deg) | DM (pc/cm3) | Seperation (deg) | period (ms) |\n")
        if result_10['nmatches']>0:
            try:
                matches = [*result_10.keys()][7:]
                for match in matches:
                    cat = result_10[match]['survey']['value']
                    ra = round(result_10[match]['ra']['value'],2)
                    dec =  round(result_10[match]['dec']['value'],2)
                    dm =  round(result_10[match]['dm']['value'],2)
                    sep =  round(result_10[match]['distance']['value'],2)
                    p = round(result_10[match]['period']['value'],2)
                    f.write(f"|{match}|{cat}|{ra}|{dec}|{dm}|{sep}|{p}|\n")
            except Exception as e:
                import pdb; pdb.set_trace()


def generate_org(result,result_10,source,ra,dec,dm,fn,org_file_name="test.org"):
    import os.path as path
    #grab the path for files
    #get J name
    coord = SkyCoord(ra,dec,unit='deg')
    r = str(int(coord.ra.hms.h)).zfill(2)+str(int(coord.ra.hms.m)).zfill(2)
    if dec>0:
        d = '+'+str(int(dec)).zfill(2)
    else:
        d = str(int(dec)).zfill(3)
    Jname = f"J{r}{d}"
    write_title_org(org_file_name,Jname)
    write_source_location(org_file_name,ra,dec,dm)
    write_table(result_10,org_file_name)
    dirname = path.join(path.dirname(fn),'likely_astro')
    cluster_directories = os.listdir(dirname)
    for cd in cluster_directories:
        if f"_C{source}_" in cd:
            dirname = path.join(dirname,cd)
    #analysis plot
    ap_name = f'analysis{source}.png'
    ns_name = f'new_source_{source}.png'
    ap = path.join(dirname,ap_name)
    nsp = path.join(dirname,ns_name)
    #copy the figures over
    #write it out in org file
    write_image_org(org_file_name,ap)
    write_image_org(org_file_name,nsp)

    all_files = os.listdir(dirname)
    for verification_im in all_files:
        if "combined.png" in verification_im:
            im_path = path.join(dirname,verification_im)
            write_image_org(org_file_name,im_path)

def period_verify(result,period_thresh_ms=50):
    d = []
    p = []
    pulsar_name = []
    for val in result:
        try:
            d.append(result[val]['distance']['value'])
            p.append(result[val]['period']['value'])
            pulsar_name.append(val)
        except:
            pass
    ind = np.argmin(d)
    print(ind)
    p = np.array(p)
    d = np.array(d)
    if (p[ind]<period_thresh_ms)&(p[ind]>0):
        return True,pulsar_name[ind]
    else:
        return False,pulsar_name[ind]

def generate_org_msp(result,result_10,source,ra,dec,dm,fn,org_file_name="test.org"):

    import os.path as path
    #grab the path for files
    #get J name
    #only proceed for msps
    msp,pn = period_verify(result)
    if msp:
        Jname = pn
        write_title_org(org_file_name,Jname)
        write_table(result,org_file_name)
        dirname = path.join(path.dirname(fn),'likely_astro')
        cluster_directories = os.listdir(dirname)
        for cd in cluster_directories:
            if f"_C{source}_" in cd:
                dirname = path.join(dirname,cd)
        ap_name = f'analysis{source}.png'
        ns_name = f'new_source_{source}.png'
        ap = path.join(dirname,ap_name)
        nsp = path.join(dirname,ns_name)
        #analysis plot
        #copy the figures over
        #write it out in org file
        write_image_org(org_file_name,ap)
        write_image_org(org_file_name,nsp)


        all_files = os.listdir(dirname)
        for verification_im in all_files:
            if "zooniverse.png" in verification_im:
                im_path = path.join(dirname,verification_im)
                write_image_org(org_file_name,im_path)

def main():
    pulsarsurveyscraper.log.setLevel("WARNING")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fn", default='my_new_sources.csv', help="Clustering csv file")
    parser.add_argument("--dmtol", default=10, type=float, help="DM tolerance")
    parser.add_argument(
        "-r", "--radius", default=5, type=float, help="Search radius (degrees)"
    )
    parser.add_argument(
        "-radec_only", action="store_true", default=False, help="Search only ra dec"
    )
    print("Loading pulsar table...")
    pulsar_table = pulsarsurveyscraper.PulsarTable()
    args = parser.parse_args()
    radec_only = args.radec_only
    print("Loading new sources...")
    chime_csv = chime_fn.load_new_sources(args.fn)

    org_file_name = preamble(args.fn)
    for source in chime_csv:
        source_dict = chime_csv[source]
        ra=float(source_dict[0])
        dec=float(source_dict[1])
        dm=float(source_dict[2])
        coord = SkyCoord(ra, dec, unit="deg")
        #just need to set DM to None to not have a DM cut
        if radec_only:
            result = pulsar_table.search(
                    coord,
                    radius=args.radius * u.deg,
                    DM=None,
                    DM_tolerance=args.dmtol,
                    return_json=True,
                    deduplicate=False,
            )
        else:
            result = pulsar_table.search(
                    coord,
                    radius=args.radius * u.deg,
                    DM=dm,
                    DM_tolerance=args.dmtol,
                    return_json=True,
                    deduplicate=False,
            )
            result_10 = pulsar_table.search(
                    coord,
                    radius=10 * u.deg,
                    DM=dm,
                    DM_tolerance=10,
                    return_json=True,
                    deduplicate=False,
            )
        if result["nmatches"]==0:
            print(result)
            print([source,float(ra),float(dec),dm])
            generate_org(result,result_10,source,ra,dec,dm,args.fn,org_file_name=org_file_name)
        else:
            # generate_org_msp(result,result_10,source,ra,dec,dm,args.fn,org_file_name="msps.org")
            pass
        # plot_matches(result,source,ra,dec,dm)
if __name__ == "__main__":
    main()
