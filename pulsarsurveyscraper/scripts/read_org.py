#!/usr/bin/env python3

import numpy as np
import sys
import yaml
from PIL import Image
import random
from astropy.coordinates import SkyCoord as sk
import os
def generate_images(pc,num_stack=4):
    image_arr = pc.image_array
    print(f"{pc.name} has {len(image_arr)} images")
    outfn = pc.name+".jpg"
    if len(image_arr)<num_stack:
        return False
    image_ind = []
    while len(image_ind)<(num_stack-2):
        ind = random.randint(2,len(image_arr)-1)
        if not (ind in image_ind):
            image_ind.append(ind)
    images = []
    images.append(image_arr[0])
    images.append(image_arr[1])
    for i in image_ind:
        images.append(image_arr[i])
    images = [Image.open(x) for x in images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(outfn)
    return True

class PulsarCandidate:
    def __init__(self,name,ra,dec,dm,image_array,ra_hms,dec_dms):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.dm = dm
        self.image_array = image_array
        self.ra_hms = ra_hms
        self.dec_dms = dec_dms

org_file = sys.argv[1]
pulsar_candidates = []
with open(org_file,'r') as org_f:
    #skip the first two lines
    coord_next = False
    file_array = []
    for i,line in enumerate(org_f):
        if i<1:
            continue
        if line[0]=="*":
            #this is the start of a new source
            #create the candidate
            if i>2:
                pc = PulsarCandidate(name,ra,dec,dm,file_array,hmsdms_str[0],hmsdms_str[1])
                pulsar_candidates.append(pc)
                print(f"Added {pc.name} to the list with ra={pc.ra}, dec={pc.dec}, dm={pc.dm}")
            #reset everything
            ra = 0
            dec = 0
            dm = 0
            file_array = []
            hmsdms_str= ["",""]
            name = line[2:-1]
        elif line=="|ra|dec|dm|\n":
            #coordinates incoming
            coord_next = True
        elif coord_next:
            radecdm = line.split('|')
            ra = float(radecdm[1])
            dec = float(radecdm[2])
            skycoord = sk(ra,dec,unit="deg")
            hmsdms_str = skycoord.to_string("hmsdms",sep=":",precision=0).split(" ")
            dm = float(radecdm[3])
            coord_next = False
        elif line[0:2]=="[[":
            my_f = line[7:-3]
            file_array.append(my_f)
    #the last source would never get registered so lets register that here
    pc = PulsarCandidate(name,ra,dec,dm,file_array,hmsdms_str[0],hmsdms_str[1])
    pulsar_candidates.append(pc)
    print(f"Added {pc.name} to the list with ra={pc.ra}, dec={pc.dec}, dm={pc.dm}")

def write_yaml(pulsar_candidate,outfn):
    #remove the if it exists

    pulsar_candidate.dm=round(pulsar_candidate.dm,2)

    out_dict = {pulsar_candidate.name:
                #set dm to 2sf
                {"dm":{"value":pulsar_candidate.dm,"error_low":0,"error_high":0},
                "ra":{"value":pulsar_candidate.ra_hms},
                "dec":{"value":pulsar_candidate.dec_dms},
                "period":{"value":0,"error_low":0,"error_high":0}}
                }
    with open(outfn,'a') as f:
        yaml.dump(out_dict,f)
outfn = sys.argv[1].strip(".org")+".yaml"
if os.path.exists(outfn):
    os.remove(outfn)
for p in pulsar_candidates:
    if generate_images(p):
        #only run if there are enough images
        write_yaml(p,outfn)

#print data to upload to spread sheet
for p in pulsar_candidates:
    print(f"{p.name},,{p.dm},{p.ra_hms},{p.dec_dms},{p.ra},{p.dec}")
#write YAML file
