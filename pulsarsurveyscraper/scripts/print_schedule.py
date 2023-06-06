#!/usr/bin/env python3

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

def main():
    pulsarsurveyscraper.log.setLevel("WARNING")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fn", default='my_new_sources.csv', help="Clustering csv file")
    parser.add_argument("--read_fn", help="csv file to read")
    pulsar_table = pulsarsurveyscraper.PulsarTable()
    args = parser.parse_args()
    chime_csv = chime_fn.load_new_sources(args.fn)
    unique_survey = []
    matched_survey = []
    for source in chime_csv:
        source_dict = chime_csv[source]
        ra=float(source_dict[0])
        dec=float(source_dict[1])
        dm=float(source_dict[2])


if __name__ == "__main__":
    main()
