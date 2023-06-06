import matplotlib.pyplot as plt
import numpy as np
import os.path
import os
from datetime import timedelta as td
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import datetime
from datetime import datetime as dt
import gc
import multiprocessing as mp
import csv
import frb_L2_L3.actors.dm_checker as dmcheck
import re
import pytz


def add_TNS_cat(original_search_params):
    from frbcat import TNS
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    tns = TNS()
    df = tns.df
    # create chime mask
    mask = list(df.telescope != "CHIME")
    # get tns param,s
    tns_ra = df.ra[mask]
    tns_dec = df.decl[mask]
    tns_dm = np.array(df.dm[mask])
    coords = SkyCoord(tns_ra, tns_dec, unit=(u.hourangle, u.deg))
    tns_ks = np.array(df.name[mask])
    tns_snr_arr = np.array(df.snr[mask])
    # make up the rest
    tns_dm_error = np.zeros(len(tns_dec))
    tns_L2_rfi_grade_threshold = np.zeros(len(tns_dec)) + 10
    tns_event_no = np.zeros(len(tns_dec)) - 1
    print(len(tns_dec))
    utc = pytz.utc
    tns_dt = np.array(
        list(
            utc.localize(dt.fromtimestamp(tns_timestamp.timestamp()))
            for tns_timestamp in df.discovery_date[mask]
        )
    )
    tns_sp = search_params(
        ra=coords.ra.deg,
        dec=coords.dec.deg,
        dm=tns_dm,
        dm_error=tns_dm_error,
        known_sources=tns_ks,
        event_no=tns_event_no,
        event_time=tns_dt,
        snr_arr=tns_snr_arr,
        l2_rfi_sifter_grade=tns_L2_rfi_grade_threshold,
    )
    original_search_params.combine(tns_sp)
    return original_search_params


def in_tolerance(number, lower, upper):
    """

    :param number:this is the number we're comparing to see if it's in tolerance
    :param lower: lower bound
    :param upper: upper bound

    """
    ret = (number > lower) & (number < upper)
    return ret


def cyclic(array, threshold, dec):
    """
    we put ra on a cyclic function because we need 360 deg and 0 to loop
    around
    :param array: this is the array we want to put on the cyclic function
    """
    # split into segments
    one = np.argwhere(array < 90)
    two = np.argwhere((array >= 90) & (array < 270))
    onecos = np.argwhere(array <= 180)
    twocos = np.argwhere(array > 180)
    three = np.argwhere(array >= 270)
    fn1 = np.copy(array)
    fn2 = np.copy(array)
    # remapping fn1
    fn1[two] = 180 - fn1[two]
    fn1[three] = fn1[three] - 360
    # remapping fn2
    fn2[onecos] = 90 - fn2[onecos]
    fn2[twocos] = fn2[twocos] - 270
    # apply thresholds
    fn1 = np.array(fn1) / np.array(threshold)
    fn2 = np.array(fn2) / np.array(threshold)
    """ 
    plt.figure()
    plt.title('Function 1')
    plt.scatter(array,fn1,c=dec)
    bar=plt.colorbar()
    bar.set_label('Declination (deg)')
    plt.xlabel('RA (deg)')
    plt.ylabel('Function 1')
    plt.figure()
    plt.title('Function 2')
    plt.scatter(array,fn2,c=dec)
    plt.xlabel('RA (deg)')
    plt.ylabel('Function 1')
    bar=plt.colorbar()
    bar.set_label('Declination (deg)')
    plt.show()
    """
    return fn1, fn2


def make_KS_plot_csv(
    repeater_folder,
    sub_folder,
    candidate_dir,
    ra,
    dec,
    dm,
    known_sources_name,
    event_time,
    event_number,
    tolerance=1,
):
    plt.figure()
    plt.title("position plot " + str(candidate_dir))
    plt.xlabel("ra(deg)")
    plt.ylabel("dec(deg)")

    my_path = repeater_folder + "/" + sub_folder + "/" + candidate_dir
    numpy_file_path = my_path + "/data.npz"
    candidate_properties = np.load(numpy_file_path, allow_pickle=1)
    candidate_event_times = candidate_properties["new_source_time"]
    candidate_cluster_no = candidate_properties["new_source_cluster_no"]
    candidate_dm = candidate_properties["new_source_dm"]
    candidate_ra = candidate_properties["new_source_ra"]
    candidate_dec = candidate_properties["new_source_dec"]
    candidate_event_no = candidate_properties["new_event_no"]
    plt.scatter(candidate_ra, candidate_dec, s=10, marker="^", label="this cluster")
    ra_lim = [np.min(candidate_ra) - tolerance, np.max(candidate_ra) + tolerance]
    dec_lim = [np.min(candidate_dec) - tolerance, np.max(candidate_dec) + tolerance]
    dm_lim = [np.min(candidate_dm) - tolerance, np.max(candidate_dm) + tolerance]
    mask = (
        (ra > ra_lim[0])
        & (ra < ra_lim[1])
        & (dec > dec_lim[0])
        & (dec < dec_lim[1])
        & (dm > dm_lim[0])
        & (dm < dm_lim[1])
    )
    known_sources_set = set(known_sources_name[mask])
    ks_score_path = my_path + "/ks_scores.csv"
    with open(ks_score_path, "w+") as scores:
        # write the scores in and also event numbers
        writer = csv.writer(scores, delimiter=" ")
        writer.writerow(["eventno", "ra", "dec", "dm", "event_time", "ks"])
        for i in range(len(candidate_dm)):
            item = [
                candidate_event_no[i],
                candidate_ra[i],
                candidate_dec[i],
                candidate_dm[i],
                candidate_event_times[i],
                "None",
            ]
            writer.writerow(item)
        for k, source in enumerate(known_sources_set):
            ks_mask = known_sources_name == source
            total = mask & ks_mask
            my_ra = ra[total]
            my_dec = dec[total]
            my_dm = dm[total]
            my_event_time = event_time[total]
            my_ks = known_sources_name[total]
            my_event_no = event_number[total]
            plt.scatter(my_ra, my_dec, marker="o", s=10, label=str(source))
            for i in range(len(my_dm)):
                item = [
                    my_event_no[i],
                    my_ra[i],
                    my_dec[i],
                    my_dm[i],
                    my_event_time[i],
                    my_ks[i],
                ]
                writer.writerow(item)
    plt.legend()
    plt.savefig(my_path + "/other_ks.png")
    print("plot output KS assoc " + str(candidate_dir))
    plt.close()


def make_single_dm_time_csv(
    repeater_folder,
    sub_folder,
    candidate_dir,
    ra,
    dec,
    dm,
    event_time,
    suffix="",
    csv_limit=500,
):
    """
    creates a csv file for all events in cluster
    :param repeater_folder: Folder to search in for repeaters
    :param sub_folder: either "astro" or "rfi"
    :param candidate_dir: director for candidates i.e. C153_Rxxx_Dxxx
    :param ra: deg
    :param dec: deg
    :param dm: cm/pc^3
    :param event_time: time of event
    :param suffix: suffix of output file (Default value = '')

    """
    td_window = td(seconds=120)
    my_path = repeater_folder + "/" + sub_folder + "/" + candidate_dir
    numpy_file_path = my_path + "/data.npz"
    candidate_properties = np.load(numpy_file_path, allow_pickle=1)
    candidate_event_no = candidate_properties["new_event_no"]
    candidate_event_times = candidate_properties["new_source_time"]
    candidate_dm_error = candidate_properties["new_source_dm_error"]
    candidate_dm = candidate_properties["new_source_dm"]
    candidate_ra = candidate_properties["new_source_ra"]
    candidate_dec = candidate_properties["new_source_dec"]
    if len(candidate_dm) < csv_limit:
        # don't plot if there's too many cause it's super slow
        for k, my_event_time in enumerate(candidate_event_times):
            min_time = my_event_time - td_window
            max_time = my_event_time + td_window
            window_index = np.where((event_time < max_time) & (event_time > min_time))
            try:
                window_dm = dm[window_index]
                window_times = event_time[window_index]
                window_ra = ra[window_index]
                window_dec = dec[window_index]
            except Exception as e:
                print(e)
                print(window_index)
                print(event_time)
            with open(
                my_path + "/" + str(int(candidate_event_no[k])) + suffix + ".csv", "w"
            ) as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                for i in range(len(window_dm) - 1):
                    writer.writerow(
                        [window_times[i], window_ra[i], window_dec[i], window_dm[i]]
                    )
    print("csv written " + str(candidate_dir))


def make_single_dmtime(
    repeater_folder,
    sub_folder,
    candidate_dir,
    dm,
    event_time,
    suffix="",
    write_scores=True,
):
    """
    creates a dm-time plot for all events in cluster

    :param repeater_folder:
    :param sub_folder: "astro" or "rfi"
    :param candidate_dir: directory of candidates
    :param dm: cm/pc^3
    :param event_time: time
    :param suffix:  suffix for the image created(Default value = '')
    :param write_scores:  will write the scores to csv file(Default value = True)

    """
    td_window = td(seconds=120)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    my_path = repeater_folder + "/" + sub_folder + "/" + candidate_dir
    numpy_file_path = my_path + "/data.npz"
    candidate_properties = np.load(numpy_file_path, allow_pickle=1)
    candidate_event_no = candidate_properties["new_event_no"]
    candidate_event_times = candidate_properties["new_source_time"]
    candidate_dm_error = candidate_properties["new_source_dm_error"]
    candidate_dm = candidate_properties["new_source_dm"]
    candidate_ra = candidate_properties["new_source_ra"]
    candidate_dec = candidate_properties["new_source_dec"]
    candidate_snr = candidate_properties["new_source_snr"]
    if hasattr(candidate_properties, "new_known_sources"):
        candidate_ks = candidate_properties["new_known_sources"]
    else:
        candidate_ks = None
    if len(candidate_dm) < 500:
        for k, my_event_time in enumerate(candidate_event_times):
            min_time = my_event_time - td_window
            max_time = my_event_time + td_window
            window_index = np.where((event_time < max_time) & (event_time > min_time))
            try:
                window_dm = dm[window_index]
                window_times = event_time[window_index]
            except Exception as e:
                print(e)
                print(window_index)
                print(event_time)
            try:
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(1, 1, 1)
                ax2.set_title(str(int(candidate_event_no[k])))
                ax2.set_xlabel("time (hh:mm:ss)")
                ax2.set_ylabel("DM (pc/cm^3)")
                ax2.tick_params(labelrotation=90)
                ax2.scatter(window_times, window_dm, color="red", marker="o", s=10)
                ax2.scatter(
                    candidate_event_times[k],
                    candidate_dm[k],
                    color="blue",
                    marker="^",
                    s=40,
                )
                ax2.set_xlim([min_time, max_time])
                ax2.set_yscale("log")
                fig2.savefig(
                    my_path + "/" + str(int(candidate_event_no[k])) + "_rfi_check.jpg"
                )
                # plt.show()
                plt.close(fig2)
            except:
                print(
                    "failed on " + my_path + " evt " + str(int(candidate_event_no[k]))
                )
    try:
        ax1.set_title("Dm-Time plot" + str(candidate_dir))
        ax1.tick_params(labelrotation=90)
        ax1.set_xlabel("time")
        ax1.set_ylabel("dm")
        ax1.scatter(candidate_event_times, candidate_dm, color="blue", marker="^", s=40)
        ax1.set_yscale("log")
        ax1.errorbar(
            candidate_event_times,
            candidate_dm,
            elinewidth=0.1,
            yerr=candidate_dm_error,
            fmt="none",
        )
        ax1.set_xlim([np.min(candidate_event_times), np.max(candidate_event_times)])
        fig1.savefig(my_path + "/dm_time" + suffix + ".png")
        plt.close(fig1)
    except:
        print("failed on " + my_path)
    print("writing score " + str(candidate_dir))
    if candidate_ks:
        if write_scores:
            score_path = my_path + "/scores.txt"
            with open(score_path, "w+") as scores:
                # write the scores in and also event numbers
                writer = csv.writer(scores, delimiter=" ")
                write_array = np.column_stack(
                    (
                        candidate_event_no,
                        candidate_ra,
                        candidate_dec,
                        candidate_dm,
                        candidate_snr,
                        candidate_ks,
                        candidate_event_times,
                    )
                )
                writer.writerow(
                    ["eventno", "ra", "dec", "dm", "snr", "ks", "event_time"]
                )
                for item in write_array:
                    writer.writerow(item)

    print("plot output " + str(candidate_dir))


def get_bright_known_sources(filename, return_sources=10):
    """

    :param filename: reads the bright source file
    :param return_sources:  choose how many bright sources to return(Default value = 10)

    """
    pulsar_name = []
    pulsar_ra = []
    pulsar_dec = []
    pulsar_dm = []
    s400 = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for i, row in enumerate(reader):
            if i > 1:
                pulsar_name.append(row[1])
                pulsar_ra.append(row[2])
                pulsar_dec.append(row[3])
                pulsar_dm.append(row[4])
                s400.append(row[5])
    # get the last x values of the array
    pulsar_name.reverse()
    pulsar_ra.reverse()
    pulsar_dec.reverse()
    pulsar_dm.reverse()
    s400.reverse()
    return_array = np.array(
        [
            np.array(pulsar_name)[0:return_sources],
            np.array(pulsar_ra)[0:return_sources],
            np.array(pulsar_dec)[0:return_sources],
            np.array(pulsar_dm)[0:return_sources],
            np.array(s400)[0:return_sources],
        ]
    )
    return return_array


class search_params:
    """ """

    def __init__(self, **kwargs):
        print("creating class and updating kwargs")
        self.__dict__.update(kwargs)

    def turn_numpy_array(self):
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],list):
                self.__dict__[key]=np.array(self.__dict__[key])

    def filter_all_attributes(self, index):
        """
        :param index: the range of index's to keep
        """
        self.pos_ra_deg = self.pos_ra_deg[index]
        self.pos_dec_deg = self.pos_dec_deg[index]
        try:
            self.l1_events = self.l1_events[index]
        except Exception as e:
            print(e)
            print("probably not to worry, there are no l1 events")
        self.dm = self.dm[index]
        self.dm_error = self.dm_error[index]
        self.known_sources_name = self.known_sources_name[index]
        self.event_time = self.event_time[index]
        self.snr_arr = self.snr_arr[index]
        self.event_number = self.event_number[index]
        try:
            self.dbscan_labels = self.dbscan_labels[index]
        except Exception as e:
            print("dbscan labels haven't been made yet probably")
        try:
            self.l2_events = self.l2_events[index]
        except Exception as e:
            print('L2 events does not exist, this is probably okay')
        try:
            self.gal_dm = self.gal_dm[index]
            self.ambiguous = self.ambiguous[index]
        except Exception as e:
            print(e)
            print(
                "you probably haven't done remove galactic/extragalactic candidates yet, don't worry for now"
            )

    def convert_to_hoursminsec(self, ra, dec):
        """
        :param ra: deg
        :param dec: deg
        """
        hours_ra = ra / 15
        minutes_ra = (hours_ra - int(hours_ra)) * 60
        seconds_ra = (minutes_ra - int(minutes_ra)) * 60
        minutes_dec = (dec - int(dec)) * 60
        seconds_dec = (minutes_dec - int(minutes_dec)) * 60
        return (
            int(hours_ra),
            int(minutes_ra),
            int(seconds_ra),
            int(dec),
            int(minutes_dec),
            int(seconds_dec),
        )

    def convert_to_deg(self, HHMMSSra, DDMMSSdec):
        """
        :param HHMMSSra: self explanatory
        :param DDMMSSdec: self explanatory
        """
        ra = re.split(":", HHMMSSra)

        dec = re.split(":", DDMMSSdec)
        rhh = ra[0]
        rmm = ra[1]
        ddd = dec[0]
        dmm = dec[1]
        if len(ra) == 3:
            rss = ra[2]
        else:
            rss = 0
        if len(dec) == 3:
            dss = dec[2]
        else:
            dss = 0
        ra_deg = (float(rhh) + float(rmm) / 60 + float(rss) / 3600) * (360 / 24)
        dec_deg = float(ddd) + float(dmm) * 60 + float(dss) * 3600
        return ra_deg, dec_deg

    def get_galactic_dm(self):
        """retrieves the galactic DM and saves it inside two attributes"""
        my_dmchecker = dmcheck.DMChecker(1, 1, 1, 1)
        ambiguous = True
        counter = 0
        self.ambiguous = []
        self.gal_dm_ne2001 = []
        self.gal_dm_ymw2016 = []
        for ra, dec, event_dm in zip(self.pos_ra_deg, self.pos_dec_deg, self.dm):
            # check for extragalactic
            ymw16 = my_dmchecker.get_dm_ymw16(dec, ra)
            ne2001 = my_dmchecker.get_dm_ne2001(dec, ra)
            # check if we're in between the two
            diffymr16 = ymw16 - event_dm
            diffne2001 = ne2001 - event_dm
            if (diffymr16 * diffne2001) < 0:
                # if it's less than 0 then it means we're in between i.e. ambiguous
                ambiguous = True
            elif diffymr16 > 0:
                # if one is larger than 0 then both will be because we check for the alternating case before i.e. galactic
                ambiguous = False
            else:
                # both is below 0 i.e. extragalactic
                ambiguous = False
            self.ambiguous.append(ambiguous)
            self.gal_dm_ne2001.append(ne2001)
            self.gal_dm_ymw2016.append(ymw16)
            counter = counter + 1
        self.ambiguous = np.array(self.ambiguous)
        gc.collect()

    def remove_unclustered(self):
        #removes all the unclustered events for smaller footprint and also easier header localisation routines
        self.filter_all_attributes(self.dbscan_labels!=-1)

    def remove_large_clusters(self):
        #remove the super bright clusters
        keep_mask = np.zeros(len(self.dbscan_labels),dtype=bool)
        for label in set(self.dbscan_labels):
            mask = (label == self.dbscan_labels)
            print(label)
            if sum(mask)<500:
                keep_mask = (mask | keep_mask)
        self.filter_all_attributes((keep_mask))

    def keep_ambiguous_only(self, keep=True):
        """
        :param dm_filter_lower:  the lower filter, optional(Default value = -1)
        :param dm_filter_upper:  the upper dm filter, optional(Default value = -1)
        :param extragalactic:  true = keep extragalactic false= keep galactic(Default value = True)
        """
        my_dmchecker = dmcheck.DMChecker(1, 1, 1, 1)
        ambiguous = True
        counter = 0
        extragalactic_mask = []
        galactic_mask = []
        self.ambiguous = []
        self.gal_dm = []

        for ra, dec, event_dm in zip(self.pos_ra_deg, self.pos_dec_deg, self.dm):
            # check for extragalactic
            ymw16 = my_dmchecker.get_dm_ymw16(dec, ra)
            ne2001 = my_dmchecker.get_dm_ne2001(dec, ra)
            # check if we're in between the two
            diffymr16 = ymw16 * 1.1 - event_dm
            diffne2001 = ne2001 * 1.1 - event_dm

            ediffymr16 = ymw16 - event_dm
            ediffne2001 = ne2001 - event_dm

            diffs = [diffymr16, diffne2001, ediffymr16, ediffne2001]
            min_diffs = np.min(diffs)
            max_diffs = np.max(diffs)

            if (min_diffs * max_diffs) < 0:
                # if it's less than 0 then it means we're in between i.e. ambiguous
                ambiguous = True
                extragalactic_mask.append(counter)
                galactic_mask.append(counter)
            elif min_diffs > 0:
                # if one is larger than 0 then both will be because we check for the alternating case before i.e. galactic
                ambiguous = False
                galactic_mask.append(counter)
            elif max_diffs < 0:
                # both is below 0 i.e. extragalactic
                ambiguous = False
                extragalactic_mask.append(counter)

            self.ambiguous.append(ambiguous)
            self.gal_dm.append([ne2001, ymw16])
            counter = counter + 1

        self.gal_dm = np.array(self.gal_dm)
        self.ambiguous = np.array(self.ambiguous)
        self.filter_all_attributes(self.ambiguous)

    def remove_galactic_sources(
        self, dm_filter_lower=-1, dm_filter_upper=-1, extragalactic=True, Filter=True
    ):
        """

        :param dm_filter_lower:  the lower filter, optional(Default value = -1)
        :param dm_filter_upper:  the upper dm filter, optional(Default value = -1)
        :param extragalactic:  true = keep extragalactic false= keep galactic(Default value = True)

        """
        my_dmchecker = dmcheck.DMChecker(1, 1, 1, 1)
        ambiguous = True
        counter = 0
        extragalactic_mask = []
        galactic_mask = []
        self.ambiguous = []
        self.gal_dm = []
        if dm_filter_lower != -1:
            dm_mask_lower = (self.dm) > dm_filter_lower
            self.filter_all_attributes(dm_mask_lower)
        if dm_filter_upper != -1:
            dm_mask_upper = (self.dm) < dm_filter_upper
            self.filter_all_attributes(dm_mask_upper)

        for ra, dec, event_dm in zip(self.pos_ra_deg, self.pos_dec_deg, self.dm):
            # check for extragalactic
            ymw16 = my_dmchecker.get_dm_ymw16(dec, ra)
            ne2001 = my_dmchecker.get_dm_ne2001(dec, ra)
            # check if we're in between the two
            diffymr16 = ymw16 * 1.1 - event_dm
            diffne2001 = ne2001 * 1.1 - event_dm

            ediffymr16 = ymw16 - event_dm
            ediffne2001 = ne2001 - event_dm

            diffs = [diffymr16, diffne2001, ediffymr16, ediffne2001]
            min_diffs = np.min(diffs)
            max_diffs = np.max(diffs)

            if (min_diffs * max_diffs) < 0:
                # if it's less than 0 then it means we're in between i.e. ambiguous
                ambiguous = True
                extragalactic_mask.append(counter)
                galactic_mask.append(counter)
            elif min_diffs > 0:
                # if one is larger than 0 then both will be because we check for the alternating case before i.e. galactic
                ambiguous = False
                galactic_mask.append(counter)
            elif max_diffs < 0:
                # both is below 0 i.e. extragalactic
                ambiguous = False
                extragalactic_mask.append(counter)

            self.ambiguous.append(ambiguous)
            self.gal_dm.append([ne2001, ymw16])
            counter = counter + 1

        self.gal_dm = np.array(self.gal_dm)
        self.ambiguous = np.array(self.ambiguous)

        extragalactic_mask = np.array(extragalactic_mask)
        galactic_mask = np.array(galactic_mask)
        if extragalactic:
            if len(extragalactic_mask) > 0:
                self.filter_all_attributes(extragalactic_mask)
        else:
            if len(galactic_mask) > 0:
                self.filter_all_attributes(galactic_mask)
        gc.collect()

    def remove_bright_sources(self, filename, dm_tolerance=3):
        """this function is to filter out all the bright sources

        :param filename: the filename for the sources obtained from atnf catalogue
        :param dm_tolerance:  tolerance for the dm of that the sources from atnf(Default value = 3)

        """
        bright_sources = get_bright_known_sources(filename, 3)
        for i, source_name in enumerate(bright_sources[0]):
            s400 = bright_sources[4][i]
            if float(s400) > 500:
                dm = float(bright_sources[3][i])
                bright_dm_mask = ~(
                    (self.dm > (dm - dm_tolerance)) & (self.dm < (dm + dm_tolerance))
                )
                self.filter_all_attributes(bright_dm_mask)
            index = np.argwhere(self.known_sources_name == source_name)
            if index.size > 0:
                # source_found
                # scale based on brightness
                print(source_name)
                # find the max ra, dec range for these events
                max_ra = (
                    (np.max(self.pos_ra_deg[index]) - np.mean(self.pos_ra_deg[index]))
                    / 0.68
                ) + np.mean(self.pos_ra_deg[index])
                min_ra = np.mean(self.pos_ra_deg[index]) - (
                    (np.mean(self.pos_ra_deg[index]) - np.min(self.pos_ra_deg[index]))
                    / 0.68
                )
                max_dec = (
                    (np.max(self.pos_dec_deg[index]) - np.mean(self.pos_dec_deg[index]))
                    / 0.68
                ) + np.mean(self.pos_dec_deg[index])
                min_dec = np.mean(self.pos_dec_deg[index]) - (
                    (np.mean(self.pos_dec_deg[index]) - np.min(self.pos_dec_deg[index]))
                    / 0.68
                )
                max_dm = np.mean(self.dm[index]) + dm_tolerance
                min_dm = np.mean(self.dm[index]) - dm_tolerance
                print([max_ra, min_ra, max_dec, min_dec, max_dm, min_dm])
                # remove things in this box I've defined
                known_source_box_mask = ~(
                    (self.pos_ra_deg < max_ra)
                    & (self.pos_ra_deg > min_ra)
                    & (self.pos_dec_deg < max_dec)
                    & (self.pos_dec_deg > min_dec)
                    & (self.dm < max_dm)
                    & (self.dm > min_dm)
                )
                print(
                    "removing "
                    + str(
                        np.size(np.argwhere(~known_source_box_mask))
                        * 100
                        / np.size(known_source_box_mask)
                    )
                    + "%"
                )
                self.filter_all_attributes(known_source_box_mask)
        gc.collect()

    def delete_event(self, my_event_numbers):
        """

        :param my_event_numbers: list, event numbers to delete

        """
        index = np.argwhere(my_event_numbers == self.event_number)
        self.pos_ra_deg = np.delete(self.pos_ra_deg, index)
        self.pos_dec_deg = np.delete(self.pos_dec_deg, index)
        try:
            self.l1_events = np.delete(self.l1_events, index)
        except:
            print("Exception in delete")
            print(
                "couldn't delete l1_event, this is okay sometimes, the numpy files have no l1 events"
            )
        self.dm = np.delete(self.dm, index)
        self.dm_error = np.delete(self.dm_error, index)
        self.known_sources_name = np.delete(self.known_sources_name, index)
        self.event_time = np.delete(self.event_time, index)
        self.snr_arr = np.delete(self.snr_arr, index)
        self.event_number = np.delete(self.event_number, index)

    def seek_rfi(self, rfi_search_params, window=5):
        """
        remove rfi events in clusters
        :param rfi_search_params is the rfi search params numpy file
        :param window of time around the event
        """
        rfi_ra = rfi_search_params.pos_ra_deg
        rfi_dec = rfi_search_params.pos_dec_deg
        rfi_dm = rfi_search_params.dm
        rfi_et = rfi_search_params.event_time
        window_dt = td(seconds=60)
        for label in set(self.dbscan_labels):
            if label != -1:
                # if label==585:
                ind = label == self.dbscan_labels
                cluster_ra = self.pos_ra_deg[ind]
                cluster_dec = self.pos_dec_deg[ind]
                cluster_dm = self.dm[ind]
                cluster_et = self.event_time[ind]

                if len(cluster_et) > 500:
                    continue
                # now lets cycle through each event
                for i, time in enumerate(cluster_et):
                    ra = cluster_ra[i]
                    dec = cluster_dec[i]
                    dm = cluster_dm[i]
                    t_low = time - window_dt
                    t_high = time + window_dt

                    dt_mask = (rfi_et > t_low) & (rfi_et < t_high)
                    r_ra_masked = rfi_ra[dt_mask]
                    r_dec_masked = rfi_dec[dt_mask]
                    r_dm_masked = rfi_dm[dt_mask]
                    r_et_masked = rfi_et[dt_mask]

                    # lets append them together
                    # r_ra_masked=np.append(r_ra_masked,ra)
                    # r_dec_masked=np.append(r_dec_masked,dec)

                    r_dm_masked = np.append(r_dm_masked, dm)
                    r_et_masked = np.append(r_et_masked, time)
                    # test out logged dm
                    # r_dm_masked = np.log10(r_dm_masked)
                    # convert all timestamps to time from epoch in seconds

                    r_et_masked = np.array(list(t.timestamp() for t in r_et_masked))
                    mean_time = np.min(r_et_masked)
                    r_et_masked = r_et_masked - mean_time

                    ra_thresh = self.thresholds()
                    dec_thresh = 0.5
                    dm_thresh = 100
                    time_thresh = 5
                    # excluding ra and dec because I think they're set to 0 for rfi events.
                    # ra_dec_dm_t[:, 0] = self.pos_ra_deg/ra_thresh
                    # ra_dec_dm_t[:, 1] = r_dec_masked/dec_thresh
                    dm_t = np.empty((len(r_et_masked), 2), np.float32)

                    dm_t[:, 0] = r_dm_masked / dm_thresh
                    dm_t[:, 1] = r_et_masked / time_thresh
                    min_pts = 20
                    # lets cluster on this.
                    print("starting dbscan " + str(label))
                    rfi_dbscan = DBSCAN(eps=1, min_samples=min_pts).fit(dm_t)
                    print("finishing dbscan")
                    rfi_dbscan_labels = rfi_dbscan.labels_

                    # lets find the clusters
                    ul = set(rfi_dbscan_labels)
                    for rfi_label in ul:
                        if rfi_label != -1:
                            mask = rfi_label == rfi_dbscan_labels
                            dm_cluster = r_dm_masked[mask]
                            t_cluster = r_et_masked[mask]
                            # check if the cluster event is in there
                            if time.timestamp() - mean_time in t_cluster:
                                if dm in dm_cluster:
                                    if np.sum(mask) > 10:
                                        max_ind = np.where(
                                            np.max(dm_cluster) == dm_cluster
                                        )[0][0]
                                        min_ind = np.where(
                                            np.min(dm_cluster) == dm_cluster
                                        )[0][0]
                                        max_min_elapse = (
                                            t_cluster[max_ind] - t_cluster[min_ind]
                                        )
                                        t_span = np.max(t_cluster) - np.min(t_cluster)
                                        if (
                                            dm_cluster[max_ind] - dm_cluster[min_ind]
                                        ) > 100:
                                            if max_min_elapse > (0.45 * t_span):
                                                event_index = self.event_time != time
                                                # filter all attributes is to keep. So we need the not
                                                self.filter_all_attributes(event_index)
                                                plt.figure()
                                                plt.scatter(
                                                    r_et_masked,
                                                    np.log10(r_dm_masked),
                                                    marker="o",
                                                )
                                                plt.scatter(
                                                    time.timestamp() - mean_time,
                                                    np.log10(dm),
                                                    marker="o",
                                                )
                                                plt.savefig(
                                                    "test/"
                                                    + str(label)
                                                    + str(i)
                                                    + str(rfi_label)
                                                    + ".png",
                                                    bbox_inches="tight",
                                                    dpi=300,
                                                )
                                                plt.close()
                                                plt.figure()
                                                plt.scatter(
                                                    r_et_masked[mask],
                                                    np.log10(dm_cluster),
                                                )
                                                plt.scatter(
                                                    time.timestamp() - mean_time,
                                                    np.log10(dm),
                                                    marker="o",
                                                )
                                                # plt.show()
                                                # import pdb; pdb.set_trace()
                                                plt.savefig(
                                                    "test/cluster_"
                                                    + str(label)
                                                    + str(i)
                                                    + str(rfi_label)
                                                    + ".png",
                                                    bbox_inches="tight",
                                                    dpi=300,
                                                )
                                                plt.close()

    def create_cluster_L2(self, cluster_number=-1, outfn="l2_information", localise_all=False, max_cluster_size=500):
        """
        create npz for Alex's header localisations
        :cluster_number is the specific cluster number
        :param outfn
        :param label this is the cluster label that we want to generate this for
        :the output will be a number file
        """
        # go through all the labels, and either localise all, or localise by clusters
        labels = self.dbscan_labels
        unique = set(labels)
        #lets just filter for non-clustered things
        if localise_all:
            keep_ind = np.array([])
            for l in unique:
                print(l)
                if l != -1:
                    if sum(labels==l) < max_cluster_size:
                        #we keep it if there are less than 500(default) points
                        keep_ind=np.append(keep_ind,np.where(labels == l))
            keep_ind = list(int(k) for k in keep_ind)
        elif cluster_number != -1:
            keep_ind = labels==cluster_number
        self.filter_all_attributes(keep_ind)

    def plot_dm_time_associations(
        self, repeater_folder, sub_folder="likely_astro", suffix=""
    ):
        """
        makes the dm_time plots to find rfi
        :param repeater_folder: folder with everything
        :param sub_folder:  "likely_astro" or "likely_rfi" (Default value = 'likely_astro')
        :param suffix:  suffix for the plot names (Default value = '')

        """
        repeater_candidate_dir = os.listdir(repeater_folder + "/" + sub_folder)
        cores = 5
        pool = mp.Pool(cores)
        # for candidate_dir in repeater_candidate_dir:
        #    make_single_dm_time_csv(repeater_folder,sub_folder,candidate_dir,self.pos_ra_deg,self.pos_dec_deg,self.dm,self.event_time,suffix)
        # result=pool.starmap(make_single_dm_time_csv, [(repeater_folder,sub_folder,candidate_dir,self.pos_ra_deg,self.pos_dec_deg,self.dm,self.event_time,suffix) for candidate_dir in repeater_candidate_dir])
        # result.get()
        result = pool.starmap(
            make_single_dmtime,
            [
                (
                    repeater_folder,
                    sub_folder,
                    candidate_dir,
                    self.dm,
                    self.event_time,
                    suffix,
                )
                for candidate_dir in repeater_candidate_dir
            ],
        )
        # result.get()

        pool.close()

    def known_sources_associations(
        self, repeater_folder, sub_folder="likely_astro", tolerance=1
    ):
        """
        this isn't really used anymore, this finds all the known sources around folder
        :param repeater_folder: folder with all the clusters
        :param sub_folder:  likely astro or likely rfi(Default value = 'likely_astro')

        """
        repeater_candidate_dir = os.listdir(repeater_folder + "/" + sub_folder)
        # print(set(self.known_sources_name))
        cores = 1
        pool = mp.Pool(cores)
        # for candidate_dir in repeater_candidate_dir:
        #    make_single_dm_time_csv(repeater_folder,sub_folder,candidate_dir,self.pos_ra_deg,self.pos_dec_deg,self.dm,self.event_time,suffix)
        # result=pool.starmap(make_KS_plot_csv, [(repeater_folder,sub_folder,candidate_dir,self.pos_ra_deg,self.pos_dec_deg,self.dm,self.known_sources_name,self.event_time,self.event_number,tolerance) for candidate_dir in repeater_candidate_dir])
        for candidate_dir in repeater_candidate_dir:
            make_KS_plot_csv(
                repeater_folder,
                sub_folder,
                candidate_dir,
                self.pos_ra_deg,
                self.pos_dec_deg,
                self.dm,
                self.known_sources_name,
                self.event_time,
                self.event_number,
                tolerance,
            )

    def compare_clusters(self, previous_search_param, output):
        """
        param: previous_search_param: The search param object for yesterday to compare
        output: directory to put the new cluster images
        """
        labels = self.dbscan_labels
        unique_labels = set(labels)
        previous_labels = previous_search_param.dbscan_labels
        previous_unique_labels = set(previous_labels)
        # delete the -1 cause they means no cluster...
        previous_unique_labels = np.array(list(previous_unique_labels))
        unique_labels = np.array(list(unique_labels))
        previous_unique_labels = np.delete(
            previous_unique_labels, (np.argwhere(previous_unique_labels == -1))
        )
        unique_labels = np.delete(unique_labels, (np.argwhere(unique_labels == -1)))
        previous_label_lookup = []
        current_label_lookup = []
        modified_dbscan_labels = np.zeros(len(labels))
        matched = False
        new_cluster_increment = 1
        for i in unique_labels:
            index = labels == i
            for k in previous_unique_labels:
                index_k = previous_labels == k
                event_numbers = self.event_number[index]
                previous_event_numbers = previous_search_param.event_number[index_k]
                intersect = np.intersect1d(event_numbers, previous_event_numbers)
                if len(intersect) > 0:
                    # if there's some intersect ,it mean's we've matched, break and move on
                    previous_label_lookup.append(k)
                    current_label_lookup.append(i)
                    matched = True
                    break
            if matched:
                # here we've matched, this means that the current clusters has a counterpart in the previous clusters. We should stabilise the cluster numbers
                modified_dbscan_labels[index] = previous_label_lookup[-1]
            else:
                modified_dbscan_labels[index] = (
                    max(previous_unique_labels) + new_cluster_increment
                )
                new_cluster_increment = new_cluster_increment + 1
                print("found a new cluster: " + str(i))
                current_label_lookup.append(
                    max(previous_unique_labels) + new_cluster_increment
                )
                previous_label_lookup.append(-1)
            matched = False

        previous_label_lookup = np.array(previous_label_lookup)
        current_label_lookup = np.array(current_label_lookup)
        print("previous cluster to current cluster pairings (-1) indicates new cluster")
        for i in range(len(current_label_lookup)):
            print(str(previous_label_lookup[i]) + ":" + str(current_label_lookup[i]))

        new_clusters_index = np.argwhere(previous_label_lookup == -1)
        self.dbscan_labels = np.array(modified_dbscan_labels, dtype=int)
        np.save("newdb", self.dbscan_labels)
        """
        for i in new_clusters_index:
            print('plotting new cluster')
            print('plotting cluster '+str(np.squeeze(i)))
            self.plot_specific_cluster(np.squeeze(current_label_lookup[np.squeeze(i)]),output_directory=output)
        """

    def remove_known_sources_sifter(self, remove_unknown=False):
        """
        this function is to only keep those sources where known sources sifter doesnt recognise
        """
        # lets remove all the pulsars
        is_new = []
        # lets make sure all the none's are the same
        if len(self.known_sources_name) > 0:
            for ks_name, i in enumerate(self.known_sources_name):
                if ks_name == None:
                    # old database entries had None as well
                    self.known_sources_name[i] = "None"
                elif ks_name == b"None":
                    # old database entries used to be binary strings
                    self.known_sources_name[i] = "None"
                elif ks_name == "None":
                    # essentially do nothing
                    self.known_sources_name[i] = "None"

            if remove_unknown:
                # this is not really recomended as it removed known FRBs
                NoneMask = self.known_sources_name != "None"
                print("start********************************\n")
                print(NoneMask)
                print(self.known_sources_name)
                self.filter_all_attributes(np.array(NoneMask))
                print(set(self.known_sources_name))
                print("\nend********************************")

            else:
                for i, source in enumerate(self.known_sources_name):
                    try:
                        plus = "+" in source
                        minus = "-" in source
                        plusminus = plus | minus
                        begins_j = source[0].lower() == "j"
                        begins_b = source[0].lower() == "b"
                        begins_pulsar = begins_j | begins_b
                        if not (plusminus & begins_pulsar):
                            is_new.append(i)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        print(source)
                        print(type(source))
                if is_new:
                    try:
                        self.filter_all_attributes(np.array(is_new))
                    except Exception as e:
                        print(e)

    def combine(self, combinee):
        """remove repeats first

        :param combinee:combines the combinee with itself

        """
        my_event_no = np.array(self.event_number)
        combinee_event_no = combinee.event_number
        """
        print('items in combiner')
        print(my_event_no)
        print('items in combinee')
        print(combinee_event_no)
        """
        intersect = np.intersect1d(my_event_no, combinee_event_no)
        keep = np.ones(len(my_event_no), dtype=bool)
        for overlap in intersect:
            temp = my_event_no != overlap
            keep = temp & keep
        if np.size(keep) > 0:
            keep = np.array(keep)
            self.filter_all_attributes(keep)
        # print('unique event nums')
        # print(len(set(self.event_number)))
        # print('total event nums')
        # print(len(self.event_number))
        self.pos_ra_deg = np.concatenate((self.pos_ra_deg, combinee.pos_ra_deg))
        self.pos_dec_deg = np.concatenate((self.pos_dec_deg, combinee.pos_dec_deg))
        self.event_time = np.concatenate((self.event_time, combinee.event_time))
        self.dm = np.concatenate((self.dm, combinee.dm))
        self.dm_error = np.concatenate((self.dm_error, combinee.dm_error))
        try:
            self.l1_events = np.concatenate((self.l1_events, combinee.l1_events))
        except Exception as e:
            print("Exception in combine")
            print("We're not using L1 events, no worries")
        try:
            self.l2_events = np.concatenate((self.l2_events,combinee.l2_events))
        except Exception as e:
            print("Exception in combine to do with L2 events")
        self.known_sources_name = np.concatenate((self.known_sources_name,combinee.known_sources_name))
        self.event_number = np.concatenate((self.event_number,combinee.event_number))
        self.snr_arr=np.concatenate((self.snr_arr,combinee.snr_arr))

    def sort_by_event_no(self):
        """ """
        # after combination, sort by event number

        sort_ind = np.argsort(self.event_number)
        self.dm_error = self.dm_error[sort_ind]
        self.event_number = self.event_number[sort_ind]
        self.pos_ra_deg = self.pos_ra_deg[sort_ind]
        self.pos_dec_deg = self.pos_dec_deg[sort_ind]
        self.event_time = self.event_time[sort_ind]
        self.dm = self.dm[sort_ind]
        self.known_sources_name = self.known_sources_name[sort_ind]
        self.snr_arr = self.snr_arr[sort_ind]
        self.l2_events = self.l2_events[sort_ind]

    def thresholds(self, dec=None):
        """Declination-dependent thresholds for rescaling of the nearest
        neighbour search.

        :param dec: Declination of the cluster.
        :type dec: float
        :returns: thresholds->     Right ascension, declination and DM threshold to rescale with.
        :rtype: array_like

        """
        # max EW spacing + 2 * 0.5 degree FWHM at 400 MHz
        if dec == None:
            dec = self.pos_dec_deg

        self.ra_error = np.array((1.2 + 1.0) / np.cos(np.deg2rad(dec)))
        return np.array(self.ra_error)

    def radec_dbscan(self, pool=None, cores=16, min_pts=2):
        """to be updated
        only does db_scan for ra_dec to find evolutions in dm
        Inputs
        ----------
        grid_parameter: int
        This is the position on the grid, 0 = ra 1= dec 2 = dm
        grid_size: float
        this is the amount of elements in grid
        pool: mp.pool
        this is the multiprocessing pool
        overlap: float
        This is the overlap of the grid, default value is set for dm

        :param pool:  (Default value = None)
        :param cores:  (Default value = 16)
        :param min_pts:  (Default value = 2)
        """
        # make a new (ra, dec, dm) array that will be scaled by thresholds
        ra_dec = np.empty((len(self.pos_ra_deg), 3), np.float32)
        # radecdm = np.empty((len(self.pos_ra_deg), 3), np.float32)

        # tree_index = np.empty(len(self.pos_ra_deg), int)
        dec_thresh = 1
        ra_thresh = self.thresholds()
        # gets the ra dec dm and scale it so that the distance away is 1
        # please note, I experimented with different combinations of dividing by thresh etc
        ra_dec[:, 0] = self.pos_ra_deg / ra_thresh
        ra_dec[:, 1] = self.pos_dec_deg / dec_thresh
        ra_dec[:, 2] = self.dm / 20
        print("starting dbscan")
        self.my_gridded_db = DBSCAN(eps=1, min_samples=min_pts, algorithm="brute").fit(
            ra_dec
        )
        self.dbscan_labels = self.my_gridded_db.labels_
        self.dbscan_event_no = self.event_number
        print("finishing db scan")
        # the return values are written as tuples

    def targetted_dbscan(self, event_numbers):
        """this function is created to test clustering on particular clusters
        inputs:
        event_numbers: array of event numbers to dbscan
        :param event_numbers:
        :returns: grid_db: dbscan object
        """
        event_index = []
        for number in event_numbers:
            event_index.append(np.argwhere(self.event_number == number))
        event_index = np.array(event_index)
        ra_dec_dm = np.empty((np.size(event_index), 3), np.float32)
        # tree_index = np.empty(len(self.pos_ra_deg), int)
        dec_thresh = 1
        dm_thresh = 13
        ra_thresh = self.thresholds()[event_index]
        # gets the ra dec dm and scale it so that the distance away is 1
        # please note, I experimented with different combinations of dividing by thresh etc
        ra_dec_dm[:, 0] = np.squeeze(self.pos_ra_deg[event_index] / ra_thresh)
        ra_dec_dm[:, 1] = np.squeeze(self.pos_dec_deg[event_index] / dec_thresh)
        ra_dec_dm[:, 2] = np.squeeze(self.dm[event_index] / dm_thresh)
        db = DBSCAN(eps=1, min_samples=2, algorithm="brute").fit(ra_dec_dm)
        labels = db.labels_
        print(labels)
        print(ra_dec_dm)
    def create_clusters(self,directory = "fine_tuning"):
        #this method would create cluster objects using the cluster.py class
        from clusters import clusters
        #get list of clusters
        labels = self.dbscan_labels
        unique = set(labels)
        self.fine_tune = []
        for l in unique:
            mask = (l==labels)
            ra = self.localised_pos_ra_deg[mask]
            ra_error = self.ra_error[mask]
            dec = self.localised_pos_dec_deg[mask]
            dec_error = self.dec_error[mask]
            dm = self.dm[mask]
            dm_error = self.dm_error[mask]
            c = clusters(ra,ra_error,dec,dec_error,dm,dm_error,l)
            c.start_dbscan()
            self.fine_tune.append(c)
            if not os.path.exists(directory + "/"):
                os.mkdir(directory + "/")

            c.plot_and_save(directory)
        self.fine_tune = np.array(self.fine_tune)

    def do_grided_dbscan(self, min_pts=2, eps=1, wrapping=True, metric="euclidean"):
        """to be updated
        Inputs
        ----------
        cyclic: bool
        this is for wrapping at 360 by mapping onto 2 cyclic functions
        grid_parameter: int
        This is the position on the grid, 0 = ra 1= dec 2 = dm
        grid_size: float
        this is the amount of elements in grid
        pool: mp.pool
        this is the multiprocessing pool
        overlap: float
        This is the overlap of the grid, default value is set for dm

        :param min_pts:  (Default value = 5)
        :param eps:  (Default value = 1)
        """
        # make a new (ra, dec, dm) array that will be scaled by thresholds
        dec_thresh = 1
        dm_thresh = 13
        ra_thresh = self.thresholds()
        if wrapping:
            ra_dec_dm = np.empty((len(self.pos_ra_deg), 4), np.float32)

            print("pocessing cyclic")
            ra1, ra2 = cyclic(self.pos_ra_deg, ra_thresh, self.pos_dec_deg)
            # ra1 and ra2 already have thresh applied
            ra_dec_dm[:, 0] = ra1
            ra_dec_dm[:, 1] = ra2
            ra_dec_dm[:, 2] = self.pos_dec_deg / dec_thresh
            ra_dec_dm[:, 3] = self.dm / dm_thresh
        else:
            ra_dec_dm = np.empty((len(self.pos_ra_deg), 3), np.float32)
            ra_dec_dm[:, 0] = self.pos_ra_deg / ra_thresh
            ra_dec_dm[:, 1] = self.pos_dec_deg / dec_thresh
            ra_dec_dm[:, 2] = self.dm / dm_thresh

        print("starting dbscan")
        self.my_gridded_db = DBSCAN(
            eps=eps, min_samples=min_pts, algorithm="brute", metric=metric
        ).fit(ra_dec_dm)
        # self.my_gridded_db=DBSCAN(eps=eps,min_samples=min_pts,algorithm='brute',metric='precomputed').fit(ra_dec_dm)
        self.dbscan_labels = self.my_gridded_db.labels_
        self.dbscan_event_no = self.event_number
        print("finishing db scan")
        # the return values are written as tuples

    def daily_new_events(self):
        """
        create a plot to show the number of new events from clusters per day
        """
        labels = self.dbscan_labels
        save_data = []
        save_event = []
        utc = pytz.utc
        start_dt = utc.localize(datetime(2020, 1, 1))
        end_dt = utc.localize(datetime(2020, 5, 1))
        t_diff = (end_dt - start_dt).days
        timedelta = td(days=1)
        my_dates = []
        my_events = []
        for day in range(t_diff):
            events = 0
            for my_labels in set(labels):
                index = my_labels == labels
                cluster_event_numbers = self.event_number[index]
                cluster_dates = self.event_time[index]
                if len(cluster_dates) < 60:
                    for date in cluster_dates:
                        if (date > (start_dt + (day * timedelta))) & (
                            date < (start_dt + ((day + 1) * timedelta))
                        ):
                            events = events + 1
            my_dates.append(start_dt + day * timedelta)
            my_events.append(events)
        plt.scatter(my_dates, my_events)
        plt.xlabel("time")
        plt.ylabel("number of new cluster events")
        plt.show()

    def plot_animation_time(self):
        """
        makes plots for animation.. it's nice for presentations
        """
        # initialise figure
        # I should change this function to be compatible with the others
        plt.figure()
        ax = plt.axes(xlim=(0, 360), ylim=(np.min(self.pos_dec_deg) - 10, 90))
        ax.plot([], [], "o", markersize=2)
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("DEC (deg)")
        # sort labels by date
        labels = self.dbscan_labels

        # get colors for plots
        unique_labels = set(labels)
        colors = [
            plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
        ]
        hours = (
            np.max(self.event_time) - np.min(self.event_time)
        ).total_seconds() / 86400
        frames = int(hours) + 5

        print(frames)
        start_time = np.min(self.event_time)
        print(start_time)
        for i in range(frames):
            time_delta = td(days=i)
            current_time = start_time + time_delta
            previous_hour = current_time - td(days=1)
            index = (self.event_time < current_time) & (self.event_time > previous_hour)
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                class_member_mask = labels == k
                ra_plot = self.pos_ra_deg[class_member_mask & index]
                dec_plot = self.pos_dec_deg[class_member_mask & index]
                plt.plot(
                    ra_plot,
                    dec_plot,
                    "o",
                    markerfacecolor=tuple(col),
                    markersize=1.5,
                    markeredgecolor=tuple(col),
                )
            ax.set_title("L2 events at " + current_time.strftime("%y_%m_%d_%H_%M_%S"))
            plt.savefig(current_time.strftime("%y_%m_%d_%H_%M_%S"))
        plt.close()

    def generate_event_no_list(self, output_log="cluster_events.txt"):
        """
        :param output_log:

        """
        labels = self.dbscan_labels
        unique_labels = set(labels)
        file = open(output_log, "a")
        event_list = []
        for i in unique_labels:
            if i != -1:
                index = labels == i
                new_event_no = self.event_number[index]
                for event in new_event_no:
                    file.write(str(int(event)) + "\n")
                    event_list.append(str(int(event)))
        file.close()
        np.save("cluster_events", event_list)

    def find_msp(
        self,
        filename,
        number_msp_plots,
        folder="msp",
        iml_plots_directory="adam_plots/",
    ):
        """
        #deprecating this in favour of straight database queries
        this plot finds all the MSPs that I'm looking for
        ############come back to fill this out
        :param filename:
        :param number_msp_plots:
        :param folder:  (Default value = 'msp')
        :param iml_plots_directory:  (Default value = 'adam_plots/')

        """
        name = []
        ra = []
        dec = []
        p0 = []
        dm = []
        with open(filename, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            for i, row in enumerate(reader):
                if i > 1:
                    row = np.array(row)
                    row[row == "*"] = "0"
                    print(row)
                    name.append(row[1])
                    ra.append(float(row[2]))
                    dec.append(float(row[3]))
                    p0.append(float(row[4]))
                    dm.append(float(row[5]))
        for i, pulsar_name in enumerate(name):
            print("on this pulsar:")
            print(pulsar_name)
            print("\n")
            if i < number_msp_plots:
                print("plotting")
                print(pulsar_name)
                print("\n")
                ra_bool = in_tolerance(self.pos_ra_deg, ra[i] - 5, ra[i] + 5)
                dec_bool = in_tolerance(self.pos_dec_deg, dec[i] - 5, dec[i] + 5)
                dm_bool = in_tolerance(self.dm, dm[i] - 5, dm[i] + 5)
                mask = ra_bool & dec_bool & dm_bool
                plot_ra = self.pos_ra_deg[mask]
                plot_dec = self.pos_dec_deg[mask]
                plot_dm = self.dm[mask]
                plot_ks = self.known_sources_name[mask]
                plot_ks[plot_ks == None] = "None"
                plot_event_no = self.event_number[mask]
                if len(plot_ks) > 0:
                    plt.figure()
                    for source in set(plot_ks):
                        ks_mask = plot_ks == source
                        plt.scatter(plot_ra[ks_mask], plot_dec[ks_mask], label=source)
                    if not os.path.exists(folder + "/" + pulsar_name):
                        os.mkdir(folder + "/" + pulsar_name)
                    plt.scatter(
                        ra[i], dec[i], marker="x", s=30, label="atnf " + pulsar_name
                    )
                    plt.title(pulsar_name + " coincidence")
                    plt.xlabel("RA deg")
                    plt.ylabel("DEC deg")
                    plt.legend()
                    plt.savefig(
                        folder + "/" + pulsar_name + "/" + pulsar_name + "_coincidence"
                    )
                    # would be good to save all the data into that folter too
                    plt.close()
                    np.savez(
                        folder + "/" + pulsar_name + "/" + "data",
                        new_source_time=self.event_time[mask],
                        new_event_no=self.event_number[mask],
                        new_source_ra=self.pos_ra_deg[mask],
                        new_source_dec=self.pos_dec_deg[mask],
                        new_source_dm=self.dm[mask],
                        new_source_dm_error=self.dm_error[mask],
                        new_source_snr=self.snr_arr[mask],
                        new_known_sources=self.known_sources_name[mask],
                    )
                intensity_ml_dir = os.listdir(iml_plots_directory)
                for file in intensity_ml_dir:
                    for my_event in plot_event_no:
                        if str(int(my_event)) in file:
                            os.system(
                                "cp "
                                + iml_plots_directory
                                + file
                                + " "
                                + folder
                                + "/"
                                + pulsar_name
                                + "/"
                                + file
                            )

    def intensity_list(
        self,
        log_file,
        iml_plots_directory,
        calibration_source="cal_source_positions.npz",
    ):
        """for the log file you can use either conbined score max or combined score average

        :param log_file: log file obtained from intesity ml
        :param iml_plots_directory: plots from intensity ml
        :param repeaters_folder:  folder where all my clusters are(Default value = 'repeaters/')
        :param label: are tehre any labels i want to look at specifically(Default value = -2)

        """
        if calibration_source:
            cal_source_positions = np.load(calibration_source, allow_pickle=1)[
                "cal_source_positions"
            ].item()
            sources = list(cal_source_positions.keys())
            # ra = cal_source_positions[source_name]['ra']
            # dec = cal_source_positions[source_name]['dec']
        with open(log_file) as csvfile:
            iml_event_no = []
            iml_score = []
            reader = csv.reader(csvfile, delimiter=" ")
            for row in reader:
                iml_event_no.append(int(row[0]))
                iml_score.append(float(row[1]))
        iml_score = np.array(iml_score)
        iml_event_no = np.array(iml_event_no)
        iml_event_no, unique_index = np.unique(iml_event_no, return_index=1)
        iml_score = iml_score[unique_index]
        # now remove all the rfi stuff
        astro = np.argwhere(iml_score == 10)
        iml_score = iml_score[astro]
        iml_event_no = iml_event_no[astro]

        matches_write = []
        for i, event in enumerate(self.event_number):
            if event in iml_event_no:
                arg_iml = np.argwhere(iml_event_no == event)
                score = np.squeeze(iml_score[arg_iml])
                ra = self.pos_ra_deg[i]
                dec = self.pos_dec_deg[i]
                dm = self.dm[i]
                ks = self.known_sources_name[i]
                snr = self.snr_arr[i]
                event_time = self.event_time[i]
                # find closes cal source
                closest_source = "None"
                dist = 999999999
                for cal in sources:

                    cal_ra = cal_source_positions[cal]["ra"]
                    cal_dec = cal_source_positions[cal]["dec"]
                    if np.abs(cal_ra - ra) > 180:
                        if cal_ra > ra:
                            d_ra = cal_ra - 360 - ra
                        elif ra > cal_ra:
                            d_ra = ra - 360 - cal_ra
                    else:
                        d_ra = cal_ra - ra
                    d_dec = cal_dec - dec
                    new_dist = np.sqrt(d_dec ** 2 + d_ra ** 2)
                    if new_dist < dist:
                        dist = new_dist
                        closest_source = cal

                write_array = [
                    str(int(event)),
                    str(score),
                    str(ra),
                    str(dec),
                    str(dm),
                    str(snr),
                    str(event_time),
                    str(ks),
                    str(closest_source),
                    str(dist),
                ]
                matches_write.append(write_array)
        matches_write = np.array(matches_write)

        with open("KS_Intensity_match.csv", "w+") as scores:
            # write the scores in and also event numbers
            writer = csv.writer(scores, delimiter=" ")
            writer.writerow(
                [
                    "eventno",
                    "score",
                    "ra",
                    "dec",
                    "dm",
                    "snr",
                    "event_time",
                    "Known Source",
                    "closest source",
                    "euclidean distance",
                ]
            )
            for item in matches_write:
                writer.writerow(item)
        intensity_ml_dir = os.listdir(iml_plots_directory)
        # now lets try find the real events
        for file in intensity_ml_dir:
            for my_event in matches_write:
                if str(int(my_event[0])) in file:
                    os.system(
                        "cp "
                        + iml_plots_directory
                        + file
                        + " "
                        + "KS_intensity/"
                        + file
                    )

    def match_archive(self, plots_directory, repeaters_folder="repeaters/", label=-2):
        """for the log file you can use either conbined score max or combined score average

        :param log_file: log file obtained from intesity ml
        :param plots_directory: plots from intensity ml
        :param repeaters_folder:  folder where all my clusters are(Default value = 'repeaters/')
        :param label: are tehre any labels i want to look at specifically(Default value = -2)

        """
        self.search_for_new_clusters(repeaters_folder)
        print("made my_new_sources.csv")
        # the log file originally used scores from intensity ml, now we just use the labelling on frb_archiver
        labels = self.dbscan_labels
        intensity_ml_dict = {}
        if label == -2:
            unique_labels = set(labels)
        else:
            unique_labels = label
        for i in unique_labels:
            index = labels == i
            my_known_sources = set(self.known_sources_name[index])
            new_event_no = self.event_number[index]
            new_source_ra = self.pos_ra_deg[index]
            new_source_dec = self.pos_dec_deg[index]
            new_source_dm = self.dm[index]
            new_source_snr = self.snr_arr[index]
            new_source_event_time = self.event_time[index]
            iml_matched_score = []
            iml_matched_event = []
            iml_matched_event_time = []
            iml_matched_ra = []
            iml_matched_dec = []
            iml_matched_dm = []
            iml_matched_snr = []
            # lets create folders with the cluster number and put all the plots inside it
            if not os.path.exists(repeaters_folder):
                os.mkdir(repeaters_folder)
            if not os.path.exists(repeaters_folder + "likely_astro"):
                os.mkdir(repeaters_folder + "likely_astro")
            if np.min(new_source_dec > -99):
                # everything negative in dec is super weird for chime, ignore
                # replace -99 with a reasonable number when we want to start ignoring negative decs again
                for my_event, my_ra, my_dec, my_dm, my_snr, my_time in zip(
                    new_event_no,
                    new_source_ra,
                    new_source_dec,
                    new_source_dm,
                    new_source_snr,
                    new_source_event_time,
                ):
                    iml_matched_event.append(int(my_event))
                    iml_matched_ra.append(my_ra)
                    iml_matched_event_time.append(my_time)
                    iml_matched_dec.append(my_dec)
                    iml_matched_dm.append(my_dm)
                    iml_matched_snr.append(my_snr)
                intensity_ml_dict.update(
                    {
                        i: np.array(
                            [
                                np.squeeze(iml_matched_event),
                                np.squeeze(iml_matched_ra),
                                np.squeeze(iml_matched_dec),
                                np.squeeze(iml_matched_dm),
                                np.squeeze(iml_matched_snr),
                                np.squeeze(iml_matched_event_time),
                            ]
                        )
                    }
                )
        # now lets try find the real events
        for i in intensity_ml_dict:
            # change the -1s to 0
            dec_range = intensity_ml_dict[i][2]
            ra_range = intensity_ml_dict[i][1]
            folder_name = (
                "C"
                + str(i)
                + "_R"
                + str(int(np.min(ra_range)))
                + "_"
                + str(int(np.max(ra_range)))
                + "_D"
                + str(int(np.min(dec_range)))
                + "_"
                + str(int(np.max(dec_range)))
            )
            intermediary_folder = "likely_astro/"
            if not os.path.exists(repeaters_folder + intermediary_folder + folder_name):
                os.mkdir(repeaters_folder + intermediary_folder + folder_name)
            output_directory = (
                repeaters_folder + intermediary_folder + folder_name + "/"
            )
            self.plot_specific_cluster(i, output_directory=output_directory)
            score_path = output_directory + "scores.txt"
            with open(score_path, "w+") as scores:
                # write the scores in and also event numbers
                writer = csv.writer(scores, delimiter=" ")
                write_array = np.column_stack(intensity_ml_dict[i])
                writer.writerow(["eventno", "ra", "dec", "dm", "snr", "event_time"])
                for item in write_array:
                    writer.writerow(item)
            # copy images over, this feature won't be used much, but can stay here.
            intensity_ml_dir = os.listdir(plots_directory)
            for file in intensity_ml_dir:
                for my_event in intensity_ml_dict[i][0]:
                    if str(int(my_event)) in file:
                        os.system(
                            "cp "
                            + plots_directory
                            + file
                            + " "
                            + repeaters_folder
                            + intermediary_folder
                            + folder_name
                            + "/"
                            + file
                        )
        self.intensity_ml_dict = intensity_ml_dict

    def load_known_repeaters(self, log_file="known_repeaters.csv"):
        """

        :param log_file:  (Default value = 'known_repeaters.csv')

        """
        with open(log_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            event_no = []
            known_repeater_prop = []
            name = []
            for i, row in enumerate(reader):
                if i > 0:
                    # print(row)
                    name.append(row[0])
                    event_no.append(int(row[1]))
                    known_repeater_prop.append(
                        [float(row[2]), float(row[3]), float(row[4])]
                    )
        return known_repeater_prop, name, event_no

    def check_repeaters(self, log_file="known_repeaters.csv"):
        """

        :param log_file:  (Default value = 'known_repeaters.csv')

        """
        # this will tell us how many of the known_repeaters are in the whole event database
        known_repeater_prop, name, event_no = self.load_known_repeaters(log_file)
        for i, num in enumerate(event_no):
            if num in self.event_number:
                print("found " + name[i] + ", " + str(num))
            else:
                print(name[i] + ", " + str(num) + " not found")

    def load_wiki(self, log_file):
        known_repeater_prop, name, event_no = self.load_known_repeaters(
            log_file=log_file
        )
        self.pos_ra_deg = np.array(list(prop[0] for prop in known_repeater_prop))
        self.pos_dec_deg = np.array(list(prop[1] for prop in known_repeater_prop))
        self.dm = np.array(list(prop[2] for prop in known_repeater_prop))
        self.event_number = event_no
        # this attribute is only available for loading wiki
        self.dbscan_labels = name

    def load_master(self, numpy_file):
        master = np.load(numpy_file, allow_pickle=1)
        self.pos_ra_deg = np.array(list(event["ra"] for event in master))
        self.pos_dec_deg = np.array(list(event["dec"] for event in master))
        self.dm = np.array(list(event["dm"] for event in master))
        self.event_number = np.array(list(event["event"] for event in master))
        """
        x=np.argwhere(self.event_number==26804247)
        print(master[x])
        sys.exit(2)
        """
        dt_arr = []
        for event in master:
            try:
                mydt = dt.strptime(event["datetime"], "%Y-%m-%d %H:%M:%S.%f UTC%z")
            except Exception as e:
                utc = pytz.utc
                mydt = utc.localize(
                    dt.strptime(event["datetime"], "%Y-%m-%d %H:%M:%S.%f")
                )

            # print(mydt)
            dt_arr.append(mydt)
        self.event_time = np.array(dt_arr)
        self.snr_arr = np.array(list(event["snr"] for event in master))
        self.dm_error = np.array(list(event["dm_error"] for event in master))

    def where_are_tns_frb(self):
        unique_labels = set(self.dbscan_labels)
        print(np.sum(self.event_number == -1))
        for i in unique_labels:
            index = (self.dbscan_labels == i) & (self.event_number == -1)

            if np.sum(index) > 0:
                print("tns frb found in cluster " + str(i))
                print(self.known_sources_name[index])

    def where_are_cs_frb(self):
        unique_labels = set(self.dbscan_labels)
        print(np.sum(self.known_sources_name == -1))
        for i in unique_labels:
            index = (self.dbscan_labels == i) & (self.known_sources_name == -1)

            if np.sum(index) > 0:
                print("cs frb found in cluster " + str(i))
                associated_events = self.event_number[index]
                if np.size(associated_events) > 1:
                    for event in associated_events:
                        print(int(event))
                else:
                    print(int(associated_events))

    def where_are_known_repeaters(self, tolerance=5, log_file="known_repeaters.csv"):
        """
        This function can be used to search for known pulsars

        :param log_file:  (Default value = 'known_repeaters.csv')
        :param frb_master:  (Default value = 'frb_master.txt')

        """
        rpname = []
        rpevent = []
        known_repeater_prop, name, event_no = self.load_known_repeaters(log_file)
        print(name)
        labels = self.dbscan_labels
        unique_labels = set(labels)
        found = False
        matched_labels = np.ones(len(unique_labels), dtype=bool)
        for name, known_repeaters, event_number in zip(
            name, known_repeater_prop, event_no
        ):
            print("\n")
            print(event_number)
            print(name)
            if event_number in self.event_number:
                print(name + " is in my event numbers")
            else:
                print(name + " is not in my event numbers")
            for j, i in enumerate(unique_labels):
                index = labels == i
                new_event_no = self.event_number[index]
                new_source_ra = self.pos_ra_deg[index]
                new_source_dec = self.pos_dec_deg[index]
                new_source_dm = self.dm[index]
                events_in_master = []
                events_not_in_master = []
                meandm = np.mean(new_source_dm)
                meanra = np.mean(new_source_ra)
                meandec = np.mean(new_source_dec)
                bool_ra = in_tolerance(
                    known_repeaters[0], meanra - tolerance, meanra + tolerance
                )
                bool_dec = in_tolerance(
                    known_repeaters[1], meandec - tolerance, meandec + tolerance
                )
                bool_dm = in_tolerance(
                    known_repeaters[2], meandm - tolerance, meandm + tolerance
                )
                if (event_number in new_event_no) & (i != -1):
                    my_index = np.argwhere(new_event_no == event_number)
                    print(
                        "ra:"
                        + str(new_source_ra[my_index].flatten()[0])
                        + " dec:"
                        + str(new_source_dec[my_index].flatten()[0]),
                        " dm:" + str(new_source_dm[my_index].flatten()[0]),
                    )
                    print(name + " belongs to group " + str(i))
                    print("event_no:" + str(event_number))
                    found = True
                    matched_labels[j] = False
                    break
                elif bool_ra & bool_dec & bool_dm:
                    print(name + " probably belongs to group " + str(i))
                    print("event_no:" + str(event_number))
                    found = True
                    matched_labels[j] = False

            if not found:
                print(name + " " + str(event_number) + " not found in clusters")
            found = False
        print(matched_labels)
        array = np.array(list(unique_labels))
        print(array[np.argwhere(matched_labels)])

    def plot_specific_cluster(self, cluster_number, output_directory=""):
        """
        :param cluster_number:
        :param output_directory:  (Default value = '')
        """
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        labels = self.dbscan_labels
        index = labels == cluster_number
        # if known sources only contain None, ie it is a new source
        new_source_time = self.event_time[index]
        new_event_no = self.event_number[index]
        new_source_ra = self.pos_ra_deg[index]
        new_source_dec = self.pos_dec_deg[index]
        new_source_dm = self.dm[index]
        new_source_snr = self.snr_arr[index]
        new_source_ne2001 = self.gal_dm[index][0]
        new_source_ymw16 = self.gal_dm[index][1]
        ambiguous = np.max(self.ambiguous[index])
        if np.max(new_source_ra) - np.min(new_source_ra) > 180:
            new_source_ra[new_source_ra > 180] = (
                new_source_ra[new_source_ra > 180] - 360
            )

        mean_ra = np.mean(new_source_ra)
        mean_dec = np.mean(new_source_dec)
        (
            mean_hour_ra,
            mean_minute_ra,
            mean_second_ra,
            mean_deg_dec,
            mean_minute_dec,
            mean_second_dec,
        ) = self.convert_to_hoursminsec(mean_ra, mean_dec)
        hhmmss_ra_str = (
            str(mean_hour_ra) + ":" + str(mean_minute_ra) + ":" + str(mean_second_ra)
        )
        ddmmss_dec_str = (
            str(mean_deg_dec) + ":" + str(mean_minute_dec) + ":" + str(mean_second_dec)
        )
        dm_str = str(np.mean(new_source_dm))
        my_save_array = [hhmmss_ra_str, ddmmss_dec_str, dm_str]
        # this plots the ra-dec-dm
        plt.figure()
        plt.title(
            "New Source "
            + str(cluster_number)
            + " mean RA="
            + hhmmss_ra_str
            + " mean Dec="
            + ddmmss_dec_str
        )
        plt.xlabel("RA (deg)")
        plt.ylabel("DEC (deg)")
        # make the largest point size 13
        point_sizes = (new_source_snr / np.max(new_source_snr)) * 40

        plt.scatter(
            new_source_ra, new_source_dec, c=new_source_dm, marker="o", s=point_sizes
        )

        xerr = np.max(2.2 / np.cos(np.deg2rad(new_source_dec)))
        Y_err = 0.5

        from matplotlib.patches import Ellipse

        ax = plt.gca()
        ax.add_patch(
            Ellipse(
                (np.mean(new_source_ra), np.mean(new_source_dec)),
                width=xerr,
                height=Y_err,
                facecolor="none",
                edgecolor="r",
                alpha=1,
            )
        )

        # plt.errorbar(new_source_ra,new_source_dec,elinewidth=0.1,xerr=xerr,yerr=0.5,fmt='none')
        bar = plt.colorbar()
        # the 100 is there to keep only 2 decimals
        mean_ne2001 = int(np.mean(new_source_ne2001) * 100)
        mean_ymw16 = int(np.mean(new_source_ymw16) * 100)
        bar.set_label(
            r"DM pc/cm$^3$, ne2001 "
            + str(mean_ne2001 / 100.0)
            + " ymw16 "
            + str(mean_ymw16 / 100.0)
        )
        if ambiguous:
            plt.savefig(
                output_directory + "new_amb_source_" + str(cluster_number),
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.savefig(
                output_directory + "new_source_" + str(cluster_number),
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

        # we should plot dm,ra,dec-time graph
        f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
        if ambiguous:
            ax1.set_title(
                "CHIME/FRB detections of source "
                + str(cluster_number)
                + " mean RA="
                + hhmmss_ra_str
                + " mean Dec="
                + ddmmss_dec_str
            )
        else:
            ax1.set_title(
                "CHIME/FRB detections of source "
                + str(cluster_number)
                + " mean RA="
                + hhmmss_ra_str
                + " mean Dec="
                + ddmmss_dec_str
            )
        # plt.subplot(3,1,1)
        ax1.set_ylabel("DM")
        point_sizes = (new_source_snr / np.max(new_source_snr)) * 20
        ax1.scatter(new_source_time, new_source_dm, marker="o", s=point_sizes)
        # plt.subplot(3,1,2)
        ax2.set_ylabel("Ra(deg)")
        point_sizes = (new_source_snr / np.max(new_source_snr)) * 20
        ax2.scatter(new_source_time, new_source_ra, marker="o", s=point_sizes)
        # plt.subplot(3,1,3)
        ax3.set_ylabel("Dec(deg)")
        ax3.set_xlabel("Time")
        ax3.set_xlim(
            [np.min(new_source_time) - td(days=5), np.max(new_source_time) + td(days=5)]
        )
        plt.xticks(rotation=90)
        point_sizes = (new_source_snr / np.max(new_source_snr)) * 20
        ax3.scatter(new_source_time, new_source_dec, marker="o", s=point_sizes)
        if ambiguous:
            plt.savefig(
                output_directory + "analysis_amb" + str(cluster_number),
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.savefig(
                output_directory + "analysis" + str(cluster_number),
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()
        # would be good to save all the data into that folter too
        np.savez(
            output_directory + "data",
            new_source_cluster_no=labels[index],
            new_source_time=self.event_time[index],
            new_event_no=self.event_number[index],
            new_source_ra=self.pos_ra_deg[index],
            new_source_dec=self.pos_dec_deg[index],
            new_source_dm=self.dm[index],
            new_source_dm_error=self.dm_error[index],
            new_source_snr=self.snr_arr[index],
            new_source_ne2001=self.gal_dm[index][0],
            new_source_ymw16=self.gal_dm[index][1],
            ambiguous=np.max(self.ambiguous[index]),
        )

    def write_new_cluster_csv(self, path, ra, dec, dm, cluster_no):
        """

        :param path:
        :param ra:
        :param dec:
        :param dm:
        :param cluster_no:

        """
        mean_ra = np.mean(ra)
        mean_dec = np.mean(dec)
        (
            mean_hour_ra,
            mean_minute_ra,
            mean_second_ra,
            mean_deg_dec,
            mean_minute_dec,
            mean_second_dec,
        ) = self.convert_to_hoursminsec(mean_ra, mean_dec)
        hhmmss_ra_str = (
            str(mean_hour_ra) + ":" + str(mean_minute_ra) + ":" + str(mean_second_ra)
        )
        ddmmss_dec_str = (
            str(mean_deg_dec) + ":" + str(mean_minute_dec) + ":" + str(mean_second_dec)
        )
        dm_str = str(np.mean(dm))
        my_save_array = [
            cluster_no,
            hhmmss_ra_str,
            ddmmss_dec_str,
            dm_str,
            mean_ra,
            mean_dec,
        ]
        with open(path + "/my_unique_sources.csv", mode="a") as file:
            print(path + "/my_unique_sources.csv")
            print(my_save_array)
            writer = csv.writer(file, delimiter=",")
            writer.writerow(my_save_array)

    def search_for_new_clusters(self, path):
        """
        writes all the clusters to csv file
        :param path: path to output csv file

        """
        labels = self.dbscan_labels

        unique_labels = set(labels)
        my_save_dict = {}
        for i in unique_labels:
            index = labels == i
            my_known_sources = set(self.known_sources_name[index])
            # if known sources only contain None, ie it is a new source
            # print(my_known_sources)
            new_event_no = self.event_number[index]
            new_source_ra = self.pos_ra_deg[index]
            # sometimes we can get wrappings at 0, we need to treat this in averaging...
            if max(new_source_ra) > min(new_source_ra) + 180:
                # subtract 360 from the value so it goes negative if it's too large
                large_ra = new_source_ra > 180
                new_source_ra[large_ra] = new_source_ra[large_ra] - 360
            new_source_dec = self.pos_dec_deg[index]
            new_source_dm = self.dm[index]
            new_source_snr = self.snr_arr[index]
            mean_ra = np.mean(new_source_ra)
            mean_dec = np.mean(new_source_dec)
            (
                mean_hour_ra,
                mean_minute_ra,
                mean_second_ra,
                mean_deg_dec,
                mean_minute_dec,
                mean_second_dec,
            ) = self.convert_to_hoursminsec(mean_ra, mean_dec)
            hhmmss_ra_str = (
                str(mean_hour_ra)
                + ":"
                + str(mean_minute_ra)
                + ":"
                + str(mean_second_ra)
            )
            ddmmss_dec_str = (
                str(mean_deg_dec)
                + ":"
                + str(mean_minute_dec)
                + ":"
                + str(mean_second_dec)
            )
            dm_str = str(np.mean(new_source_dm))
            my_save_array = [
                hhmmss_ra_str,
                ddmmss_dec_str,
                dm_str,
                mean_ra,
                mean_dec,
                str(len(new_event_no)),
            ]
            my_save_dict.update({i: my_save_array})
        # np.savez('my_new_sources',data=my_save_dict)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + "/my_new_sources.csv", mode="w") as file:
            writer = csv.writer(file, delimiter=",")
            for item in my_save_dict:
                writer.writerow(np.insert(my_save_dict[item], 0, item))

    def write_scores(self, folder=""):
        """

        :param folder:  (Default value = '')

        """
        labels = self.dbscan_labels
        for label_ind in set(labels):
            excess_dm = self.dm - self.gal_dm_ne2001
            mask = np.where(label_ind == np.array(labels))
            print(mask)
            with open(folder + "/" + str(label_ind) + ".csv", mode="w") as file:
                writer = csv.writer(file, delimiter=",")
                for item in mask[0]:
                    writer.writerow(
                        [
                            self.pos_ra_deg[item],
                            self.pos_dec_deg[item],
                            self.dm[item],
                            excess_dm[item],
                            self.event_time[item],
                            self.event_number[item],
                        ]
                    )

    def dm_time(self, folder=""):
        """

        :param folder:  (Default value = '')

        """
        labels = self.dbscan_labels
        print(np.size(labels))
        print(np.size(self.dm))
        for label_ind in set(labels):
            excess_dm = self.dm - self.gal_dm_ne2001
            mask = label_ind == labels
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            plt.title("cluster dm time " + str(label_ind))
            ax1.scatter(self.event_time[mask], self.dm[mask])
            ax2.scatter(self.event_time[mask], excess_dm[mask])
            ax1.set_ylabel("dm")
            ax2.set_ylabel("excess dm")
            plt.savefig(folder + "/" + "dm_time_cluster_" + str(label_ind))
            plt.close()

    def cluster_hist_dm(self, folder=""):
        """
        :param folder:  (Default value = '')
        """
        labels = self.dbscan_labels
        for label_ind in set(labels):
            excess_dm = self.dm - self.gal_dm_ne2001
            plt.figure()
            plt.hist(x=excess_dm[labels == label_ind], bins=5)
            plt.title("cluster dm hist ,5 bins for cluster " + str(label_ind))
            plt.savefig(folder + "/" + "hist cluster " + str(label_ind))
            plt.close()

    def plot_all_groups(self, folder=""):
        """
        :param folder:  (Default value = '')
        """
        labels = self.dbscan_labels
        for label_ind in set(labels):
            plt.figure()
            mask = labels == label_ind
            excess_dm = self.dm - self.gal_dm_ne2001
            plt.title("ra dec plot for cluster " + str(label_ind))
            plt.xlabel("ra deg")
            plt.ylabel("dec deg")
            plt.scatter(
                self.pos_ra_deg[mask],
                self.pos_dec_deg[mask],
                c=excess_dm[mask],
                marker=".",
                s=self.snr_arr[mask] * 2.3,
            )
            bar = plt.colorbar()
            bar.set_label("excess dm")
            plt.savefig(folder + "/" + "cluster excess" + str(label_ind))
            plt.close()
            plt.figure()
            mask = labels == label_ind
            excess_dm = self.dm - self.gal_dm_ne2001
            plt.title("ra dec plot for cluster " + str(label_ind))
            plt.xlabel("ra deg")
            plt.ylabel("dec deg")
            plt.scatter(
                self.pos_ra_deg[mask],
                self.pos_dec_deg[mask],
                c=self.dm[mask],
                marker=".",
                s=self.snr_arr[mask] * 2.3,
            )
            bar = plt.colorbar()
            bar.set_label("dm")
            plt.savefig(folder + "/" + "cluster " + str(label_ind))
            plt.close()

    def plot_results(self):
        """This function will plot clusters in DB scan, it will also plot all results for diagnostics"""
        labels = self.dbscan_labels
        again = True
        while again:
            try:
                plot_all = int(
                    input(
                        "plot_all? 0=no 1=plot with groups 2=plot with DM 3=plot group"
                    )
                )
                """plotting all data"""
                point_sizes = (self.snr_arr / np.max(self.snr_arr)) * 13
                if plot_all == 1:
                    unique_labels = set(labels)
                    colors = [
                        plt.cm.Spectral(each)
                        for each in np.linspace(0, 1, len(unique_labels))
                    ]
                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            # Black used for noise.
                            col = [0, 0, 0, 1]

                        class_member_mask = labels == k

                        ra_scale = self.pos_ra_deg[class_member_mask]
                        dec_scale = self.pos_dec_deg[class_member_mask]
                        point_sizes = (
                            self.snr_arr[class_member_mask]
                            / np.max(self.snr_arr[class_member_mask])
                        ) * 13
                        col_arr = col
                        plt.scatter(
                            ra_scale, dec_scale, marker="o", c=[col], s=point_sizes
                        )
                elif plot_all == 2:
                    plt.scatter(
                        self.pos_ra_deg, self.pos_dec_deg, c=self.dm, s=point_sizes
                    )
                    plt.colorbar()
                elif plot_all == 3:
                    label_ind = int(input("Please choose the group number"))
                    """plotting got a specific group"""
                    plt.scatter(
                        self.pos_ra_deg[labels == label_ind],
                        self.pos_dec_deg[labels == label_ind],
                        c=self.dm[labels == label_ind],
                        marker="o",
                        s=point_sizes[labels == label_ind],
                    )
                    print(set(self.known_sources_name[labels == label_ind]))
                    print("ra:")
                    print(self.pos_ra_deg[labels == label_ind])
                    print("dec:")
                    print(self.pos_dec_deg[labels == label_ind])
                    print("dm:")
                    print(self.dm[labels == label_ind])
                    plt.colorbar()

                plt.show()
                again = int(input("again? 0=no 1=yes"))
            except Exception as e:
                print(e)
                print(
                    "you need to enter an integer between 0 and " + str(np.max(labels))
                )
    def get_frbcat(self, file="frbcat_20191125.csv"):
        """

        :param file: catalogue file from http://frbcat.org(Default value = 'frbcat_20191125.csv')
        #this function is now deprecated as frb_cat has moved onto TNS
        """
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            counter = 0
            for row in csv_reader:
                if not (counter == 0):
                    ra = row[3]
                    dec = row[4]
                    my_ra, my_dec = self.convert_to_deg(ra, dec)
                    self.event_number = np.append(self.event_number, counter)
                    self.pos_ra_deg = np.append(self.pos_ra_deg, my_ra)
                    if float(row[9]) == 0:
                        print("changing")
                        print(row[9])
                        row[9] = 10
                        print(row[9])
                    self.snr_arr = np.append(self.snr_arr, float(row[9]))
                    timestr = row[1]
                    mytime = dt.strptime(timestr, "%Y/%m/%d %H:%M:%S.%f")
                    self.event_time = np.append(self.event_time, mytime)
                    dm = re.split("&plusmn", row[7])
                    my_dm = float(dm[0])
                    if len(dm) == 2:
                        my_dm_err = float(dm[1])
                    else:
                        my_dm_err = 0
                    self.dm_error = np.append(self.dm_error, my_dm_err)
                    self.dm = np.append(self.dm, my_dm)
                    self.pos_dec_deg = np.append(self.pos_dec_deg, my_dec)
                counter = counter + 1
