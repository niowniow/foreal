import io

import obspy
import pandas as pd


def get_obspy_stream(
    store,
    start_time,
    end_time,
    network,
    station,
    channels,
    location,
    pad=False,
    fill=None,
    old_stationname=False,
    extension="mseed",
):
    """
    Loads the microseismic data for the given timeframe into a miniseed file.

    Arguments:
        start_time {datetime} -- start timestamp
        p of the desired obspy stream
        end_time {datetime} -- end timestamp of the desired obspy stream

    Keyword Arguments:
        pad {bool} -- If padding is true, the data will be zero padded if the data is not consistent
        fill {} -- If numpy.nan or fill value: error in the seismic stream will be filled with the value. If None no fill will be used
        verbose {bool} -- If info should be printed

    Returns:
        obspy stream -- obspy stream with up to three channels
                        the stream's channels will be sorted alphabetically
    """

    if not isinstance(channels, list):
        channels = [channels]

    # We will get the full hours seismic data and trim it to the desired length afterwards
    tbeg_hours = pd.to_datetime(start_time).replace(minute=0, second=0, microsecond=0)
    timerange = pd.date_range(
        start=tbeg_hours - pd.to_timedelta("1 H"), end=end_time, freq="H"
    )

    non_existing_files_ts = []  # keep track of nonexisting files

    stream = obspy.Stream()

    idx = 0
    # loop through all hours
    for i in range(len(timerange)):
        # start = time.time()
        h = timerange[i]

        st_list = obspy.Stream()

        datayear = timerange[i].strftime("%Y")
        if old_stationname:
            station = (
                "MHDL" if station == "MH36" else "MHDT"
            )  # TODO: do not hardcode it
        filenames = {}
        for channel in channels:
            filenames[channel] = [
                station,
                datayear,
                "%s.D" % channel,
                "%s.%s.%s.%s.D." % (network, station, location, channel)
                + timerange[i].strftime("%Y%m%d_%H%M%S")
                + "."
                + extension,
            ]

            # Load either from store or from filename
            if store is not None:
                # get the file relative to the store
                filename = "/".join(filenames[channel])
                try:
                    st = obspy.read(io.BytesIO(store[str(filename)]))
                except Exception as ex:
                    # print(f"SEISMIC SOURCE Exception: {e}")#pass
                    st = obspy.Stream()
            else:
                raise NotImplementedError("Please provide a store")
            st_list += st

        stream_h = st_list.merge(method=0, fill_value=fill)
        segment_h = stream_h

        stream += segment_h

    if not stream:
        raise RuntimeError(
            f'files not found {["/".join(filenames[fn]) for fn in filenames]}'
        )

    stream = stream.merge(method=0, fill_value=fill)

    stream = stream.trim(
        starttime=obspy.UTCDateTime(start_time),
        pad=pad,
        fill_value=fill,
        nearest_sample=True,
    )
    stream = stream.trim(
        endtime=obspy.UTCDateTime(end_time),
        pad=pad,
        fill_value=fill,
        nearest_sample=False,
    )

    stream.sort(
        keys=["channel"]
    )  # TODO: change this so that the order of the input channels list is maintained

    return stream
