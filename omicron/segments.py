# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of LIGO-Omicron.
#
# LIGO-Omicron is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LIGO-Omicron is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LIGO-Omicron.  If not, see <http://www.gnu.org/licenses/>.

"""Segment utilities for Omicron
"""

from __future__ import print_function

import os.path
from functools import wraps

from glue.segmentsUtils import fromsegwizard
from glue.segments import (segmentlist as SegmentList, segment as Segment)

from . import (const, data)

STATE_CHANNEL = {
    'H1:DMT-UP:1': 'H1:GRD-ISC_LOCK_OK',
    'L1:DMT-UP:1': 'L1:GRD-ISC_LOCK_OK',
    'H1:DMT-CALIBRATED:1': 'H1:GDS-CALIB_STATE_VECTOR',
    'L1:DMT-CALIBRATED:1': 'L1:GDS-CALIB_STATE_VECTOR',
}


def integer_segments(f):
    @wraps(f)
    def decorated_method(*args, **kwargs):
        segs = f(*args, **kwargs)
        return type(segs)(type(s)(int(s[0]), int(s[1])) for s in segs)
    return decorated_method


def read_segments(filename, coltype=int):
   with open(filename, 'r') as fp:
        return fromsegwizard(fp, coltype=coltype)


def get_last_run_segment(segfile):
    return read_segments(segfile, coltype=int)[-1]


def write_segments(segmentlist, outfile, coltype=int):
    with open(outfile, 'w') as fp:
       for seg in segmentlist:
           print('%d %d' % seg, file=fp)


@integer_segments
def get_state_segments(channel, frametype, start, end, bits=[0], nproc=1):
    from gwpy.timeseries import StateVector
    obs = channel[0]
    cache = data.find_frames(obs, frametype, start, end)
    bits = map(str, bits)
    sv = StateVector.read(cache, channel, nproc=nproc, start=start, end=end,
                          bits=bits, gap='pad', pad=0).astype('uint32')
    segs = sv.to_dqflags().intersection()
    return segs.active


@integer_segments
def get_frame_segments(obs, frametype, start, end):
    cache = data.find_frames(obs, frametype, start, end)
    span = SegmentList([Segment(start, end)])
    return cache_segments(cache) & span


@integer_segments
def cache_segments(cache, span=None):
    segmentlist = SegmentList(e.segment for e in
                              cache.checkfilesexist(on_missing='warn')[0])
    return segmentlist.coalesce()


def segmentlist_from_tree(tree, coalesce=False):
    """Read a `~glue.segments.segmentlist` from a 'segments' `ROOT.Tree`
    """
    segs = SegmentList()
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        segs.append(Segment(tree.start, tree.end))
    return segs


@integer_segments
def omicron_output_segments(start, end, chunk, segment, overlap):
    """Work out the segments written to disk by an Omicron instance

    The Omicron process writes multiple files per process depending on
    the chunk duration, segment duration, and overlap, this method works out
    the segments that will be represented by those files.

    Parameters
    ----------
    start : `int`
        the start GPS time of the Omicron job
    end : `int`
        the end GPS time of the Omicron job
    chunk : `int`
        the CHUNKDURATION parameter for this configuration
    segment : `int`
        the SEGMENTDURATION parameter for this configuration
    overlap : `int`
        the OVERLAPDURATION parameter for this configuration

    Returns
    -------
    filesegs : `~glue.segments.segmentlist`
        a `~glue.segments.segmentlist`, one `~glue.segments.segment` for each
        file that _should_ be written by Omicron
    """
    padding = overlap / 2.
    out = SegmentList()
    fstart = start + padding
    fend = end - padding
    fdur = chunk - overlap
    fseg = segment - overlap
    t = fstart
    while t < fend:
        e = min(t + fdur, fend)
        seg = Segment(t, e)
        # if shorter than a chunk, but longer than a segment, omicron will
        # write files of length 'segment' until exhausted
        if fseg < abs(seg) < fdur:
            while t < e:
                seg = Segment(t, min(t + fseg, fend))
                out.append(seg)
                t += fseg
        else:
            out.append(seg)
        t = e
    return out