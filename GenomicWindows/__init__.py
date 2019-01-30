from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from math import log
from functools import wraps
from operator import itemgetter
from bisect import bisect
import sys
import pandas as pd
from collections import OrderedDict, defaultdict

import unittest, itertools

# interval tests

# single pos intervals
#|#|#|#|#|#|#
# end to end
###|###|###
# first starts aat start and last ends at last (3 intervals)
###  ###  ###
# first starts after start and last ends before last (3 intervals)
 ### ### ###
# Three intervals tiled
 ######
  ######
   ######
# Three intevals inside each other 
 #######
  #####
   ###


# in each case test:
# 

def print_intervals(df):
    starts, ends = list(df['start']), list(df['end'])
    m = max(ends)
    for s, e in zip(starts, ends):
        print(" " * (s) + "#" * (e-s) + " " * (m - e))

class TestEvenWindows(unittest.TestCase):

    def setUp(self):

        self.df_single_end_to_end = pd.DataFrame({'start': list(range(30)), 'end': list(range(1, 31))})

        self.df_three_end_to_end = pd.DataFrame({'start': [0, 10, 20], 'end': [10, 20, 30]})

        self.df_three_nonoverl = pd.DataFrame({'start': [5, 15, 25], 'end': [10, 20, 30]})

        self.df_three_tiled = pd.DataFrame({'start': [0, 3, 6], 'end': [21, 24, 27]})
        # print_intervals(self.df_three_tiled)

        ########## ####### ####
           ####### ####### #######
              #### ####### ##########

        self.df_three_nested = pd.DataFrame({'start': [0, 3, 6], 'end': [27, 24, 21]})
        # print_intervals(self.df_three_nested)

        ###########################
           #####################
              ###############

        ########## ####### ##########
           ####### ####### #######
              #### ####### ####

        self.df_empty = pd.DataFrame()

    def test_empty(self):
        with self.assertRaises(KeyError):
            even_windows(self.df_empty, 3)


    def test_single_end_to_end_1(self):
        # three perfect windows
        self.assertEqual(even_windows(self.df_single_end_to_end, 10), [10, 10, 10])

    def test_single_end_to_end_2(self):
        # one perfect window
        self.assertEqual(even_windows(self.df_single_end_to_end, 30), [30])

    def test_single_end_to_end_3(self):
        # one window too large
        self.assertEqual(even_windows(self.df_single_end_to_end, 31), [30])

    def test_single_end_to_end_4(self):
        # three windows, last one not filled
        self.assertEqual(even_windows(self.df_single_end_to_end, 12), [12, 12, 6])


    def test_three_end_to_end_1(self):
        # three perfect windows
        self.assertEqual(even_windows(self.df_three_end_to_end, 10), [10, 10, 10])

    def test_three_end_to_end_2(self):
        # one perfect window
        self.assertEqual(even_windows(self.df_three_end_to_end, 30), [30])

    def test_three_end_to_end_3(self):
        # one window too large
        self.assertEqual(even_windows(self.df_three_end_to_end, 31), [30])

    def test_three_end_to_end_4(self):
        # last window one not filled
        self.assertEqual(even_windows(self.df_three_end_to_end, 12), [12, 12, 6])

    def test_three_nonoverl_1(self):
        # three perfect windows
        self.assertEqual(even_windows(self.df_three_nonoverl, 5), [10, 10, 10])

    def test_three_nonoverl_2(self):
        # one perfect window
        self.assertEqual(even_windows(self.df_three_nonoverl, 15), [30])

    def test_three_nonoverl_3(self):
        # one window too large
        self.assertEqual(even_windows(self.df_three_nonoverl, 16), [30])

    def test_three_nonoverl_4(self):
        # last window not filled
        self.assertEqual(even_windows(self.df_three_nonoverl, 10), [20, 10])

    def test_three_tiled_1(self):
        # three perfect windows
        self.assertEqual(even_windows(self.df_three_tiled, 21), [10, 7, 10])

    def test_three_tiled_2(self):
        # one perfect window
        self.assertEqual(even_windows(self.df_three_tiled, 63), [27])

    def test_three_tiled_3(self):
        # one window too large
        self.assertEqual(even_windows(self.df_three_tiled, 64), [27])

    def test_three_tiled_4(self):
        # last window not filled
        self.assertEqual(even_windows(self.df_three_tiled, 60), [24, 3])

    def test_three_nested_1(self):
        # three perfect windows
        self.assertEqual(even_windows(self.df_three_nested, 21), [10, 7, 10])

    def test_three_nested_2(self):
        # one perfect window
        self.assertEqual(even_windows(self.df_three_nested, 63), [27])

    def test_three_nested_3(self):
        # one window too large
        self.assertEqual(even_windows(self.df_three_nested, 64), [27])

    def test_three_nested_4(self):
        # last window not filled
        self.assertEqual(even_windows(self.df_three_nested, 60), [24, 3])


class TestWindowCoordinates(unittest.TestCase):

    def test_window_coord_1(self):
        w = iter(WindowCoordinates(binsize=10, logbase=1, bins=None))
        self.assertEqual(list(itertools.islice(w, 4)), [(0, 10), (10, 10), (20, 10), (30, 10)])

    def test_window_coord_2(self):
        w = iter(WindowCoordinates(binsize=2, logbase=2, bins=None))
        self.assertEqual(list(itertools.islice(w, 4)), [(0, 2), (2, 4), (6, 8), (14, 16)])

    def test_window_coord_3(self):
        w = iter(WindowCoordinates(binsize=None, logbase=1, bins=[10, 20, 30]))
        self.assertEqual(list(itertools.islice(w, 4)), [(0, 10), (10, 20), (30, 30), (60, float('inf'))])


class TestConcatDicts(unittest.TestCase):
    pass

class TestStatsToDataFrame(unittest.TestCase):
    pass

class TestGenomicWindows(unittest.TestCase):
    pass

class TestStoreGroupbyApply(unittest.TestCase):
    pass

# concat_dicts(l):
# stats_data_frame(list_of_stat_results):
# genomic_windows(full_df, func, bin_iter):
# window(size=None, logbase=1, even=None):
# store_groupby_apply(store_file_name, col_names, fun, df_name='df', group_keys=True):


class TestWindowDecorator(unittest.TestCase):

    def setUp(self):

        self.df_single_end_to_end = pd.DataFrame({'start': list(range(30)), 'end': list(range(1, 31))})

        self.df_three_end_to_end = pd.DataFrame({'start': [0, 10, 20], 'end': [10, 20, 30]})

        self.df_three_nonoverl = pd.DataFrame({'start': [5, 15, 25], 'end': [10, 20, 30]})

        self.df_three_tiled = pd.DataFrame({'start': [0, 3, 6], 'end': [21, 24, 27]})

        self.df_three_nested = pd.DataFrame({'start': [0, 3, 6], 'end': [27, 24, 21]})



    # def test_(self):

    #     df = self.df_single_end_to_end.assign(chrom = 'X', val = 7)

    #     @window(size=10)
    #     def stat(df):
    #         return np.mean(df['val'])

    #     result_df = pd.DataFrame({'chrom':   , 'start':    , 'end':     , 'stat':      })
    #     self.assertEqual(f(df), result_df)




    # # call this function windows of size 5
    # @window(size=5)
    # def count1(df):
    #     return len(df.index)

    # print(full_df.groupby(['chrom', 'bar']).apply(count1))
    # #print(full_df.groupby(['chrom', 'bar']).apply(count1).reset_index())

    # print(full_df.groupby('chrom').apply(count1))#.reset_index())

    # # call this function on windows beginning at size 2 increasing by log 2
    # @window(size=2, logbase=2)
    # def count2(df):
    #     return len(df.index)

    # print(full_df.groupby('chrom').apply(count2))#.reset_index(drop=True))

    # # call this function on windows with ~10 observations in each
    # @window(even=10)
    # def count3(df):
    #     return {'count': len(df.index), 'sum': sum(df.end-df.start)}

    # print(full_df.groupby('chrom').apply(count3))#.reset_index(drop=True))

    # # call this function on windows with ~10 observations in each
    # @window(even=10)
    # def stats_fun(df):
    #     sr = df[['foo','bar']].sum()
    #     return sr.to_dict()


def even_windows(df, nrobs):

    intervals = sorted((x.start, x.end) for x in df[['start', 'end']].itertuples())

    bins = list()

    queue = list()
    total = 0
    i = 0 # interval index
    pos = 0 # sequence index
    prev_bin_end = 0

    #intervals_end = intervals[-1][1]
    intervals_end = max(df['end'])
    while pos < intervals_end:

        # get any new intervals
        while i < len(intervals) and pos == intervals[i][0]:
            assert intervals[i][0] == int(intervals[i][0]), 'only ints please'
            # put the end in a sorted queue
            queue.insert(bisect(queue, intervals[i][1]), intervals[i][1]) 
            i += 1

        # remove intervals no longer overlapping:
        while queue and queue[0] <= pos:
            queue.pop(0)

        # update running total
        total += len(queue)

        if total >= nrobs:
            binsize = pos + 1 - prev_bin_end
            bins.append(binsize)
            prev_bin_end = pos + 1
            total = 0

        pos += 1

    binsize = pos - prev_bin_end
    if binsize:
        bins.append(binsize)

    return bins


class WindowCoordinates(object):
    def __init__(self, binsize=None, logbase=1, bins=None):

        self.bin_size = binsize
        self.log_base = logbase
        self.bin_list = bins
        if self.bin_list is not None:
            assert logbase == 1 and not binsize, "Don't use bins with binsize or logbase"
            self.bin_list = bins[:]
            self.bin_size = self.bin_list.pop(0)

    def __iter__(self):
        self.bin_start = 0
        self.exhausted = False
        return self

    def next(self):
        next_bin = self.bin_start, self.bin_size
        if self.bin_list is not None:
            self.bin_start += self.bin_size
            if self.bin_list:
                self.bin_size = self.bin_list.pop(0)
            else:
                self.bin_size = float('inf')
        elif self.log_base == 1:# or self.bin_start == 0:
            self.bin_start += self.bin_size

        else:
            prev_bin_size = self.bin_size
            self.bin_size = self.log_base**(log(self.bin_size, self.log_base)+1)
            self.bin_start += prev_bin_size
        return next_bin

    __next__ = next # Python 3.X compatibility


def concat_dicts(l):
    d = dict()
    pairs = [b for a in zip(*[x.items() for x in l]) for b in a]
    for k, v in pairs:
        d.setdefault(k, []).append(v)
    return d


# def window(size=None, logbase=1, fixed=None, even=None):
#     def window_decorator(func):
#         @wraps(func)
#         def func_wrapper(full_df):

#             assert not(fixed and even), "only fixed or even bins - not both"
#             if even is None:
#                 get_bin = iter(WindowCoordinates(binsize=size, logbase=logbase, 
#                                                  bins=fixed))
#             else:
#                 get_bin = iter(WindowCoordinates(binsize=size, logbase=logbase, 
#                                                  bins=even_windows(full_df, even)))

#             bin_start, bin_size = get_bin.next()

#             buf = list()
#             list_of_stat_results = list()

#             def process(buf):
#                 df = pd.DataFrame(buf)
#                 df.loc[df.start < bin_start, 'start'] = bin_start
#                 df.loc[df.end > bin_start + bin_size, 'end'] = bin_start + bin_size
#                 list_of_stat_results.append(([bin_start, bin_start + bin_size], func(df)))

#             for row_sr in full_df.itertuples():
#                 while row_sr.start >= bin_start + bin_size:
#                     if buf:
#                         process(buf)
#                     bin_start, bin_size = get_bin.next()
#                     buf = [x for x in buf if x.end > bin_start]
#                 buf.append(row_sr)

#             # empty buffer
#             while buf:
#                 process(buf)
#                 bin_start, bin_size = get_bin.next()
#                 buf = [x for x in buf if x.end > bin_start]

#             coordinates, stats = zip(*list_of_stat_results)
#             if type(stats[0]) is dict:
#                 d = OrderedDict(zip(('start', 'end'), zip(*coordinates)))
#                 d.update(_concat_dicts(stats))
#                 return pd.DataFrame(d)#.drop([0], axis=1)

#             else:
#                 return pd.DataFrame([x + [y] for x, y in list_of_stat_results],
#                                     columns=['start', 'end', func.__name__])

#         return func_wrapper
#     return window_decorator


def stats_data_frame(list_of_stat_results, func):

    coordinates, stats = zip(*list_of_stat_results)
    if type(stats[0]) is dict:
        d = OrderedDict(zip(('start', 'end'), zip(*coordinates)))
        d.update(concat_dicts(stats))
        df = pd.DataFrame(d)
    else:
        df = pd.DataFrame([x + [y] for x, y in list_of_stat_results],
                            columns=['start', 'end', func.__name__])
    return df.reset_index(level=[x for x in df.index.names if x is not None])
        


def genomic_windows(full_df, func, bin_iter, fill=True):

    def process(buf):
        try:
            df = pd.DataFrame(buf)
            df.loc[df.start < bin_start, 'start'] = bin_start
            df.loc[df.end > bin_start + bin_size, 'end'] = bin_start + bin_size
        except AttributeError:
            list_of_stat_results.append(([bin_start, bin_start + bin_size], fill))
        else:
            list_of_stat_results.append(([bin_start, bin_start + bin_size], func(df)))


    list_of_stat_results = list()
    
    bin_start, bin_size = bin_iter.next()
    buf = list()
    for row_sr in full_df.itertuples():
        while row_sr.start >= bin_start + bin_size:
            if buf or fill is not None:
                process(buf)
            bin_start, bin_size = bin_iter.next()
            buf = [x for x in buf if x.end > bin_start]
        buf.append(row_sr)

    # empty buffer
    while buf:
        process(buf)
        bin_start, bin_size = bin_iter.next()
        buf = [x for x in buf if x.end > bin_start]

    return stats_data_frame(list_of_stat_results, func)


def get_bin_iterator(full_df, binsize, logbase, even):

    if even is not None:
        fixed_bins = even_windows(full_df, even)
    else:
        fixed_bins = None
    bin_iter = iter(WindowCoordinates(binsize=binsize, logbase=logbase, bins=fixed_bins))

    return bin_iter


def window(size=None, logbase=1, even=None, fill=None):
    def window_decorator(func):
        @wraps(func)
        def func_wrapper(full_df):
            bin_iter = get_bin_iterator(full_df, size, logbase, even)

            return genomic_windows(full_df, func, bin_iter, fill=fill)

        return func_wrapper
    return window_decorator


def store_groupby_apply(store_file_name, col_names, fun, df_name='df', group_keys=True):

    if type(col_names) is str:
        col_names = [col_names]

    with pd.get_store(store_file_name) as store:
        groups = store.select(df_name, columns=col_names).drop_duplicates()
        df_list = []
        for tup in groups.itertuples():
            mask = ["{}={}".format(col, getattr(tup, col)) for col in col_names]
            grp_df = store.select(df_name, where = mask)
            stats_df = fun(grp_df)
            if group_keys:
                stats_df = (stats_df
                            .assign(**dict((col, getattr(tup, col)) for col in col_names))
                            .set_index(col_names)
                            )
            df_list.append(stats_df)

    return pd.concat(df_list)


if __name__ == "__main__":

    #unittest.main()


    full_df = pd.DataFrame({'chrom': ['chr1']+['chr2']*10,
                        'start': list(range(11)),
                        'end': list(map(sum, zip(range(11), [5, 1]*5+[20]))),
                        'value': 'AAA',
                       'foo': 7, 'bar': 9, 'baz' : 4})
    print(full_df)

    # call this function windows of size 5
    @window(size=5)
    def count1(df):
        return len(df.index)

    print(full_df.groupby(['chrom', 'bar']).apply(count1))
    #print(full_df.groupby(['chrom', 'bar']).apply(count1).reset_index())

    print(full_df.groupby('chrom').apply(count1))#.reset_index())

    # call this function on windows beginning at size 2 increasing by log 2
    @window(size=2, logbase=2)
    def count2(df):
        return len(df.index)

    print(full_df.groupby('chrom').apply(count2))#.reset_index(drop=True))

    # call this function on windows with ~10 observations in each
    @window(even=10)
    def count3(df):
        return {'count': len(df.index), 'sum': sum(df.end-df.start)}

    print(full_df.groupby('chrom').apply(count3))#.reset_index(drop=True))

    # call this function on windows with ~10 observations in each
    @window(even=10)
    def stats_fun(df):
        sr = df[['foo','bar']].sum()
        return sr.to_dict()

    df = full_df.groupby(['chrom', 'baz']).apply(stats_fun)
    print('##')
    print(df)
    print(df.index)

    # # write the data frame to a hdf5 store
    # def write_df_store(df, store_file_name, df_name='df', table=True, append=False):
    #     with pd.get_store(store_file_name) as store:
    #         store.append(df_name, df, data_columns=True, #list(df.columns.values), 
    #             table=table, append=append)

#    write_df_store(full_df2, 'groupby.h5')

    # write the data frame to a hdf5 store
    with pd.get_store('groupby.h5') as store:
        store.append('df', full_df, data_columns=['chrom', 'baz'], table=True, append=False)

    # perform the same groupby and apply operation as on the data frame
    df = store_groupby_apply('groupby.h5', ['chrom', 'baz'], stats_fun)
    print(df)
    print(df.index)
    #print(df.reset_index())

    # TODO: 
    # make sure exceptions fro the stats functions are raised properly
    # Also dangerous that AttributeError in the stats function silently makes it fill with the fill stats

    # the stats function should also be allowed to return a dataframe instead of a dict

    # next question: What happens when we group by the index (can we do that?) - and if so: should
    # we then still add the "groupby" columns to the resulting dataframes?
    

    # also next: I should use a by_chrom decorator like in GenomicIntervals to make windows handle separate choromosomes. 
    # That would still allow me to group by chormosomes (then there will just be one chrom in each df) if need be.

