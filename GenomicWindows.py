

from math import log
from functools import wraps
from operator import itemgetter
from bisect import bisect
import sys
import pandas as pd
from collections import OrderedDict, defaultdict


def even_windows(df, nrobs):

	intervals = sorted((x.start, x.end) for x in full_df[['start', 'end']].itertuples())
#    intervals = list(df[['start', 'end']].itertuples())
	bins = list()

	queue = list()
	total = 0
	i = 0 # interval index
	pos = 0 # sequence index
	prev_bin_end = 0

	intervals_end = intervals[-1][1]
	while pos < intervals_end:

		# get any new intervals
		while i < len(intervals) and pos == intervals[i][0]:
			assert intervals[i][0] == int(intervals[i][0]), 'only ints please'
			queue.insert(bisect(queue, intervals[i][1]), intervals[i][1]) # put the end in a sorted queue
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
	bins.append(binsize)

	return bins


class Bin(object):
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
		elif self.log_base == 1 or self.bin_start == 0:
			self.bin_start += self.bin_size

		else:
			prev_bin_size = self.bin_size
			self.bin_size = self.log_base**(log(self.bin_size, self.log_base)+1)
			self.bin_start += prev_bin_size
		return next_bin


def windows(size=None, logbase=1, fixed=None, even=None):
	def window_decorator(func):
		@wraps(func)
		def func_wrapper(full_df):

			assert not(fixed and even), "only fixed or even bins - not both"
			if even is None:
				get_bin = iter(Bin(binsize=size, logbase=logbase, bins=fixed))
			else:
				get_bin = iter(Bin(binsize=size, logbase=logbase, bins=even_windows(full_df, even)))    
			
			bin_start, bin_size = get_bin.next()
			
			buf = list()
			prev_chrom = None
			list_of_stat_results = list()

			def process(buf):
				df = pd.DataFrame(buf)
				df.loc[df.start < bin_start, 'start'] = bin_start
				df.loc[df.end > bin_start + bin_size, 'end'] = bin_start + bin_size
				list_of_stat_results.append(([prev_chrom, bin_start, bin_start + bin_size], func(df)))

			for row_sr in full_df.itertuples():

				while row_sr.start >= bin_start + bin_size:
					process(buf)
					bin_start, bin_size = get_bin.next()
					buf = [x for x in buf if x.end > bin_start]

				buf.append(row_sr)
				prev_chrom = row_sr.chrom

			# empty buffer
			while buf:
				process(buf)
				bin_start, bin_size = get_bin.next()
				buf = [x for x in buf if x.end > bin_start]

			# format output
			def concat_dicts(l):
				d = dict()
				pairs = [b for a in zip(*[x.items() for x in l]) for b in a]
				for k, v in pairs:
					d.setdefault(k, []).append(v)
				return d                

			coordinates, stats = zip(*list_of_stat_results)
			if type(stats[0]) is dict:
				d = OrderedDict(zip(('chrom', 'start', 'end'), zip(*coordinates)))
				d.update(concat_dicts(stats))
				return pd.DataFrame(d)

			else:
				return pd.DataFrame([x + [y] for x, y in list_of_stat_results],
									columns=['chrom', 'start', 'end', func.__name__])
		
		return func_wrapper
	return window_decorator


def store_groupby_apply(store_file_name, col_name, fun, df_name='df'):

	with pd.get_store(store_file_name) as store:
		groups = store.select_column(df_name, col_name).unique()
		df_list = []
		for g in groups:
			grp_df = store.select(df_name, where = ['{}={}'.format(col_name, g)])
			stats_df = fun(grp_df)
			df_list.append(stats_df)

	return pd.concat(df_list)


if __name__ == "__main__":

	full_df = pd.DataFrame({'chrom': ['chr1']+['chr2']*10, 
						'start': list(range(11)), 
						'end': map(sum, zip(range(11), [5, 1]*5+[20])),
						'value': 'AAA',
					   'foo': 7, 'bar': 9})
	print full_df

	# call this function windows of size 5
	@windows(size=5)
	def count1(df):
		return len(df.index)

	print full_df.groupby('chrom').apply(count1).reset_index(drop=True)

	# call this function on windows beginning at size 2 increasing by log 2
	@windows(size=2, logbase=2)
	def count2(df):
		return len(df.index)

	print full_df.groupby('chrom').apply(count2).reset_index(drop=True)

	# call this function on windows with ~10 observations in each
	@windows(even=10)
	def count3(df):
		return {'count': len(df.index), 'sum': sum(df.end-df.start)}

	print full_df.groupby('chrom').apply(count3).reset_index(drop=True)

	# call this function on windows with ~10 observations in each
	@windows(even=10)
	def stats_fun(df):
		sr = df[['foo','bar']].sum()
		return sr.to_dict()

	print full_df.groupby('chrom').apply(stats_fun).reset_index(drop=True)

	# write the data frame to a hdf5 store
	def write_df_store(df, store_file_name, df_name='df', table=True, append=False):
		with pd.get_store(store_file_name) as store:
			store.append(df_name, df, data_columns=list(df.columns.values), table=table, append=append)

	write_df_store(full_df, 'groupby.h5')

	# perform the same groupby and apply operation as on the data frame
	print store_groupby_apply('groupby.h5', 'chrom', stats_fun).reset_index(drop=True)



