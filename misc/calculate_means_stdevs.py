#!/usr/bin/env python
import pandas

dataframe = pandas.read_csv("/mnt/nvme/traffic/Annotations.txt", delimiter=";",
                            header=None)

xsize = 1280.
ysize = 960.
means = dataframe.mean()
scaled_means = []
scaled_means.append(means[1]/xsize)
scaled_means.append(means[2]/ysize)
scaled_means.append(means[3]/xsize)
scaled_means.append(means[4]/ysize)
stdevs = dataframe.std()
scaled_stdevs = []
scaled_stdevs.append(stdevs[1]/xsize)
scaled_stdevs.append(stdevs[2]/ysize)
scaled_stdevs.append(stdevs[3]/xsize)
scaled_stdevs.append(stdevs[4]/ysize)
print("means:\n%s" % scaled_means)
print("stdevs:\n%s" % scaled_stdevs)
