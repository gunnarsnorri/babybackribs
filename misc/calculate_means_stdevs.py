import pandas

dataframe = pandas.read_csv("/mnt/nvme/traffic/Annotations.txt", delimiter=";",
                            header=None)

means = dataframe.mean()
stdevs = dataframe.std()
