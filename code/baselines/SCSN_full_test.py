from obspy import read

st = read('/home/bizon/Desktop/worker5/Depth_detect/southernCalifornia/full_dataset/37844807.ms')

print(st)

for tr in st:
    print(tr.stats)