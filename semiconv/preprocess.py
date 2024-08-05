import argparse
import pandas as pd
from shapely import wkb
import binascii
import numpy as np

def create_overlapping_slices(array, slice_size, overlap):
    slices = []
    start = 0
    while start + slice_size <= len(array):
        end = start + slice_size
        slices.append(array[start:end])
        start = end - overlap  
    return np.array(slices)

def main(args):
    SIZE = 32  # size of the grid

    # Load location data
    locationData = pd.read_csv(args.meta_point_path)
    locationData = locationData[locationData['source'] == "scoot"]
    
    matching = pd.read_csv(args.scoot_detector_path)
    matching = matching[['detector_n', 'point_id']]

    locationData.drop(columns=['source'], inplace=True)
    locationData.rename(columns={'id': 'point_id'}, inplace=True)

    idToLoc = pd.merge(matching, locationData, on='point_id', how='inner')

    detectorLongLat = pd.DataFrame(columns=['detector_n', 'Long', 'Lat'], index=idToLoc.index)

    for index in range(len(idToLoc)):
        element = idToLoc.iloc[index]
        point = wkb.loads(binascii.unhexlify(element["location"]))
        detectorLongLat.iloc[index] = {'detector_n': element['detector_n'], 'Long': point.x, 'Lat': point.y}

    grid = [[[] for _ in range(SIZE)] for _ in range(SIZE)]

    _, minLong, minLat = detectorLongLat.min()
    _, maxLong, maxLat = detectorLongLat.max()

    longBucketSize = (maxLong - minLong) / SIZE
    latBucketSize = (maxLat - minLat) / SIZE

    detectorToLongLat = {}

    for index in range(len(detectorLongLat)):
        latBucket = int((detectorLongLat.iloc[index]['Lat'] - minLat) / latBucketSize) - 1
        longBucket = int((detectorLongLat.iloc[index]['Long'] - minLong) / longBucketSize) - 1

        grid[latBucket][longBucket].append(detectorLongLat.iloc[index]['detector_n'])
        detectorToLongLat[detectorLongLat.iloc[index]['detector_n']] = (longBucket, latBucket)

    scoot = pd.read_csv(args.scoot_data_path)
    scoot['measurement_start_utc'] = pd.to_datetime(scoot['measurement_start_utc'])
    scoot['measurement_end_utc'] = pd.to_datetime(scoot['measurement_end_utc'])

    scoot = scoot[['detector_id', 'measurement_start_utc', 'measurement_end_utc', 'n_vehicles_in_interval']]
    scoot = scoot.dropna(subset=['n_vehicles_in_interval'])
    scoot = scoot.groupby('measurement_start_utc')

    frames = []

    for time, group in scoot:
        countMask = np.zeros((SIZE, SIZE))
        frame = np.zeros((SIZE, SIZE))
        for index in range(len(group)):
            entry = group.iloc[index]
            x, y = detectorToLongLat[entry['detector_id']]
            countMask[y][x] += 1
            frame[y][x] += entry['n_vehicles_in_interval']
        for row in range(SIZE):
            for entry in range(SIZE):
                if countMask[row][entry] >= 1:
                    frame[row][entry] = int(frame[row][entry] / countMask[row][entry])
        frames.append(frame)

    frames = np.array(frames)

    result_array = create_overlapping_slices(frames, args.slice_size, args.overlap)
    np.save(args.output_path, result_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SCOOT traffic data.')
    parser.add_argument('--meta_point_path', type=str, required=True, help='Path to the meta_point CSV file.')
    parser.add_argument('--scoot_detector_path', type=str, required=True, help='Path to the scoot_detector CSV file.')
    parser.add_argument('--scoot_data_path', type=str, required=True, help='Path to the SCOOT data CSV file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed data (npy format).')
    parser.add_argument('--slice_size', type=int, default=7*24, help='Size of each data slice.')
    parser.add_argument('--overlap', type=int, default=7*24-1, help='Number of overlapping elements between slices.')

    args = parser.parse_args()
    main(args)

