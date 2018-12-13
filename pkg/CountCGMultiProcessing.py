import os
import sys
from CountSampleGC import readSampleInfo
from multiprocessing import Process, Pool

IGNORE_EXISTING_MAT = True  

def countAllSamples(ref_file, folder_containing_samples, bin_size, stride_size, tolerance, label, process_num=5):
    bin_size, stride_size, tolerance = int(bin_size), int(stride_size), int(tolerance)
    process_pool = Pool(int(process_num))
    results = []
    folders = os.listdir(folder_containing_samples)
    for folder in folders: 
        if '.log' in folder or '.sh' in folder:
            continue
        folder = os.path.join(folder_containing_samples, folder)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        if IGNORE_EXISTING_MAT:
            if '.mat' in ''.join(files):
                print('Ignoring '+str(folder))
                continue
        res = process_pool.apply_async(readSampleInfo, (ref_file, folder, bin_size, stride_size, tolerance, label))
        results.append(res)
    process_pool.close()
    process_pool.join()
if __name__=='__main__':
    # ref_file, folder_reads, bin_size, stride_size, tolerance, label
    ref_file, folder_containing_samples, bin_size, stride_size, tolerance, label, process_num = sys.argv[1:8]
    countAllSamples(ref_file, folder_containing_samples, bin_size, stride_size, tolerance, label, process_num) 
