import sys, os
import pickle
from pkg.EnumerateRefBins import RefGenomeReader
from pkg.Align import Aligner
from numpy import mat
import glob

def seperateBinsAndCountReads(ref_file, file_reads, bin_size, stride_size, allow_residue=False, min_base_num=30, tolerance=0):
    # Return a list of list containing read_counts. 
    enumerator = RefGenomeReader(ref_file)
    ref_seqs = enumerator.getAllBins(bin_size, stride_size, allow_residue)
    print('Reference genome split by bins read in. ')
    # This is a list of list containing bins. 
    aligner = Aligner(ref_seqs, file_reads)
    read_counts = aligner.countAlignedReads(tolerance=tolerance)
    
    print(read_counts)
    return read_counts

def complementList(read_counts, fill_with = 0):
    width = max(map(len, read_counts))
    for lst in read_counts:
        lst.extend([fill_with]*(width - len(lst)))
    shape = [len(read_counts), width]
    print('shape after filling: '+str(shape))
    return read_counts, shape

# deprecated
def mergeInOneLine(list_of_lists):
    # NOT APPLICABLE to reads number greater than 9.
    re = ''
    for lst in list_of_lists:
        lst_merged = ''.join(lst)
        re += lst_merged
    return re

# Read the following info: 
#   sample label
#   sample reads_counts_mat (each row as an exon and each column as a bin). 
def readSampleInfo(ref_file, folder_reads, bin_size, stride_size, tolerance, label):
    file_reads = glob.glob(os.path.join(folder_reads,"*.fq[.gz]*"))
    read_counts = seperateBinsAndCountReads(ref_file, file_reads, bin_size, stride_size, tolerance)
    filled_read_counts, shape = complementList(read_counts, 0)
    mat_filled_read_counts = mat(filled_read_counts)
    with open(os.path.join(folder_reads,str(label)+'_'+os.path.basename(folder_reads)+'.mat'), 'w') as fw: 
        pickle.dump(mat_filled_read_counts, fw)
    print(mat_filled_read_counts)

if __name__ == '__main__':
    ref_file, folder_reads, bin_size, stride_size, tolerance, label = sys.argv[1:7]
    bin_size, stride_size, tolerance =  int(bin_size), int(stride_size), int(tolerance)
    readSampleInfo(ref_file, folder_reads, bin_size, stride_size, tolerance, label)
    


