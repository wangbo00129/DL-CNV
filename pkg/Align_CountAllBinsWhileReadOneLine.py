from CommonPythonMethods import getCompRevSeq
from subprocess import call
import os
from numpy import mat
import copy

class Aligner(object):
    grep_tmp_prefix = ''
    def __init__(self, ref_seqs, files_reads):
        # ref_seqs is a list of lists containing bins of reference sequences. 
        assert type(ref_seqs[0])==type([])
        self.ref_seqs = ref_seqs
        self.files_reads = files_reads
    # While reading the reads file, count all bins. 
    def countAlignedReads(self, tolerance = 0):
        # Output a list of lists containing reads count at different bins. 
        # Tolerance not implemented. 
        files_reads = self.files_reads
        ref_seqs = self.ref_seqs
        
        list_of_list_counts = copy.deepcopy(ref_seqs)
        for l in range(len(list_of_list_counts)):
            for e in range(len(list_of_list_counts[l])):
                list_of_list_counts[l][e]=0
        
        for file in files_reads: 
            for line in open(file): 
                for m in range(len(ref_seqs)):
                    list_bin=ref_seqs[m]
                    for n in range(len(list_bin)):
                        # bin is the a sequence. 
                        bin = ref_seqs[m][n]
                        cr_bin = getCompRevSeq(bin)
                        if line.find(bin) > -1:
                            list_of_list_counts[m][n] += 1
                        if line.find(cr_bin) > -1: 
                            list_of_list_counts[m][n] += 1 
        return list_of_list_counts

    # Count matched reads for bin (this is a sequence).
    def countMatchedReadsNotGrep(self, bin, files_reads, tolerance):
        res = 0
        
        for file in files_reads: 
            for line in open(file): 
                if line.find(bin) > -1: 
                    res += 1
        return res
