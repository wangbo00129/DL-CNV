from CommonPythonMethods import getCompRevSeq
from subprocess import call
import os

class Aligner(object):
    grep_temp_prefix = #''
    
    def __init__(self, ref_seqs, files_reads):
        # ref_seqs is a list of lists containing bins of reference sequences. 
        assert type(ref_seqs[0])==type([])
        self.ref_seqs = ref_seqs
        self.files_reads = files_reads

    def countAlignedReads(self, tolerance = 0):
        # Output a list of lists containing reads count at different bins. 
        # Tolerance not implemented. 
        ref_seqs = self.ref_seqs
        files_reads = self.files_reads
        list_of_list_counts = []
        for list_bin in ref_seqs:
            list_counts = []
            for bin in list_bin:
                # bin is the a sequence. 
                cr_bin = getCompRevSeq(bin)
                reads_count_of_bin = self.countMatchedReads(bin, files_reads, tolerance)
                reads_count_of_bin += self.countMatchedReads(cr_bin, files_reads, tolerance)
                list_counts.append(reads_count_of_bin)
                print(bin+' counting finished: '+str(reads_count_of_bin))
            list_of_list_counts.append(list_counts)
        return list_of_list_counts

    # Count matched reads for bin (this is a sequence).
    def countMatchedReadsNotGrep(self, bin, files_reads, tolerance):
        res = 0
        for file in files_reads: 
            for line in open(file): 
                if line.find(bin) > -1: 
                    res += 1
        return res

    # grep version for countMatchedReads
    def countMatchedReads(self, bin, files_reads, tolerance):
        res = 0
        for file in files_reads:
            cmd_sep = ['zgrep', bin, file, '|wc -l']
            cmd = ' '.join(cmd_sep) 
            p = os.popen(cmd)
            s = int(p.read())
            p.close()
            
            res += s
        print(res)
        return res
