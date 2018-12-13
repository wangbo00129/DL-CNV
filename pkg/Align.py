from CommonPythonMethods import getCompRevSeq
import os
import time
import collections
# Must be even. 
NUM_PER_GREP = 100

class Aligner(object): 
    
    def __init__(self, ref_seqs, files_reads):
        # ref_seqs is a list of lists containing bins of reference sequences. 
        assert type(ref_seqs[0])==type([])
        self.ref_seqs = ref_seqs
        self.files_reads = files_reads

    def countAlignedReads(self, tolerance = 0):
        start = time.time()
        # Output a list of lists containing reads count at different bins. 
        # Tolerance not implemented. 
        ref_seqs = self.ref_seqs # This is a list of list. 
        files_reads = self.files_reads
        list_for_grep_awk_all = []
        for list_bin in ref_seqs: 
            for bin_seq in list_bin:
                # bin_seq is the a sequence.
                cr_bin = getCompRevSeq(bin_seq)
                list_for_grep_awk_all.append(bin_seq)
                list_for_grep_awk_all.append(cr_bin)
        lists_for_grep_awk = [list_for_grep_awk_all[i:i+NUM_PER_GREP] for i in range(len(list_for_grep_awk_all)) if i%NUM_PER_GREP==0]
        counts_files_all = []
        for list_for_grep_awk in lists_for_grep_awk:
            counts_files = collections.OrderedDict()
            for i in list_for_grep_awk:
                counts_files[i] = 0
            print('initial counts_files: ')
            print(counts_files)
            
            for f in files_reads: 
                awk_begin = " BEGIN { "+' '.join(['pat["'+seq+'"] = 0;' for seq in list_for_grep_awk])+" } " 
                awk_cmd = ' '.join(['/'+seq+'/ {pat[\"'+seq+'\"] += 1;}' for seq in list_for_grep_awk])            
                cmd = "zgrep -E '"+'|'.join(list_for_grep_awk)+"' " + f + "|awk '" + awk_begin +awk_cmd+  " END {for(i in pat){print i,pat[i]}}'"
                print(cmd)
                p = os.popen(cmd).readlines()
                # The result is the awk list of bin_seq complementary-reversed bin_seq counts. 
                print('from shell pipeline: ')
                print(p)
                for kv in p:
                    counts_files[kv.split(' ')[0]]+=int(kv.split(' ')[1]) 
            
            # Odd for bin_seq and even for cr of bin_seq. 
            tmp = list(counts_files.values())
            print('counts_files values: ')
            print(tmp)
            counts_files = [tmp[i]+tmp[i+1] for i in range(len(tmp)) if i%2==0]
            print('counts_files: ')
            print(counts_files)
            counts_files_all += counts_files
        # Reformat the list into the 2-d array. 
        list_of_list_counts = []
        idx_counts_files = 0
        for list_bin in ref_seqs:
            list_counts = [] 
            for i in range(len(list_bin)):
                list_counts.append(counts_files_all[idx_counts_files])
                idx_counts_files += 1
            list_of_list_counts.append(list_counts)
        end = time.time()
        print('time elapsed for countAlignedReads: '+str(end-start)+' s')
        return list_of_list_counts

    # Count matched reads for bin_seq (this is a sequence).
    def countMatchedReadsNotGrep(self, bin_seq, files_reads, tolerance):
        res = 0
        for file in files_reads: 
            for line in open(file): 
                if line.find(bin_seq) > -1: 
                    res += 1
        return res

    # grep version for countMatchedReads
    def countMatchedReads(self, bin_seq, files_reads, tolerance):
        res = 0
        for file in files_reads:
            cmd_sep = ['zgrep', bin_seq, file, '|wc -l']
            cmd = ' '.join(cmd_sep) 
            p = os.popen(cmd)
            s = int(p.read())
            p.close()
            
            res += s
        print(res)
        return res
