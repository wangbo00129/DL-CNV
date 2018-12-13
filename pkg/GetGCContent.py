from CommonPythonMethods import getCompRevSeq
import os
import time
import collections
# Must be even.  

class CGContentGetter: 
    
    def __init__(self, ref_seqs, files_reads, output=None):
        # ref_seqs is a list of lists containing bins of reference sequences. 
        assert type(ref_seqs[0])==type([])
        self.ref_seqs = ref_seqs 
        self.fw = None
        self.output = output 
        self.files_reads = files_reads

    def countAlignedReads(self, tolerance = 0):
        start = time.time()
        # Output a list of lists containing reads count at different bins. 
        # Tolerance not implemented. 
        ref_seqs = self.ref_seqs # This is a list of list. 
        list_for_grep_awk_all = []
        fw = None
        if self.output:
            fw = open(self.output, 'w')
        cur1 = 1
        for list_bin in ref_seqs:
            cur2 = 1 
            for bin_seq in list_bin:
                # bin_seq is the a sequence.
                cr_bin = getCompRevSeq(bin_seq)
                bin_cr_bin = '"{}|{}"'.format(bin_seq, cr_bin)
                if fw:
                    fw.write('exon '+str(cur1)+' window '+str(cur2)+'\n')
                else:
                    print('exon '+str(cur1)+' window '+str(cur2))
                list_for_grep_awk_all.append(bin_cr_bin) 
                cmd = 'zgrep -E {} {} '.format(bin_cr_bin, ' '.join(self.files_reads))
                
                if fw:
                    fw.write(os.popen(cmd).read()+'\n')
                else:
                    print(os.popen(cmd).read())
                cur2+=1
            cur1+=1
        end = time.time()        
        print('time elapsed for countAlignedReads: '+str(end-start)+' s') 
