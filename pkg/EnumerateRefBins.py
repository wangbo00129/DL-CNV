import sys
import os
class RefGenomeReader(object):
    
    def __init__(self, ref_file):
        self.ref_file = ref_file
        
    def getAllBins(self, bin_size, stride_size, allow_residue = False, min_base_num = 30):
        # When allow_residue is True, min_base_num will be used. 
        fr = open(self.ref_file, 'r')
        assert stride_size <= bin_size, 'bin size shall be greater than stride size'
        # Assuming that the file contains only sequences without descriptions/titles. 
        bs = bin_size
        ss = stride_size
        list_of_lists=[]
        for line in fr: 
            line = line.strip()
            list_of_lists.append(self.splitLineToListOfBin(line, bs, ss, allow_residue, min_base_num))

        return list_of_lists

    def splitLineToListOfBin(self, line, bs, ss, allow_residue, min_base_num): 
        exon_len = len(line)
        list_re = []
        # exon_len/ss means the last stride 
        for i in range(int(exon_len/ss)): 
            bin_start = ss * i
            bin_stop = bin_start + bs
            if bin_stop > exon_len: 
                if not allow_residue: 
                    # The last bin is smaller than bin_size. 
                    break
                bin_stop = exon_len
                if bin_stop - bin_start < min_base_num: 
                    # The last bin is smaller than min_base_num
                    break
            
            current_bin = line[bin_start:bin_stop]
            current_bin = current_bin.upper()
            list_re.append(current_bin)
        
        return list_re

    def nextBin(self):
        pass
