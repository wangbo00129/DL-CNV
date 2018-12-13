DEBUG_LEVEL = 1

def Log(err, debug_level=0):
    if debug_level>=DEBUG_LEVEL:
        print(str(err))
def getNumberFromListOfSeqs(list_of_seqs):
    sum_of_base = 0
    for i in list_of_seqs:
        sum_of_base += len(i.strip())
    return sum_of_base

def calculateCGInWindow(lines):
    
    seqs = []
    for line in lines:
        if ':' in line:
            seqs.append(line.split(":")[1])
        else:
            seqs.append(line)
    # Log(seqs)
    total_base_num = getNumberFromListOfSeqs(seqs)
    # Assume all seqs are in capital. 
    seqs_cg = map(lambda x:x.replace('A', '').replace('T',''), seqs)
    cg_base_num = getNumberFromListOfSeqs(seqs_cg)
    if total_base_num == 0:
        return 0.5
    return cg_base_num/float(total_base_num)
    
def calculateMeanCGInFile(f):
    # f is the file containing reads with '...window...' as delimiter 
    #                    for different windows.
    fr = open(f, 'r')
    lines_for_window = ['INIT']
    list_cg_ratios_for_windows=[]
    for line in fr:
        if 'window' in line: 
            if not lines_for_window[0] == 'INIT':
                # Not the initial lines_for_window
                Log(line)
                Log(lines_for_window)
                cg_ratio = calculateCGInWindow(lines_for_window)
                Log(cg_ratio)
                list_cg_ratios_for_windows.append(cg_ratio)
            lines_for_window = []
        else:
            lines_for_window.append(line)
    # Calculate the last. 
    Log('last window')
    Log(lines_for_window)
    cg_ratio = calculateCGInWindow(lines_for_window)
    Log(cg_ratio)
    list_cg_ratios_for_windows.append(cg_ratio)
    # Return mean of the list.
    return sum(list_cg_ratios_for_windows)/len(list_cg_ratios_for_windows)
    
if __name__ == '__main__':
    import sys
    mean = calculateMeanCGInFile(sys.argv[1])
    print(mean)