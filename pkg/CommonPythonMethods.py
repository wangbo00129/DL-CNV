def getCompRevSeq(seq, upper = True):
    assert type(seq)==type(''), 'getCompRevSeq: not a sequence'
    r_seq = seq[::-1]
    r_seq = r_seq.replace('a', 'b')
    r_seq = r_seq.replace('A', 'B')
    r_seq = r_seq.replace('t', 'a')
    r_seq = r_seq.replace('T', 'A')
    r_seq = r_seq.replace('b', 't')
    r_seq = r_seq.replace('B', 'T')
    r_seq = r_seq.replace('c', 'd')
    r_seq = r_seq.replace('C', 'D')
    r_seq = r_seq.replace('g', 'c')
    r_seq = r_seq.replace('G', 'C')
    r_seq = r_seq.replace('d', 'g')
    r_seq = r_seq.replace('D', 'G')
    if upper: 
        r_seq = r_seq.upper()
    return r_seq
 
if __name__ == '__main__': 
    import sys
    print(getCompRevSeq(sys.argv[1]))

