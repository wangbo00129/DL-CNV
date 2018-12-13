# CNN for the detection of CNV
## Generating matrix. 
Run
```bash
python CountSample.py ref_file  folder_reads  bin_size  stride_size  tolerance  label
# ref_file: a plain text file for the fasta sequence of a gene, each exon as a line. 
# folder_reads: the sample folder who contains fq or fq.gz files. 
# bin_size: the window size to split the exon. 
# stride_size: the stride size to split the exon. 
# tolerance: the tolerance of the difference between the reads and the ref_file in a window. This is not implemented yet, so it's 0 whatever the input is. 
# label: 0 for negative or 1 for positive. 
```

to generate matrix for a sample. Notice that the python program calls shell commands "grep", so this shall run on linux. 

For multiple samples, use the following 

```bash
python CountSamplesMultiProcessing.py ref_file  folder_containing_samples  bin_size  stride_size  tolerance  label process_num
# folder_containing_samples: the folder containing the sample folders who contains .fq or .fq.gz files.
# process_num: the max process number for parallel counting. 
# other parameters are the same as in CountSample.py. 
```
to implement counting parallel. It also needs to be run on linux. 
## CNN Model training. 
For model training, the cross validation method was used. 
First, make a directory for all training set matrices. This directory shall be under a clean folder for later cross validation usage. 
Use 
```
python CNNCrossValidation.py input_dir cross_num fold_num learning_rate iter_num keep_probability kernel_size over_sampling_class over_sampling_fold
# input_dir: the above-mentioned path of the directory containing matrices. 
# cross_num: the number that cross validation to be done. 
# fold_num: the fold of the cross validation. 
# learning_rate: learning rate for gradient descent. 
# iter_num: iteration number for gradient descent. 
# keep_probability: the keep probability for dropout layer in CNN. 
# kernel_size: the filter size (width and height) for the convolutional layer. 
# over_sampling_class: the class to oversample if necessary [optional].  
# over_sampling_fold: the over sampling fold for the over_sampling_class [optional]. 
```
for cross validation. 
## Model comparison. 
Use shell commands to implement. 
```
tail -n 10 *xCV_data_rand_*/one_kernel_0.5_oversamp1_20fold_normalizebymean_fillby0_stride25_part_*/log.log |grep accuracy    |awk '{print $2}' > accuracy_20000iter.txt
tail -n 10 *xCV_data_rand_*/one_kernel_0.5_oversamp1_20fold_normalizebymean_fillby0_stride25_part_*/log.log |grep sensitivity |awk '{print $2}' > sensitivity_20000iter.txt
tail -n 10 *xCV_data_rand_*/*_kernel_0.5_oversamp1_20fold_normalizebymean_fillby0_stride25_part_*/log.log |grep specificity |awk '{print $2}' > specificity_20000iter.txt
# Convert to unix format in case the previous step was run on Windows. 
dos2unix *_20000iter.txt
paste accuracy_20000iter.txt sensitivity_20000iter.txt specificity_20000iter > accu_sens_spec_20000iter.txt
```
Check the performances of the models in the last .txt file. 