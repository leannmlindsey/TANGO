# TANGO 
Sequence alignment algorithms play a central role in most bioinformatics software.  Here, we present an optimized GPU implementation of the Smith-Waterman (SW) algorithm with a focus on the traceback phase. We leverage stacked diagonal-major indexing and compressed binary representation for efficient adaptation of the traceback phase to GPUs. Our proposed implementation achieves speedups of 12.6 and 9.9x compared to state-of-the-art CPU libraries for DNA and protein alignments respectively. It provides comparable performance to other GPU libraries for DNA while being the fastest SW library for protein alignments on GPUs. Further, we integrate TANGO into a large-scale metagenome assembly software to speed up a production workflow
 
### To Build:


`mkdir build `

`cd build `

`cmake CMAKE_BUILD_TYPE=Release .. `

`make `


### To Execute DNA test run: 

`./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file`

### To Execute Protein test run: <br />

`./program_gpu aa ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./out_file`

### Contact
If you need help modifying the library to match your specific use-case or for other issues and bug reports please open an issue or reach out at leann.lindsey@utah.edu


### Citation
Paper under review

### License:
        
**GPU accelerated Smith-Waterman for performing batch alignments (GPU-BSW) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.**

**If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.**

**NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.**
