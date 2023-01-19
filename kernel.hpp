#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#define NUM_OF_AA 21
#define ENCOD_MAT_SIZE 91
#define SCORE_MAT_SIZE 576
#define DEBUG_PRINT 0
#define TIMING_PRINT 0

namespace gpu_bsw{
__device__ short
warpReduceMax_with_index(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
warpReduceMax_with_index_reverse(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
warpReduceMax(short val, unsigned lengthSeqB);

__device__ short
blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
blockShuffleReduce_with_index_reverse(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
blockShuffleReduce(short val, unsigned lengthSeqB);

__device__ __host__ short
            findMaxFour(short first, short second, short third, short fourth, int* ind);

__device__ short
intToCharPlusWrite(int num, char* CIGAR, short cigar_position);

__device__ void
createCIGAR(char* longCIGAR, char* CIGAR, int maxCIGAR, 
        const char* seqA, const char* seqB, unsigned lengthShorterSeq, unsigned lengthLongerSeq, 
        bool seqBShorter, short first_j, short last_j, short first_i, short last_i); 

__device__ void
traceBack(short current_i, short current_j, char* seqA_array, char* seqB_array, unsigned* prefix_lengthA, 
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, unsigned const maxMatrixSize, int maxCIGAR,
                    char* longCIGAR, char* CIGAR, char* H_ptr, uint32_t* diagOffset);

__global__ void
sequence_dna_kernel_traceback(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, 
                    char* longCIGAR_array, char* CIGAR_array, char* H_ptr_array,
                    int maxCIGAR, unsigned const maxMatrixSize,
                    short matchScore, short misMatchScore, short startGap, short extendGap);

__global__ void
sequence_aa_kernel_traceback(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, 
                    char* longCIGAR_array, char* CIGAR_array, char* H_ptr_array, 
                    int maxCIGAR, unsigned const maxMatrixSize,
                    short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix);

}
#endif
