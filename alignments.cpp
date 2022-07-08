#include "alignments.hpp"
#include"utils.hpp"
gpu_alignments::gpu_alignments(int max_alignments, int maxCIGAR, unsigned const maxMatrixSize){
    cudaErrchk(cudaMalloc(&offset_query_gpu, (max_alignments) * sizeof(int)));
    cudaErrchk(cudaMalloc(&offset_ref_gpu, (max_alignments) * sizeof(int)));
    cudaErrchk(cudaMalloc(&ref_start_gpu, (max_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&ref_end_gpu, (max_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&query_end_gpu, (max_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&query_start_gpu, (max_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&scores_gpu, (max_alignments) * sizeof(short)));
    printf("the size allocated for  CIGAR is %d\n", max_alignments*maxCIGAR);
    cudaErrchk(cudaMalloc(&CIGAR_gpu, (max_alignments) * sizeof(char) * maxCIGAR));
    cudaErrchk(cudaMalloc(&H_ptr_gpu, 1.25*sizeof(char)*maxMatrixSize * (max_alignments))); // added a buffer because of cuda-error in larger sequences
    cudaErrchk(cudaMalloc(&I_gpu, 1.25*sizeof(short)*maxMatrixSize * (max_alignments)));
    printf("the size allocated for longCIGAR is %d\n", max_alignments*maxCIGAR);
    cudaErrchk(cudaMalloc(&longCIGAR_gpu, sizeof(char)*maxCIGAR * (max_alignments)));
}

gpu_alignments::~gpu_alignments(){
    cudaErrchk(cudaFree(offset_ref_gpu));
    cudaErrchk(cudaFree(offset_query_gpu));
    cudaErrchk(cudaFree(ref_start_gpu));
    cudaErrchk(cudaFree(ref_end_gpu)); //is there a reason that he doesn't cudaFree scores here?
    cudaErrchk(cudaFree(query_start_gpu));
    cudaErrchk(cudaFree(CIGAR_gpu));
    cudaErrchk(cudaFree(H_ptr_gpu));
    cudaErrchk(cudaFree(I_gpu));
    cudaErrchk(cudaFree(longCIGAR_gpu));
    
}