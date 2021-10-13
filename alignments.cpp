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
    cudaErrchk(cudaMalloc(&CIGAR_gpu, (max_alignments) * sizeof(char) * maxCIGAR));
    cudaErrchk(cudaMalloc(&H_ptr_gpu, sizeof(char)*maxMatrixSize * (max_alignments)));
    cudaErrchk(cudaMalloc(&E_ptr_gpu, sizeof(char)*maxMatrixSize * (max_alignments)));
    cudaErrchk(cudaMalloc(&F_ptr_gpu, sizeof(char)*maxMatrixSize * (max_alignments)));
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
    cudaErrchk(cudaFree(E_ptr_gpu));
    cudaErrchk(cudaFree(F_ptr_gpu));
    cudaErrchk(cudaFree(longCIGAR_gpu));
    
}