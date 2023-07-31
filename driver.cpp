#include "driver.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <cmath>
#include <string>

size_t gpu_bsw_driver::get_tot_gpu_mem(int id) {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, id));
  return prop.totalGlobalMem;
}
void
gpu_bsw_driver::kernel_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, int maxCIGAR, short scores[4], float factor)
{

    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    //printf("matchScore = %d, misMatchScore = %d, startGap = %d, extendGap = %d\n", matchScore, misMatchScore, startGap, extendGap);
    unsigned maxContigSize = getMaxLength(contigs);
    //printf("\nmaxContigSize = %d\n",maxContigSize);
    unsigned maxReadSize = getMaxLength(reads);
    //printf("\nmaxReadSize = %d\n",maxReadSize);
    unsigned maxCigar = (maxContigSize > maxReadSize ) ? maxContigSize : maxReadSize;
  
    unsigned const maxMatrixSize = (maxContigSize + 1 ) * (maxReadSize + 1);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount);
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;
   // printf("NBLOCKS = %d, alignmentsPerDevice = %d, leftover_device = %d, max_per_device = %d", NBLOCKS, alignmentsPerDevice, leftOver_device, max_per_device);

    initialize_alignments(alignments, totalAlignments, maxCIGAR); // pinned memory allocation
    auto start = NOW;

    //total global memory required per alignment: maxReadSize(char) + maxContigSize(char) + results (score, ref_start, ref_end, que_start, que_end) + matrixSize (maxReadSize * maxContigSize) + 2 * maxCIGAR
    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 6 * sizeof(short) +  (1.25*maxReadSize * maxContigSize) + 2 * (maxCigar);
    
    #pragma omp parallel
    {
      int my_cpu_id = omp_get_thread_num();
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      float total_time_cpu = 0;
      cudaStream_t streams_cuda[NSTREAMS];
      for(int stm = 0; stm < NSTREAMS; stm++){
        cudaStreamCreate(&streams_cuda[stm]);
      }

        //if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        factor = 0.75*1/NSTREAMS;
        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = ceil(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);

        unsigned max_alns_sugg = 8192;  //this is a function of the size of the sequences and making sure you can saturate the GPU for computation
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu; 
        int       its    = (max_per_device>max_alns_gpu)?(ceil((double)max_per_device/max_alns_gpu)):1;
        
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<gpu_mem_avail*factor<<"\n";
        std::cout<<"Diff "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail-gpu_mem_avail*factor<<"\n";
        std::cout<<"MaxRead = "<<maxReadSize<<", MaxRef = "<<maxContigSize<<", Memory required per alignment: "<<tot_mem_req_per_aln<<"\n";
        std::cout<<"Maximum Alignments: "<<max_alns_gpu<<"\n";
        

        int BLOCKS_l = alignmentsPerDevice;
        if(my_cpu_id == deviceCount - 1)
            BLOCKS_l += leftOver_device;
        unsigned leftOvers    = BLOCKS_l % its;
        unsigned stringsPerIt = BLOCKS_l / its;
        gpu_alignments gpu_data(stringsPerIt + leftOvers, maxCIGAR, maxMatrixSize); // gpu mallocs

        short* alAbeg = alignments->ref_begin + my_cpu_id * alignmentsPerDevice;
        short* alBbeg = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
        short* alAend = alignments->ref_end + my_cpu_id * alignmentsPerDevice;
        short* alBend = alignments->query_end + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
        short* top_scores_cpu = alignments->top_scores + my_cpu_id * alignmentsPerDevice;
        char* CIGAR_cpu = alignments->CIGAR + my_cpu_id * alignmentsPerDevice * maxCIGAR;

        
        unsigned* offsetA_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetA_h, sizeof(int)*(stringsPerIt + leftOvers));
        unsigned* offsetB_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetB_h, sizeof(int)*(stringsPerIt + leftOvers));

        char *strA_d, *strB_d;
        cudaErrchk(cudaMalloc(&strA_d, maxContigSize * (stringsPerIt + leftOvers) * sizeof(char)));
        cudaErrchk(cudaMalloc(&strB_d, maxReadSize *(stringsPerIt + leftOvers)* sizeof(char)));

        char* strA;
        cudaMallocHost(&strA, sizeof(char)*maxContigSize * (stringsPerIt + leftOvers));
        char* strB;
        cudaMallocHost(&strB, sizeof(char)* maxReadSize *(stringsPerIt + leftOvers));

        float total_packing = 0;

        auto    end  = NOW;
        std::chrono::duration<double>diff = end - start;
        //std::cout << "Total Execution Time (seconds) - Memory Allocation Host & Device:"<< diff.count() <<std::endl;

        auto start2 = NOW;
        for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
        {
            //printf("perGPUIts = %d of %d its\n",perGPUIts, its);
            auto packing_start = NOW;
            int                                      blocksLaunched = 0;
            std::vector<std::string>::const_iterator beginAVec;
            std::vector<std::string>::const_iterator endAVec;
            std::vector<std::string>::const_iterator beginBVec;
            std::vector<std::string>::const_iterator endBVec;
            if(perGPUIts == its - 1)
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt + leftOvers;
            }
            else
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + (alignmentsPerDevice * my_cpu_id) +  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt;
            }

            std::vector<std::string> sequencesA(beginAVec, endAVec);
            std::vector<std::string> sequencesB(beginBVec, endBVec);
            long unsigned running_sum = 0;
            int sequences_per_stream = (blocksLaunched) / NSTREAMS;
            int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
            long unsigned half_length_A = 0;
            long unsigned half_length_B = 0;

            auto start_cpu = NOW;

            for(int i = 0; i < sequencesA.size(); i++)
            {
                running_sum +=sequencesA[i].size();
                offsetA_h[i] = running_sum;//sequencesA[i].size();
                if(i == sequences_per_stream - 1){
                    half_length_A = running_sum;
                    running_sum = 0;
                  }
            }
            long unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

            running_sum = 0;
            for(int i = 0; i < sequencesB.size(); i++)
            {
                running_sum +=sequencesB[i].size();
                offsetB_h[i] = running_sum; //sequencesB[i].size();
                if(i == sequences_per_stream - 1){
                  half_length_B = running_sum;
                  running_sum = 0;
                }
            }
            long unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

            auto end_cpu = NOW;
            std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

            total_time_cpu += cpu_dur.count();
            long unsigned offsetSumA = 0;
            long unsigned offsetSumB = 0;

            for(int i = 0; i < sequencesA.size(); i++)
            {
                char* seqptrA = strA + offsetSumA;
                memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
                char* seqptrB = strB + offsetSumB;
                memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
                offsetSumA += sequencesA[i].size();
                offsetSumB += sequencesB[i].size();
            }

            auto packing_end = NOW;
            std::chrono::duration<double> packing_dur = packing_end - packing_start;

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - Sequence Packing:"<< diff.count() <<std::endl;

            total_packing += packing_dur.count();

            asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);
            unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
            unsigned maxSize = (maxReadSize > maxContigSize) ? maxReadSize : maxContigSize;
            //printf("minSize = %d, maxReadSize = %d, maxContigSize = %d \n", minSize, maxReadSize, maxContigSize);
            unsigned totShmem = 6 * (maxSize + 1) * sizeof(short) + 6 * maxSize + (maxSize & 1) + maxSize;
            
            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes = totShmem + alignmentPad + sizeof(int) * (maxContigSize + maxReadSize + 2 ) ;
            //printf("totShmem = %d, alignmentPad = %d, maxContigSize = %d, maxReadSize = %d, ShmemBytes = %d\n", totShmem, alignmentPad, maxContigSize, maxReadSize, ShmemBytes);
            if(ShmemBytes > 48000)
                cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel_traceback, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - Move sequence data to device:"<< diff.count() <<std::endl;
             

            //printf("traceback kernel 1: parameters: seq_per_stream: %d, minsize = %d, ShmemBytes = %d, streams_cuda[0] = %d, maxCIGAR = %d\n", sequences_per_stream, minSize, ShmemBytes, streams_cuda[1], maxCIGAR);
            gpu_bsw::sequence_dna_kernel_traceback<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, 
                gpu_data.longCIGAR_gpu, gpu_data.CIGAR_gpu, gpu_data.H_ptr_gpu,
                maxCIGAR, maxMatrixSize, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            //printf("traceback kernel 2: parameters: seq_per_stream: %d, minsize = %d, ShmemBytes = %d, streams_cuda[1] = %d, maxCIGAR = %d\n", sequences_per_stream, minSize, ShmemBytes, streams_cuda[1], maxCIGAR);
            gpu_bsw::sequence_dna_kernel_traceback<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
                strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                gpu_data.scores_gpu + sequences_per_stream, gpu_data.longCIGAR_gpu + sequences_per_stream * maxCIGAR, gpu_data.CIGAR_gpu + sequences_per_stream * maxCIGAR , 
                gpu_data.H_ptr_gpu + sequences_per_stream * maxMatrixSize,
               maxCIGAR, maxMatrixSize, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - DNA Forward kernel:"<< diff.count() <<std::endl;
            asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, alAend, alBend, top_scores_cpu, CIGAR_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda, maxCIGAR);

                 alAbeg += stringsPerIt;
                 alBbeg += stringsPerIt;
                 alAend += stringsPerIt;
                 alBend += stringsPerIt;
                 top_scores_cpu += stringsPerIt;
                 CIGAR_cpu += stringsPerIt*maxCIGAR;

		        cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - Copy results from device to host:"<< diff.count() <<std::endl;

        }  // for iterations end here

        auto end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;
        cudaErrchk(cudaFree(strA_d));
        cudaErrchk(cudaFree(strB_d));
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);
        cudaFreeHost(strA);
        cudaFreeHost(strB);

        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);

        //std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        //std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends

    auto                         end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
}// end of DNA kernel

void
gpu_bsw_driver::kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, int maxCIGAR, short scoring_matrix[], short startGap, short extendGap, float factor)
{

    short misMatchScore = -3;
    short matchScore = 3;

    short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                             13,7,8,9,0,11,10,12,2,0,14,5,
                             1,15,16,0,19,17,22,18,21};
    
    //short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    //printf("matchScore = %d, misMatchScore = %d, startGap = %d, extendGap = %d\n", matchScore, misMatchScore, startGap, extendGap);
    unsigned maxContigSize = getMaxLength(contigs);
    //printf("\nmaxContigSize = %d\n",maxContigSize);
    unsigned maxReadSize = getMaxLength(reads);
    //printf("\nmaxReadSize = %d\n",maxReadSize);
    unsigned maxCigar = (maxContigSize > maxReadSize ) ? maxContigSize : maxReadSize;
  
    unsigned const maxMatrixSize = (maxContigSize + 1 ) * (maxReadSize + 1);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount);
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;
   // printf("NBLOCKS = %d, alignmentsPerDevice = %d, leftover_device = %d, max_per_device = %d", NBLOCKS, alignmentsPerDevice, leftOver_device, max_per_device);

    initialize_alignments(alignments, totalAlignments, maxCIGAR); // pinned memory allocation
    auto start = NOW;

    //total global memory required per alignment: maxReadSize(char) + maxContigSize(char) + results (score, ref_start, ref_end, que_start, que_end) + matrixSize (maxReadSize * maxContigSize) + 2 * maxCIGAR
    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 6 * sizeof(short) +  (1.25*maxReadSize * maxContigSize) + 2 * (maxCigar);
    
    #pragma omp parallel
    {
      int my_cpu_id = omp_get_thread_num();
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      float total_time_cpu = 0;
      cudaStream_t streams_cuda[NSTREAMS];
      for(int stm = 0; stm < NSTREAMS; stm++){
        cudaStreamCreate(&streams_cuda[stm]);
      }

        //if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        factor = 0.75*1/NSTREAMS;
        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = ceil(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);

        unsigned max_alns_sugg = 8192;  //this is a function of the size of the sequences and making sure you can saturate the GPU for computation
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu; 
        int       its    = (max_per_device>max_alns_gpu)?(ceil((double)max_per_device/max_alns_gpu)):1;
        
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<gpu_mem_avail*factor<<"\n";
        std::cout<<"Diff "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail-gpu_mem_avail*factor<<"\n";
        std::cout<<"MaxRead = "<<maxReadSize<<", MaxRef = "<<maxContigSize<<", Memory required per alignment: "<<tot_mem_req_per_aln<<"\n";
        std::cout<<"Maximum Alignments: "<<max_alns_gpu<<"\n";
        

        int BLOCKS_l = alignmentsPerDevice;
        if(my_cpu_id == deviceCount - 1)
            BLOCKS_l += leftOver_device;
        unsigned leftOvers    = BLOCKS_l % its;
        unsigned stringsPerIt = BLOCKS_l / its;
        gpu_alignments gpu_data(stringsPerIt + leftOvers, maxCIGAR, maxMatrixSize); // gpu mallocs
        short *d_encoding_matrix, *d_scoring_matrix;
        cudaErrchk(cudaMalloc(&d_encoding_matrix, ENCOD_MAT_SIZE * sizeof(short)));
        cudaErrchk(cudaMalloc(&d_scoring_matrix, SCORE_MAT_SIZE * sizeof(short)));
        cudaErrchk(cudaMemcpy(d_encoding_matrix, encoding_matrix, ENCOD_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(d_scoring_matrix, scoring_matrix, SCORE_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice));


        short* alAbeg = alignments->ref_begin + my_cpu_id * alignmentsPerDevice;
        short* alBbeg = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
        short* alAend = alignments->ref_end + my_cpu_id * alignmentsPerDevice;
        short* alBend = alignments->query_end + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
        short* top_scores_cpu = alignments->top_scores + my_cpu_id * alignmentsPerDevice;
        char* CIGAR_cpu = alignments->CIGAR + my_cpu_id * alignmentsPerDevice * maxCIGAR;

        
        unsigned* offsetA_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetA_h, sizeof(int)*(stringsPerIt + leftOvers));
        unsigned* offsetB_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetB_h, sizeof(int)*(stringsPerIt + leftOvers));

        char *strA_d, *strB_d;
        cudaErrchk(cudaMalloc(&strA_d, maxContigSize * (stringsPerIt + leftOvers) * sizeof(char)));
        cudaErrchk(cudaMalloc(&strB_d, maxReadSize *(stringsPerIt + leftOvers)* sizeof(char)));

        char* strA;
        cudaMallocHost(&strA, sizeof(char)*maxContigSize * (stringsPerIt + leftOvers));
        char* strB;
        cudaMallocHost(&strB, sizeof(char)* maxReadSize *(stringsPerIt + leftOvers));

        float total_packing = 0;

        auto    end  = NOW;
        std::chrono::duration<double>diff = end - start;
        //std::cout << "Total Execution Time (seconds) - Memory Allocation Host & Device:"<< diff.count() <<std::endl;

        auto start2 = NOW;
        for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
        {
            //printf("perGPUIts = %d of %d its\n",perGPUIts, its);
            auto packing_start = NOW;
            int                                      blocksLaunched = 0;
            std::vector<std::string>::const_iterator beginAVec;
            std::vector<std::string>::const_iterator endAVec;
            std::vector<std::string>::const_iterator beginBVec;
            std::vector<std::string>::const_iterator endBVec;
            if(perGPUIts == its - 1)
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt + leftOvers;
            }
            else
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + (alignmentsPerDevice * my_cpu_id) +  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt;
            }

            std::vector<std::string> sequencesA(beginAVec, endAVec);
            std::vector<std::string> sequencesB(beginBVec, endBVec);
            long unsigned running_sum = 0;
            int sequences_per_stream = (blocksLaunched) / NSTREAMS;
            int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
            long unsigned half_length_A = 0;
            long unsigned half_length_B = 0;

            auto start_cpu = NOW;

            for(int i = 0; i < sequencesA.size(); i++)
            {
                running_sum +=sequencesA[i].size();
                offsetA_h[i] = running_sum;//sequencesA[i].size();
                if(i == sequences_per_stream - 1){
                    half_length_A = running_sum;
                    running_sum = 0;
                  }
            }
            long unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

            running_sum = 0;
            for(int i = 0; i < sequencesB.size(); i++)
            {
                running_sum +=sequencesB[i].size();
                offsetB_h[i] = running_sum; //sequencesB[i].size();
                if(i == sequences_per_stream - 1){
                  half_length_B = running_sum;
                  running_sum = 0;
                }
            }
            long unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

            auto end_cpu = NOW;
            std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

            total_time_cpu += cpu_dur.count();
            long unsigned offsetSumA = 0;
            long unsigned offsetSumB = 0;

            for(int i = 0; i < sequencesA.size(); i++)
            {
                char* seqptrA = strA + offsetSumA;
                memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
                char* seqptrB = strB + offsetSumB;
                memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
                offsetSumA += sequencesA[i].size();
                offsetSumB += sequencesB[i].size();
            }

            auto packing_end = NOW;
            std::chrono::duration<double> packing_dur = packing_end - packing_start;

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - Sequence Packing:"<< diff.count() <<std::endl;

            total_packing += packing_dur.count();

            asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);
            unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
            unsigned maxSize = (maxReadSize > maxContigSize) ? maxReadSize : maxContigSize;
            //printf("minSize = %d, maxReadSize = %d, maxContigSize = %d \n", minSize, maxReadSize, maxContigSize);
            unsigned totShmem = 3 * (maxSize + 1) * sizeof(short) + 6 * maxSize + (maxSize & 1) + maxSize + SCORE_MAT_SIZE * sizeof(short) + ENCOD_MAT_SIZE * sizeof(short);
            
            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes = totShmem + alignmentPad + sizeof(int) * (maxContigSize + maxReadSize + 2 ) ;
            //printf("totShmem = %d, alignmentPad = %d, maxContigSize = %d, maxReadSize = %d, ShmemBytes = %d\n", totShmem, alignmentPad, maxContigSize, maxReadSize, ShmemBytes);
            if(ShmemBytes > 48000)
                cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel_traceback, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

            end  = NOW;
            diff = end - start;
		
	    //printf("totShmem = %d\n", totShmem);	    
            gpu_bsw::sequence_aa_kernel_traceback<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, 
                gpu_data.longCIGAR_gpu, gpu_data.CIGAR_gpu, gpu_data.H_ptr_gpu,
                maxCIGAR, maxMatrixSize, startGap, extendGap, d_scoring_matrix, d_encoding_matrix);
            cudaErrchk(cudaGetLastError());

	    //printf("totShmem = %d\n", totShmem);
            gpu_bsw::sequence_aa_kernel_traceback<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                gpu_data.scores_gpu + sequences_per_stream, gpu_data.longCIGAR_gpu + sequences_per_stream * maxCIGAR, gpu_data.CIGAR_gpu + sequences_per_stream * maxCIGAR , 
                gpu_data.H_ptr_gpu + sequences_per_stream * maxMatrixSize, maxCIGAR, maxMatrixSize,
                startGap, extendGap, d_scoring_matrix, d_encoding_matrix);
            cudaErrchk(cudaGetLastError());
          

            cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - DNA Forward kernel:"<< diff.count() <<std::endl;
            asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, alAend, alBend, top_scores_cpu, CIGAR_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda, maxCIGAR);

                 alAbeg += stringsPerIt;
                 alBbeg += stringsPerIt;
                 alAend += stringsPerIt;
                 alBend += stringsPerIt;
                 top_scores_cpu += stringsPerIt;
                 CIGAR_cpu += stringsPerIt*maxCIGAR;

		        cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - Copy results from device to host:"<< diff.count() <<std::endl;

        }  // for iterations end here

        auto end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;
        cudaErrchk(cudaFree(strA_d));
        cudaErrchk(cudaFree(strB_d));
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);
        cudaFreeHost(strA);
        cudaFreeHost(strB);

        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);

        //std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        //std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends

    auto                         end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
    
}// end of amino acids kernel



