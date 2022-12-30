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
    unsigned maxCigar = (maxContigSize > maxReadSize ) ? 3*maxContigSize : 3*maxReadSize;
  
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
        printf("stringsPerIt = %d, leftovers = %d\n",stringsPerIt, leftOvers);
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
            //unsigned totShmem = 6 * (minSize + 1) * sizeof(short) + 6 * minSize + (minSize & 1) + maxSize;
            unsigned totShmem = (3 * (minSize + 1) * sizeof(long)) + (minSize+1 + maxSize + 1) * sizeof(long);
            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes = totShmem + alignmentPad + sizeof(long) * (maxContigSize + maxReadSize + 2 );
             
            //printf("totShmem = %d, alignmentPad = %d, maxContigSize = %d, maxReadSize = %d, ShmemBytes = %d\n", totShmem, alignmentPad, maxContigSize, maxReadSize, ShmemBytes);
            if(ShmemBytes > 48000)
                cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel_traceback, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

            end  = NOW;
            diff = end - start;
            //std::cout << "Total Execution Time (seconds) - Move sequence data to device:"<< diff.count() <<std::endl;
                    
            gpu_bsw::sequence_dna_kernel_traceback<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, 
                gpu_data.longCIGAR_gpu, gpu_data.CIGAR_gpu, gpu_data.H_ptr_gpu, gpu_data.I_gpu,
                maxCIGAR, maxMatrixSize, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            // printf("traceback kernel 2: parameters: seq_per_stream: %d, minsize = %d, ShmemBytes = %d, streams_cuda[1] = %d, maxCIGAR = %d\n", sequences_per_stream, minSize, ShmemBytes, streams_cuda[1], maxCIGAR);
            gpu_bsw::sequence_dna_kernel_traceback<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
                 strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                 gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                 gpu_data.scores_gpu + sequences_per_stream, gpu_data.longCIGAR_gpu + sequences_per_stream * maxCIGAR, gpu_data.CIGAR_gpu + sequences_per_stream * maxCIGAR , 
                 gpu_data.H_ptr_gpu + sequences_per_stream * maxMatrixSize, gpu_data.I_gpu + sequences_per_stream * maxMatrixSize,
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
gpu_bsw_driver::kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, int maxCIGAR, short scoring_matrix[], short openGap, short extendGap, float factor)
{
    unsigned maxContigSize = getMaxLength(contigs);
    //("\nmaxContigSize = %d\n",maxContigSize);
    unsigned maxReadSize = getMaxLength(reads);
    //printf("\nmaxReadSize = %d\n",maxReadSize);

    unsigned const maxMatrixSize = (maxContigSize + 1 ) * (maxReadSize + 1);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                             13,7,8,9,0,11,10,12,2,0,14,5,
                             1,15,16,0,19,17,22,18,21};

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount);// one OMP thread per GPU
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    cudaDeviceProp prop[deviceCount];
    for(int i = 0; i < deviceCount; i++)
      cudaGetDeviceProperties(&prop[i], 0);

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;
    
    initialize_alignments(alignments, totalAlignments, maxCIGAR); // pinned memory allocation
    auto start = NOW;

    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 6 * sizeof(short) + (1.25 * maxReadSize * maxContigSize) + 2 * (maxCIGAR);
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
      if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        factor = 0.75*1/NSTREAMS;
        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = ceil(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);
        unsigned max_alns_sugg = 8192;
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu;
        int       its    = (max_per_device>max_alns_gpu)?(ceil((double)max_per_device/max_alns_gpu)):1;
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail*factor<<"\n";
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

      auto start2 = NOW;
      //std::cout<<"loop begin\n";
      for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
      {
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
          unsigned running_sum = 0;
          int sequences_per_stream = (blocksLaunched) / NSTREAMS;
          int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
          unsigned half_length_A = 0;
          unsigned half_length_B = 0;

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
          unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

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
          unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

          auto end_cpu = NOW;
          std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

          total_time_cpu += cpu_dur.count();
          unsigned offsetSumA = 0;
          unsigned offsetSumB = 0;

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

          total_packing += packing_dur.count();

          asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);
          unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
          unsigned maxSize = (maxReadSize > maxContigSize) ? maxReadSize : maxContigSize;
          unsigned totShmem = 6 * (minSize + 1) * sizeof(long) + 6 * minSize + (minSize & 1) + maxSize + SCORE_MAT_SIZE * sizeof(short) + ENCOD_MAT_SIZE * sizeof(short); //check this
         
          unsigned alignmentPad = 4 + (4 - totShmem % 4);
          size_t   ShmemBytes = totShmem + alignmentPad;
          if(ShmemBytes > 48000)
              cudaFuncSetAttribute(gpu_bsw::sequence_aa_kernel_traceback, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);


          gpu_bsw::sequence_aa_kernel_traceback<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, 
                gpu_data.longCIGAR_gpu, gpu_data.CIGAR_gpu, gpu_data.H_ptr_gpu,
                maxCIGAR, maxMatrixSize, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
            cudaErrchk(cudaGetLastError());

          gpu_bsw::sequence_aa_kernel_traceback<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                gpu_data.scores_gpu + sequences_per_stream, gpu_data.longCIGAR_gpu + sequences_per_stream * maxCIGAR, gpu_data.CIGAR_gpu + sequences_per_stream * maxCIGAR , 
                gpu_data.H_ptr_gpu + sequences_per_stream * maxMatrixSize, maxCIGAR, maxMatrixSize,
                openGap, extendGap, d_scoring_matrix, d_encoding_matrix);

            cudaErrchk(cudaGetLastError());


          cudaStreamSynchronize (streams_cuda[0]);
          cudaStreamSynchronize (streams_cuda[1]);

          auto sec_cpu_start = NOW;
          int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
          auto sec_cpu_end = NOW;
          std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
          total_time_cpu += dur_sec_cpu.count();

          asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, alAend, alBend, top_scores_cpu, CIGAR_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda, maxCIGAR);

                alAbeg += stringsPerIt;
                alBbeg += stringsPerIt;
                alAend += stringsPerIt;
                alBend += stringsPerIt;
                top_scores_cpu += stringsPerIt;
                CIGAR_cpu += stringsPerIt*maxCIGAR;

		 cudaStreamSynchronize (streams_cuda[0]);
     cudaStreamSynchronize (streams_cuda[1]);

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

        std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends
    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
}// end of amino acids kernel

void
gpu_bsw_driver::verificationTest(char* rstFile, short scores[4], short* top_scores, short* ref_beg, short* ref_end, short* query_beg, short* query_end, char* CIGAR, int maxCIGAR)
{
    //printf("Verification Testing now\n");
    std::string   rstLine;
    std::ifstream rst_file(rstFile);
    int           k = 0, beg_errors = 0, end_errors = 0, cig_errors = 0;
    int cigar_k;
    char *seq_cigar[maxCIGAR];
    int cigScore, actual_cigScore;
    int cigScoreErrors = 0;

    if(rst_file.is_open())
    {
      while(getline(rst_file, rstLine))
        {
            std::string in = rstLine.substr(rstLine.find(":") + 1, rstLine.size());
            std::string valsVec;
            std::stringstream myStream(in);
            
            
            int actual_score;
            int actual_ref_st ; 
            int actual_ref_end ;
            int actual_que_st  ;
            int actual_que_end ;
            std::string cigar;

            //myStream >> score >> ref_st >> ref_end >> que_st >> que_end >> cigar;
            myStream >> actual_score >> actual_ref_st >> actual_ref_end >> actual_que_st >> actual_que_end >> cigar;
            std::cout << "***************************************************************************\n" << std::endl;
            std::cout << "***************************************************************************\n" << std::endl;
            std::cout << "***************************************************************************\n" << std::endl;
            std::cout << "Vtest File k=" << k << "\t" << actual_score << "\t" << actual_ref_st << "\t" << actual_ref_end << "\t" << actual_que_st << "\t" << actual_que_end << std:: endl;
            std::cout << "Traceback k=" << k << "\t" << top_scores[k] << "\t" << ref_beg[k] << "\t" << ref_end[k] << "\t" << query_beg[k] << "\t" << query_end[k] << std:: endl;
            std::cout << "***************************************************************************\n" << std::endl;
            if (abs(top_scores[k] - actual_score) > 0 || abs(ref_beg[k] - actual_ref_st) > 0 ||
              abs(query_beg[k] - actual_que_st) >0)
            {
                 beg_errors++;
                //printf("%d: %d, %d : %d, %d : %d, %d : %d, %d : %d, %d \n ", k, top_scores[k], actual_score, ref_beg[k], actual_ref_st, ref_end[k], actual_ref_end, 
                   // query_beg[k], actual_que_st, query_end[k], actual_que_end);
                printf("ERROR #%d, %d: diff in ref start: %d, diff in que start: %d\n",beg_errors, k, ref_beg[k] - actual_ref_st, query_beg[k] - actual_que_st);
               
            } else {
                printf("CORRECT! no differences in start positions.\n");
            }

            if (abs(top_scores[k] - actual_score) > 0 || abs(ref_end[k] - actual_ref_end) > 0  ||
              abs(query_end[k] - actual_que_end) >0)
            {
                 end_errors++;
                //printf("%d: %d, %d : %d, %d : %d, %d : %d, %d : %d, %d \n ", k, top_scores[k], actual_score, ref_beg[k], actual_ref_st, ref_end[k], actual_ref_end, 
                   // query_beg[k], actual_que_st, query_end[k], actual_que_end);
                printf("ERROR #%d, %d: diff in ref end: %d, diff in que end: %d\n",end_errors, k, ref_end[k] - actual_ref_end, query_end[k] - actual_que_end);
               
            } else {
                printf("CORRECT! no differences in end positions.\n");
            }
             
            std::cout << "***************************************************************************" << std::endl;
            cigar_k=k*maxCIGAR; //get the index for each cigar string
            *seq_cigar = &CIGAR[cigar_k];
         

            if (cigar != *seq_cigar){
                std::cout << "#" << k << "\nSSW:\t" << cigar << "\nGSW:\t" << *seq_cigar << std::endl;
                cig_errors++;
            }  

            
            //verify that the SSW CIGAR score = result score
            printf("> %d\n", k);
            cigScore = gpu_bsw_driver::scoreCIGAR(*seq_cigar, scores);
            actual_cigScore = gpu_bsw_driver::scoreCIGAR(cigar, scores);
            
            if (top_scores[k] != cigScore){
                printf("The result score from ADEPT traceback %d does not match the cigScore %d\n", top_scores[k], cigScore);
                cigScoreErrors ++;
            } 
            else {
                printf("The cigScore from ADEPT traceback is a MATCH!\n");
            }
            if (actual_score != actual_cigScore){
                printf("The result score from vTEST %d does not match the cigScore from vTest %d\n", actual_score, actual_cigScore);
                cigScoreErrors ++;
            } 
            else {
                printf("The cigScore from vTEST is a MATCH!\n");
            }
            k++;
        }
        if((beg_errors + end_errors) == 0)
            std::cout << "VERIFICATION TEST PASSED" << std::endl;
        else
            std::cout << "ERRORS OCCURRED DURING VERIFICATION TEST, beg_error count:" <<beg_errors<< " end_error count:" <<end_errors<< ", cigar error count:" << cig_errors << ", cigar Score Errors: " << cigScoreErrors << std::endl;
    }
}

int
gpu_bsw_driver::scoreCIGAR(std::string CIGAR, short scores[4])
{
    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    int digit_length = 0;
    char digit_array[5]; 
    int cigar_n =0;
    short score = 0;
    
    for (int i = 0; CIGAR[i] != '\0'; i++){
        
        // create int from numeric portion of cigar string
        if ( CIGAR[i] >= 48 && CIGAR[i] <= 57){
            digit_array[digit_length] = CIGAR[i]; //adds one number to digit array 
            digit_length++;
            //if (myId ==0 && myTId ==0){
                //printf("char is a digit! add to the array \n");
                //printf("digit array length is %d\n", digit_length);
                //printf("Current number in digit array = ");
                //for (int q = 0; q < digit_length; q++){
                    //printf("%c", digit_array[q]);
                //}
                //printf("\n\n");
            //}
        } else {
            //first convert int from char array
            for (int j = 0; j < digit_length; j++ ){
                
                cigar_n = cigar_n + (exp10f((float)digit_length-j-1) * (digit_array[j]-'0'));
                digit_array[j] = '\0';
                
            }
            digit_length = 0;
            
            //add scoring multiply here 
            if (CIGAR[i] == '='){
               
                score = score + (cigar_n * matchScore);
                //printf("char is =, r_score is  : %d\n\n", r_score);
              
            } else if (CIGAR[i]== 'X') {
                score = score + (cigar_n * misMatchScore);
                //printf("char is X, r_score is  : %d\n\n", r_score);
            } else if (CIGAR[i]== 'D') {
                score = score + startGap + (cigar_n - 1) *extendGap;
                //printf("char is D, r_score is  : %d\n\n", r_score);
            } else if (CIGAR[i]== 'I') {
                score = score + startGap + (cigar_n - 1) * extendGap;
                //printf("char is I, r_score is  : %d\n\n", r_score);
            } else {}

            cigar_n=0;
        }
        

    }
    
    //std::cout << "Here is the CIGAR " << CIGAR << " Here is the SCORE = " << score << std::endl;
    
    return score;
}
