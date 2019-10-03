
#include <kernel.hpp>

__inline__ __device__ short
warpReduceMax(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int   warpSize = 32;
    short myMax    = 0;
    short newInd   = 0;
    short newInd2  = 0;
    short ind      = myIndex;
    short ind2     = myIndex2;
    myMax          = val;
    unsigned mask  = __ballot_sync(0xffffffff, threadIdx.x < lengthSeqB);  // blockDim.x
    // unsigned newmask;
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val     = max(val, __shfl_down_sync(mask, val, offset));
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);
      //  __syncthreads();

        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
      //  __syncthreads();
    }
//    __syncthreads();
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}

__device__ short
blockShuffleReduce(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int              laneId = threadIdx.x % 32;
    int              warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short            myInd  = myIndex;
    short            myInd2 = myIndex2;
    myVal                   = warpReduceMax(myVal, myInd, myInd2, lengthSeqB);

    __syncthreads();
    if(laneId == 0)
        locTots[warpId] = myVal;
    if(laneId == 0)
        locInds[warpId] = myInd;
    if(laneId == 0)
        locInds2[warpId] = myInd2;

    __syncthreads();
    unsigned check =
        ((32 + blockDim.x - 1) / 32);  // mimicing the ceil function for floats
                                       // float check = ((float)blockDim.x / 32);

    if(threadIdx.x < check)  /////******//////
    {
        myVal  = locTots[threadIdx.x];
        myInd  = locInds[threadIdx.x];
        myInd2 = locInds2[threadIdx.x];
    }
    else
    {
        myVal  = 0;
        myInd  = -1;
        myInd2 = -1;
    }
    __syncthreads();

    if(warpId == 0)
    {
        myVal    = warpReduceMax(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
    return myVal;
}
__device__ __host__ short
           findMax(short array[], int length, int* ind)
{
    short max = array[0];
    *ind      = 0;

    for(int i = 1; i < length; i++)
    {
        if(array[i] > max)
        {
            max  = array[i];
            *ind = i;
        }
    }
    return max;
}

__device__ void
traceBack(short current_i, short current_j, short* seqA_align_begin,
          short* seqB_align_begin, const char* seqA, const char* seqB, short* I_i,
          short* I_j, unsigned lengthSeqB, unsigned lengthSeqA, unsigned int* diagOffset)
{
    int            myId = blockIdx.x;
    unsigned short current_diagId;     // = current_i+current_j;
    unsigned short current_locOffset;  // = 0;
    unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;

    current_diagId    = current_i + current_j;
    current_locOffset = 0;
    if(current_diagId < maxSize + 1)
    {
        current_locOffset = current_j;
    }
    else
    {
        unsigned short myOff = current_diagId - maxSize;
        current_locOffset    = current_j - myOff;
    }

    // if(myId == 0)
    // printf("diagID:%d locoffset:%d current_i:%d, current_j:%d\n",current_diagId
    // ,current_locOffset, current_i, current_j);
    short next_i = I_i[diagOffset[current_diagId] + current_locOffset];
    short next_j = I_j[diagOffset[current_diagId] + current_locOffset];

    while(((current_i != next_i) || (current_j != next_j)) && (next_j != 0) &&
          (next_i != 0))
    {
        current_i = next_i;
        current_j = next_j;

        current_diagId    = current_i + current_j;
        current_locOffset = 0;
        if(current_diagId < maxSize + 1)
        {
            current_locOffset = current_j;
        }
        else
        {
            unsigned short myOff2 = current_diagId - maxSize;
            current_locOffset     = current_j - myOff2;
        }

        next_i = I_i[diagOffset[current_diagId] + current_locOffset];
        next_j = I_j[diagOffset[current_diagId] + current_locOffset];
    }
    // printf("final current_i=%d, current_j=%d\n", current_i, current_j);
if(lengthSeqA < lengthSeqB){
  seqB_align_begin[myId] = current_i;
  seqA_align_begin[myId] = current_j;
}else{
  seqA_align_begin[myId] = current_i;
  seqB_align_begin[myId] = current_j;
}

    // printf("traceback done\n");

    // *tick_out = tick;
}

__global__ void
align_sequences_gpu(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, unsigned maxMatrixSize, short* I_i_array,
                    short* I_j_array, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end)
{
    int myId  = blockIdx.x;
    int myTId = threadIdx.x;

    //	if(myId == 4)
    //	printf("block3 lenA:%d",prefix_lengthA[myId]);
    unsigned lengthSeqA;
    unsigned lengthSeqB;
    // local pointers
    char*    seqA;
    char*    seqB;
    short *  I_i, *I_j;
    unsigned totBytes = 0;

    extern __shared__ char is_valid_array[];
    char*                  is_valid = &is_valid_array[0];

    if(myId == 0)
    {
        lengthSeqA = prefix_lengthA[0];
        lengthSeqB = prefix_lengthB[0];
        seqA       = seqA_array;
        seqB       = seqB_array;
        I_i        = I_i_array + (myId * maxMatrixSize);
        I_j        = I_j_array + (myId * maxMatrixSize);
    }
    else
    {
        lengthSeqA = prefix_lengthA[myId] - prefix_lengthA[myId - 1];
        lengthSeqB = prefix_lengthB[myId] - prefix_lengthB[myId - 1];
        seqA       = seqA_array + prefix_lengthA[myId - 1];
        seqB       = seqB_array + prefix_lengthB[myId - 1];
        I_i        = I_i_array + (myId * maxMatrixSize);
        I_j        = I_j_array + (myId * maxMatrixSize);
    }

    unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    unsigned minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

    short* curr_H =
        (short*) (&is_valid_array[3 * minSize +
                                  (minSize & 1)]);  // point where the valid_array ends
    short* prev_H      = &curr_H[minSize + 1];      // where the curr_H array ends
    short* prev_prev_H = &prev_H[minSize + 1];
    totBytes += (3 * minSize + (minSize & 1)) * sizeof(char) +
                ((minSize + 1) + (minSize + 1)) * sizeof(short);

    short* curr_E      = &prev_prev_H[minSize + 1];
    short* prev_E      = &curr_E[minSize + 1];
    short* prev_prev_E = &prev_E[minSize + 1];
    totBytes += ((minSize + 1) + (minSize + 1) + (minSize + 1)) * sizeof(short);

    short* curr_F      = &prev_prev_E[minSize + 1];
    short* prev_F      = &curr_F[minSize + 1];
    short* prev_prev_F = &prev_F[minSize + 1];
    totBytes += ((minSize + 1) + (minSize + 1) + (minSize + 1)) * sizeof(short);

    char* myLocString = (char*) &prev_prev_F[minSize + 1];
    totBytes += (minSize + 1) * sizeof(short) + (maxSize) * sizeof(char);

    unsigned      alignmentPad = 4 + (4 - totBytes % 4);
    unsigned int* diagOffset   = (unsigned int*) &myLocString[maxSize + alignmentPad];
    // char* v = is_valid;


    __syncthreads();
    memset(is_valid, 0, minSize);
    is_valid += minSize;
    memset(is_valid, 1, minSize);
    is_valid += minSize;
    memset(is_valid, 0, minSize);

    memset(curr_H, 0, 9 * (minSize + 1) * sizeof(short));

    __shared__ int i_max;
    __shared__ int j_max;
    int            j            = myTId + 1;
      char myColumnChar;
    if(lengthSeqA < lengthSeqB){
       myColumnChar = seqA[j - 1];  // read only once
      for(int i = myTId; i < lengthSeqB; i += 32)
      {
          myLocString[i] = seqB[i];
      }
      }
    else{
       myColumnChar = seqB[j - 1];
      for(int i = myTId; i < lengthSeqA; i += 32)
      {
          myLocString[i] = seqA[i];
      }
      }
    ///////////locsl dtring read in
    // for(int i = myTId; i < lengthSeqA; i += 32)
    // {
    //     myLocString[i] = seqA[i];
    // }

    __syncthreads();

    short            traceback[4];
    __shared__ short iVal[4];  //= {-1,-1,0,0};
    iVal[0] = -1;
    iVal[1] = -1;
    iVal[2] = 0;
    iVal[3] = 0;
    __shared__ short jVal[4];  //= {-1,0,-1,0};
    jVal[0] = -1;
    jVal[1] = 0;
    jVal[2] = -1;
    jVal[3] = 0;

    int ind;

    int   i            = 1;
    short thread_max   = 0;
    short thread_max_i = 0;
    short thread_max_j = 0;

    short* tmp_ptr;
    int    locSum = 0;
    // for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    // {
    //     int locDiagId = diag + 1;
    //     if(myTId == 0)
    //     {  // this computes the prefixSum for diagonal offset look up table.
    //         if(locDiagId <= minSize + 1)
    //         {
    //             locSum += locDiagId;
    //             diagOffset[locDiagId] = locSum;
    //         }
    //         else if(locDiagId > maxSize + 1)
    //         {
    //             locSum += (minSize + 1) - (locDiagId - (maxSize + 1));
    //             diagOffset[locDiagId] = locSum;
    //         }
    //         else
    //         {
    //             locSum += minSize + 1;
    //             diagOffset[locDiagId] = locSum;
    //         }
    //         diagOffset[lengthSeqA + lengthSeqB] = locSum + 2;
    //         // printf("diag:%d\tlocSum:%d\n",diag,diagOffset[locDiagId]);
    //     }
    // }
    __syncthreads();
    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {  // iterate for the number of anti-diagonals

        is_valid = is_valid - (diag < minSize || diag >= maxSize);

        tmp_ptr     = prev_H;
        prev_H      = curr_H;
        curr_H      = prev_prev_H;
        prev_prev_H = tmp_ptr;

        memset(curr_H, 0, (minSize + 1) * sizeof(short));
        __syncthreads();
        tmp_ptr     = prev_E;
        prev_E      = curr_E;
        curr_E      = prev_prev_E;
        prev_prev_E = tmp_ptr;

        memset(curr_E, 0, (minSize + 1) * sizeof(short));
        __syncthreads();
        tmp_ptr     = prev_F;
        prev_F      = curr_F;
        curr_F      = prev_prev_F;
        prev_prev_F = tmp_ptr;

        memset(curr_F, 0, (minSize + 1) * sizeof(short));

          __syncthreads();

        if(is_valid[myTId] && myTId < minSize)
        {
            short fVal  = prev_F[j] + EXTEND_GAP;
            short hfVal = prev_H[j] + START_GAP;
            short eVal  = prev_E[j - 1] + EXTEND_GAP;
            short heVal = prev_H[j - 1] + START_GAP;

            curr_F[j] = (fVal > hfVal) ? fVal : hfVal;
            curr_E[j] = (eVal > heVal) ? eVal : heVal;

        //    (myLocString[i-1] == myColumnChar)?MATCH:MISMATCH

            traceback[0] =
                prev_prev_H[j - 1] +
                ((myLocString[i - 1] == myColumnChar)
                     ? MATCH
                     : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
            traceback[1] = curr_F[j];
            traceback[2] = curr_E[j];
            traceback[3] = 0;

            curr_H[j] = findMax(traceback, 4, &ind);
            //
            // unsigned short diagId    = i + j;
            // unsigned short locOffset = 0;
            // if(diagId < maxSize + 1)
            // {
            //     locOffset = j;
            // }
            // else
            // {
            //     unsigned short myOff = diagId - maxSize;
            //     locOffset            = j - myOff;
            // }
            //
            // I_i[diagOffset[diagId] + locOffset] =
            //     i + iVal[ind];  // coalesced accesses, need to change
            // I_j[diagOffset[diagId] + locOffset] = j + jVal[ind];
            //
            thread_max_i = (thread_max >= curr_H[j]) ? thread_max_i : i;
            thread_max_j = (thread_max >= curr_H[j]) ? thread_max_j : myTId + 1;
            thread_max   = (thread_max >= curr_H[j]) ? thread_max : curr_H[j];

            i++;
        }
        __syncthreads();
    }
    __syncthreads();

    thread_max = blockShuffleReduce(thread_max, thread_max_i, thread_max_j,
                                    minSize);  // thread 0 will have the correct values

    __syncthreads();

    if(myTId == 0)
    {
        //  if(myId == 0)printf("max score fwd:%d\n", thread_max );
      //  i_max           = thread_max_i;
      //  j_max           = thread_max_j;


    //    short current_i = i_max, current_j = j_max;
        if(lengthSeqA < lengthSeqB){
          seqB_align_end[myId] = thread_max_i;
          seqA_align_end[myId] = thread_max_j;
        }else{
        seqA_align_end[myId] = thread_max_i;
        seqB_align_end[myId] = thread_max_j;
        }
      //  traceBack(current_i, current_j, seqA_align_begin, seqB_align_begin, seqA, seqB,
      //            I_i, I_j, lengthSeqB, lengthSeqA, diagOffset);
  //  }

}
__syncthreads();

//**************************************//
j            = myTId + 1;

 lengthSeqA = seqA_align_end[myId];
 lengthSeqB = seqB_align_end[myId];
int seqAend = lengthSeqA -1;
int seqBend = lengthSeqB -1;
__syncthreads();
  //if(myId == 0 && myTId == 0)printf("seqA end:%d seqB end: %d thread:%d block: %d\n",seqA_align_end[myId], seqB_align_end[myId], myTId,myId );

  // if(myId == 0 && myTId == 0){
  //   for(int l = 0; l < lengthSeqB; l++){
  //     printf("%c",seqB[l]);
  //   }
  //   printf("\n");
  // }
__syncthreads();
if(lengthSeqA < lengthSeqB){
  if(myTId < lengthSeqA)
    seqA[seqAend - (j-1)] = myColumnChar;
  //printf("thread:%d char:%c charNew:%c\n",j-1,myColumnChar,seqA[seqAend - (j-1)]);

  for(int i = myTId; i < lengthSeqB; i+=32){
    seqB[seqBend - i] = myLocString[i];
  }

}else{
  if(myTId < lengthSeqB)
  seqB[seqBend - (j-1)] = myColumnChar;
    //printf("thread:%d char:%c charNew:%c\n",j-1,myColumnChar,seqB[seqAend - (j-1)]);

  for(int i = myTId; i < lengthSeqA; i+=32){
    seqA[seqAend - i] = myLocString[i];
  }
}

__syncthreads();

// if(myId == 0 && myTId == 0){
//   for(int l = 0; l < lengthSeqA; l++){
//     printf("%c",seqA[l]);
//   }
//   printf("\n");
// }
//
// __syncthreads();



maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;
__syncthreads();
     curr_H =
        (short*) (&is_valid_array[3 * minSize +
                                  (minSize & 1)]);  // point where the valid_array ends
     prev_H      = &curr_H[minSize + 1];      // where the curr_H array ends
     prev_prev_H = &prev_H[minSize + 1];
  //  totBytes += (3 * minSize + (minSize & 1)) * sizeof(char) +
  //              ((minSize + 1) + (minSize + 1)) * sizeof(short);

     curr_E      = &prev_prev_H[minSize + 1];
     prev_E      = &curr_E[minSize + 1];
     prev_prev_E = &prev_E[minSize + 1];
    //totBytes += ((minSize + 1) + (minSize + 1) + (minSize + 1)) * sizeof(short);

     curr_F      = &prev_prev_E[minSize + 1];
     prev_F      = &curr_F[minSize + 1];
     prev_prev_F = &prev_F[minSize + 1];
  //  totBytes += ((minSize + 1) + (minSize + 1) + (minSize + 1)) * sizeof(short);

     myLocString = (char*) &prev_prev_F[minSize + 1];
__syncthreads();

           memset(curr_H, 0, 9 * (minSize + 1) * sizeof(short));

           is_valid = &is_valid_array[0]; //reset is_valid array for second iter
           memset(is_valid, 0, minSize);
           is_valid += minSize;
           memset(is_valid, 1, minSize);
           is_valid += minSize;
           memset(is_valid, 0, minSize);
__syncthreads();

if(lengthSeqA < lengthSeqB){
   if(j < lengthSeqA);
    myColumnChar = seqA[j - 1];  // read only once
  for(int i = myTId; i < lengthSeqB; i += 32)
  {
      myLocString[i] = seqB[i]; // locString contains reference/longer string
  }
  }
else{
  if(j < lengthSeqB);
   myColumnChar = seqB[j - 1];
  for(int i = myTId; i < lengthSeqA; i += 32)
  {
      myLocString[i] = seqA[i];
  }
  }

__syncthreads();
      i            = 1;
      thread_max   = 0;
      thread_max_i = 0;
      thread_max_j = 0;




  //    short* tmp_ptr;
  //  locSum = 0;

  //    __syncthreads();
      for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
      {  // iterate for the number of anti-diagonals

          is_valid = is_valid - (diag < minSize || diag >= maxSize);

          tmp_ptr     = prev_H;
          prev_H      = curr_H;
          curr_H      = prev_prev_H;
          prev_prev_H = tmp_ptr;

          memset(curr_H, 0, (minSize + 1) * sizeof(short));
          __syncthreads();
          tmp_ptr     = prev_E;
          prev_E      = curr_E;
          curr_E      = prev_prev_E;
          prev_prev_E = tmp_ptr;

          memset(curr_E, 0, (minSize + 1) * sizeof(short));
          __syncthreads();
          tmp_ptr     = prev_F;
          prev_F      = curr_F;
          curr_F      = prev_prev_F;
          prev_prev_F = tmp_ptr;

          memset(curr_F, 0, (minSize + 1) * sizeof(short));

          __syncthreads();

          if(is_valid[myTId] && myTId < minSize)
          {
              short fVal  = prev_F[j] + EXTEND_GAP;
              short hfVal = prev_H[j] + START_GAP;
              short eVal  = prev_E[j - 1] + EXTEND_GAP;
              short heVal = prev_H[j - 1] + START_GAP;

              curr_F[j] = (fVal > hfVal) ? fVal : hfVal;
              curr_E[j] = (eVal > heVal) ? eVal : heVal;

          //    (myLocString[i-1] == myColumnChar)?MATCH:MISMATCH

              traceback[0] =
                  prev_prev_H[j - 1] +
                  ((myLocString[i - 1] == myColumnChar)
                       ? MATCH
                       : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
              traceback[1] = curr_F[j];
              traceback[2] = curr_E[j];
              traceback[3] = 0;

              curr_H[j] = findMax(traceback, 4, &ind);
              //
              // unsigned short diagId    = i + j;
              // unsigned short locOffset = 0;
              // if(diagId < maxSize + 1)
              // {
              //     locOffset = j;
              // }
              // else
              // {
              //     unsigned short myOff = diagId - maxSize;
              //     locOffset            = j - myOff;
              // }
              //
              // I_i[diagOffset[diagId] + locOffset] =
              //     i + iVal[ind];  // coalesced accesses, need to change
              // I_j[diagOffset[diagId] + locOffset] = j + jVal[ind];
              //
              thread_max_i = (thread_max >= curr_H[j]) ? thread_max_i : i;
              thread_max_j = (thread_max >= curr_H[j]) ? thread_max_j : myTId + 1;
              thread_max   = (thread_max >= curr_H[j]) ? thread_max : curr_H[j];

              i++;
          }
          __syncthreads();
      }
      __syncthreads();

      thread_max = blockShuffleReduce(thread_max, thread_max_i, thread_max_j,
                                      minSize);  // thread 0 will have the correct values

      __syncthreads();


          // if(lengthSeqA < lengthSeqB){
          //   seqB_align_begin[myId] = (lengthSeqB-1) - thread_max_i;
          //   seqA_align_begin[myId] = (lengthSeqA-1) - thread_max_j;
          // }else{
          // seqA_align_begin[myId] = (lengthSeqA-1) - thread_max_i;
          // seqB_align_begin[myId] = (lengthSeqB-1) - thread_max_j;
          // }

          if(myTId == 0)
          {
            //  if(myId == 0)printf("max score rev:%d\n", thread_max );
            //   if(myId == 0)printf("max:%d thread_i:%d thread_j:%d\n", thread_max, thread_max_i, thread_max_j );
              // i_max           = thread_max_i;
              // j_max           = thread_max_j;
              // short current_i = i_max, current_j = j_max;
              if(lengthSeqA < lengthSeqB){
                seqB_align_begin[myId] = lengthSeqB - (thread_max_i);
                seqA_align_begin[myId] = lengthSeqA - (thread_max_j);
              }else{
              seqA_align_begin[myId] = lengthSeqA - (thread_max_i);
              seqB_align_begin[myId] = lengthSeqB - (thread_max_j);
              }
            //  traceBack(current_i, current_j, seqA_align_begin, seqB_align_begin, seqA, seqB,
            //            I_i, I_j, lengthSeqB, lengthSeqA, diagOffset);
          }
           __syncthreads();
 //if(myId == 0 && myTId == 0)printf("seqA begin:%d seqB begin: %d thread:%d block: %d\n",seqA_align_begin[myId], seqB_align_begin[myId], myTId,myId );
//*************************************//

    // if(myTId == 0)
    // {
    //     // if(myId == 0)printf("max:%d thread_i:%d\n", thread_max, thread_max_i );
    //     i_max           = thread_max_i;
    //     j_max           = thread_max_j;
    //     short current_i = i_max, current_j = j_max;
    //     if(lengthSeqA < lengthSeqB){
    //       seqB_align_end[myId] = current_i;
    //       seqA_align_end[myId] = current_j;
    //     }else{
    //     seqA_align_end[myId] = current_i;
    //     seqB_align_end[myId] = current_j;
    //     }
    //     traceBack(current_i, current_j, seqA_align_begin, seqB_align_begin, seqA, seqB,
    //               I_i, I_j, lengthSeqB, lengthSeqA, diagOffset);
    // }
    //  __syncthreads();
}
