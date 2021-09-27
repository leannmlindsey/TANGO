#include <kernel.hpp>

__inline__ __device__ short
gpu_bsw::warpReduceMax_with_index_reverse(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB)
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

        int tempVal = __shfl_down_sync(mask, val, offset);
        val     = max(val,tempVal);
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);

      //  if(threadIdx.x == 0)printf("index1:%d, index2:%d, max:%d\n", newInd, newInd2, val);
        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
        else if((val == tempVal) ) // this is kind of redundant and has been done purely to match the results
                                    // with SSW to get the smallest alignment with highest score. Theoreticaly
                                    // all the alignmnts with same score are same.
        {
          if(newInd2 > ind2){
            ind = newInd;
            ind2 = newInd2;

          }
        }
    }
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}

__inline__ __device__ short
gpu_bsw::warpReduceMax_with_index(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB)
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

        int tempVal = __shfl_down_sync(mask, val, offset);
        val     = max(val,tempVal);
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);
        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
        else if((val == tempVal) ) // this is kind of redundant and has been done purely to match the results
                                    // with SSW to get the smallest alignment with highest score. Theoreticaly
                                    // all the alignmnts with same score are same.
        {
          if(newInd < ind){
            ind = newInd;
            ind2 = newInd2;
          }
        }
    }
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}



__device__ short
gpu_bsw::blockShuffleReduce_with_index_reverse(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int              laneId = threadIdx.x % 32;
    int              warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short            myInd  = myIndex;
    short            myInd2 = myIndex2;
    myVal                   = warpReduceMax_with_index_reverse(myVal, myInd, myInd2, lengthSeqB);

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
        myVal    = warpReduceMax_with_index_reverse(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
    return myVal;
}

__device__ short
gpu_bsw::blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int              laneId = threadIdx.x % 32;
    int              warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short            myInd  = myIndex;
    short            myInd2 = myIndex2;
    myVal                   = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB);

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
        myVal    = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
    return myVal;
}



__device__ __host__ short
           gpu_bsw::findMaxFour(short first, short second, short third, short fourth, int* ind)
{
    short traceback[4];
    traceback[0] = first;
    traceback[1] = second;
    traceback[2] = third;
    traceback[3] = fourth;

    short maxScore = 0;
    *ind = 0;

    for (int i=0; i<4; i++){
      if (traceback[i] > maxScore){
        maxScore = traceback[i];
        *ind = i;
      }
    }

    return maxScore;
}

__device__ short
gpu_bsw::intToCharPlusWrite(int num, char* CIGAR, short cigar_position)
{
    int last_digit = 0;
    int digit_length = 0;
    char digit_array[5];

    // convert the int num to ASCII digit by digit and record in a digit_array
    while (num != 0){
        last_digit = num%10;
        digit_array[digit_length] = char('0' + last_digit);
        num = num/10;
        digit_length++;
    }

    //write each char of the digit_array to the CIGAR string
    for (int q = 0; q < digit_length; q++){
        CIGAR[cigar_position]=digit_array[q];
        cigar_position++; 
    }

    return cigar_position;
}

__device__ void
gpu_bsw::createCIGAR(char* longCIGAR, char* CIGAR, int maxCIGAR, 
        const char* seqA, const char* seqB, unsigned lengthShorterSeq, unsigned lengthLongerSeq, 
        bool seqBShorter, short first_j, short last_j, short first_i, short last_i) 
{
  
    //printf("first_j = %d, last_j = %d, first_i = %d, last_i = %d\n", first_j, last_j, first_i, last_i);
    int myId  = blockIdx.x;
    int myTId = threadIdx.x;
    short cigar_position = 0;
    int last_digit = 0;

    short beg_S;
    short end_S;

    const char* longerSeq;
    const char* shorterSeq;
    
    if (seqBShorter){
        longerSeq = seqA;
        shorterSeq = seqB;
        beg_S = lengthShorterSeq - first_j;
        end_S = last_j-1; 
    } else {
        longerSeq = seqB;
        shorterSeq = seqA;
        beg_S = lengthLongerSeq - first_i; 
        end_S = last_i-1;
    }
    
    if ( beg_S != 0){
        //if (myId ==0)
          //printf("made it inside the if loop, thread_Id = %d\n", myId);
        CIGAR[0]='S';
        cigar_position++ ; 
        cigar_position = intToCharPlusWrite(beg_S, CIGAR, cigar_position);
    }
    //if(myId==0 && myTId ==0) {
      //      printf("printing the CIGAR, cigar_position = %d, beg_S = %d\n", cigar_position,beg_S);
      //      for (int i = 0; i <= cigar_position; i++){
      //          printf("%c",CIGAR[i]);
      //      }
      //   printf("\n");
      // }  

    int q = 0;

    int p = 0;
    while(longCIGAR[p] != '\0'){
        int digit_length = 0;
        char digit_array[5];
        int letter_count = 1;
        
        while (longCIGAR[p] == longCIGAR[p+1]){
            letter_count++; 
            p++; 
        }

        CIGAR[cigar_position]=longCIGAR[p];
        cigar_position++ ; 

        cigar_position = intToCharPlusWrite(letter_count, CIGAR, cigar_position); 
        p++;
    }
    
    if ( end_S != 0){
        CIGAR[cigar_position]='S';
        cigar_position++ ; 
        cigar_position = intToCharPlusWrite(end_S, CIGAR, cigar_position);
    }    
    cigar_position--;
    char temp;

    for(int i = 0; i<(cigar_position)/2+1;i++){
        temp = CIGAR[i]; 
        CIGAR[i]=CIGAR[cigar_position-i]; 
        CIGAR[cigar_position-i] = temp; 
    }

    //if(myId==0 && myTId ==0) {
            //printf("in the CREATE CIGAR loop, printing the CIGAR, cigar_position = %d\n", cigar_position);
            //for (int i = 0; i <= cigar_position; i++){
              //  printf("%c",CIGAR[i]);
            //}
            //printf("cigar pointer = %p\n",&CIGAR);
    //}  

}

__device__ void
traceBack(short current_i, short current_j, char* seqA_array, char* seqB_array, unsigned* prefix_lengthA, 
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, unsigned const maxMatrixSize, int maxCIGAR,
                    char* longCIGAR, char* CIGAR, char* H_ptr, char* E_ptr, char* F_ptr, unsigned int* diagOffset)
{                    
    int myId = blockIdx.x;
    int myTId = threadIdx.x; 

    char*    seqA;
    char*    seqB;

    int lengthSeqA;
    int lengthSeqB;

    if(myId == 0)
    {
        lengthSeqA = prefix_lengthA[0];
        lengthSeqB = prefix_lengthB[0];
        seqA       = seqA_array;
        seqB       = seqB_array;
    
    }
    else
    {
        lengthSeqA = prefix_lengthA[myId] - prefix_lengthA[myId - 1];
        lengthSeqB = prefix_lengthB[myId] - prefix_lengthB[myId - 1];
        seqA       = seqA_array + prefix_lengthA[myId - 1];
        seqB       = seqB_array + prefix_lengthB[myId - 1];

    }

    unsigned short current_diagId;     // = current_i+current_j;
    unsigned short current_locOffset;  // = 0;
    unsigned short last_diagId;
    unsigned short last_locOffset;
    unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    bool gapOpen = 0;

    const char* longerSeq = lengthSeqA < lengthSeqB ? seqB : seqA; 
    const char* shorterSeq = lengthSeqA < lengthSeqB ? seqA : seqB; 
    unsigned lengthShorterSeq = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;  
    unsigned lengthLongerSeq = lengthSeqA < lengthSeqB ? lengthSeqB : lengthSeqA;    
    bool seqBShorter = lengthSeqA < lengthSeqB ? false : true;  //need to keep track of whether query or ref is shorter for CIGAR 

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
    
    short next_i;
    short next_j;
    short last_i = 0;
    short last_j = 0;

    switch (H_ptr[diagOffset[current_diagId] + current_locOffset]){
        case '\\' :
            next_i = current_i - 1;
            next_j = current_j - 1;
            break;
        case '|' :
            next_i = current_i - 1;
            next_j = current_j;
            break;
        case '-' :
            next_i = current_i;
            next_j = current_j - 1;
            break;

    }

    short first_j = current_j; //recording the first_j, first_i values for use in calculating S
    short first_i = current_i;
    short current_score = 0;
    short last_score = 0;
    char matrix = 'H'; //initialize with H

    int counter = 0;
   
    while(((current_i != next_i) || (current_j != next_j)) && (next_j != 0) && (next_i != 0))
    {
        
        //if(myId==0 && myTId ==0) {
            //printf("printing the longCIGAR, counter = %d\n", counter);
            //for (int i = 0; i <= counter; i++){
                //printf("%c",longCIGAR[i]);
            //}
         //printf("\n");
       //}  
      
        if (next_i == current_i){
            
            if (E_ptr[diagOffset[current_diagId] + current_locOffset] == 'H') {
                gapOpen = 1;
            }
            if (gapOpen == 0) {
                matrix = 'E';
            }
            
            // no match, no vertical movement, query has Insertion if que shorter and Deletion if ref is shorter       
            longCIGAR[counter] = seqBShorter ? 'I' : 'D';
            
            if( longCIGAR[counter-1] == 'X' && last_i == current_i){
                longCIGAR[counter-1] = seqBShorter ? 'I' : 'D'; //an aligned mismatch before an insertion or deletion is a part of the insertion or deletion
            }     

            if (shorterSeq[current_j-1] == longerSeq[current_i-1] && gapOpen == 1 && next_i == current_i -1) {
                longCIGAR[counter] = '='; //if match at the end of an indel, = in CIGAR
            }
        } else if (next_j == current_j) {
            if (F_ptr[diagOffset[current_diagId] + current_locOffset] == 'H') {
                gapOpen = 1;
            }
            if (gapOpen == 0) 
                matrix = 'F';
            // no match, no horizontal movement, query has Deletion if que shorter and Deletion if ref is shorter
          
            longCIGAR[counter] = seqBShorter ? 'D' : 'I';

            if( longCIGAR[counter-1] == 'X' && last_j == current_j){
                longCIGAR[counter-1] = seqBShorter ? 'D' : 'I'; //an aligned mismatch before an insertion or deletion is a part of the insertion or deletion
            } 
            if (shorterSeq[current_j-1] == longerSeq[current_i-1] && gapOpen == 1 && next_j == current_j -1) {
                longCIGAR[counter] = '=';
            }

        } else if ((next_i == current_i - 1) && (next_j == current_j - 1)  ){
                matrix = 'H';
                gapOpen = 0;
            
            if (shorterSeq[current_j-1] == longerSeq[current_i-1]) {
                longCIGAR[counter] = '=';
            } else { 
                longCIGAR[counter] = 'X';
            }
        }
        last_i = current_i;
        last_j = current_j;

        current_i = next_i;
        current_j = next_j;

        last_diagId = current_diagId;
        last_locOffset = current_locOffset;

        current_diagId    = current_i + current_j;
        current_locOffset = 0;

        if(current_diagId < maxSize + 1)
        {
            current_locOffset = current_j;
        } else {
            unsigned short myOff2 = current_diagId - maxSize;
            current_locOffset     = current_j - myOff2;
        }

        if (matrix == 'E'){
            switch (E_ptr[diagOffset[current_diagId] + current_locOffset]){
                case '-' :
                    next_i = current_i;
                    next_j = current_j - 1;
                    break;
                case 'H' :
                    matrix = 'H';
                    gapOpen = 1;
                    next_i = current_i;
                    next_j = current_j - 1;
                    break;
            }
        } else if (matrix == 'F'){
            switch (F_ptr[diagOffset[current_diagId] + current_locOffset]){
                case '|' :
                    next_i = current_i - 1;
                    next_j = current_j;
                    break;
                case 'H' :
                    matrix = 'H';
                    gapOpen = 1;
                    next_i = current_i - 1;
                    next_j = current_j;
                    break;
            }
        } else if (matrix == 'H') { 
            switch (H_ptr[diagOffset[current_diagId] + current_locOffset]){
                case '\\' :
                    next_i = current_i - 1;
                    next_j = current_j - 1;
                    break;
                case '|' :
                    next_i = current_i - 1;
                    next_j = current_j;
                    break;
                case '-' :
                    next_i = current_i;
                    next_j = current_j - 1;
                    break;
            }
        } 


        //last_score = current_score;
        counter++;   
    }

    if (shorterSeq[current_j-1] == longerSeq[current_i-1]){
        longCIGAR[counter] = '=';
    } else {
        current_i=current_i+1;
        current_j=current_j+1;
    }
    
    //This section commented out because we don't need to record the start positions from traceback because
    //we have them already from the reverse traceback.
    //if(lengthSeqA < lengthSeqB){  
        //seqB_align_begin[myId] = current_i;
        //seqA_align_begin[myId] = current_j;
    //}else{
        //seqA_align_begin[myId] = current_i;
        //seqB_align_begin[myId] = current_j;
    //}
 

    if (myTId == 0){
      createCIGAR(longCIGAR, CIGAR, maxCIGAR, seqA, seqB, lengthShorterSeq, lengthLongerSeq, seqBShorter, first_j, current_j, first_i, current_i);
    }

    
    // *tick_out = tick;
}



__global__ void
gpu_bsw::sequence_dna_kernel(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, 
                    char* longCIGAR_array, char* CIGAR_array, char* H_ptr_array, char* E_ptr_array, char* F_ptr_array, 
                    int maxCIGAR, unsigned const maxMatrixSize, short matchScore, short misMatchScore, short startGap, short extendGap)
{
    int block_Id  = blockIdx.x;
    int thread_Id = threadIdx.x;
    short laneId = threadIdx.x%32;
    short warpId = threadIdx.x/32;

    unsigned lengthSeqA;
    unsigned lengthSeqB;
    // local pointers
    char*    seqA;
    char*    seqB;

    char* H_ptr, *E_ptr, *F_ptr;
    char* CIGAR, *longCIGAR;

    extern __shared__ char is_valid_array[];
    char*                  is_valid = &is_valid_array[0];

// setting up block local sequences and their lengths.
    if(block_Id == 0)
    {
        lengthSeqA = prefix_lengthA[0];
        lengthSeqB = prefix_lengthB[0];
        seqA       = seqA_array;
        seqB       = seqB_array;
    }
    else
    {
        lengthSeqA = prefix_lengthA[block_Id] - prefix_lengthA[block_Id - 1];
        lengthSeqB = prefix_lengthB[block_Id] - prefix_lengthB[block_Id - 1];
        seqA       = seqA_array + prefix_lengthA[block_Id - 1];
        seqB       = seqB_array + prefix_lengthB[block_Id - 1];
    }

    // what is the max length and what is the min length
    unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    unsigned minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

    H_ptr = H_ptr_array + (block_Id * maxMatrixSize);
    E_ptr = E_ptr_array + (block_Id * maxMatrixSize);
    F_ptr = F_ptr_array + (block_Id * maxMatrixSize);

    longCIGAR = longCIGAR_array + (block_Id * maxCIGAR);
    CIGAR = CIGAR_array + (block_Id * maxCIGAR);

    char* longer_seq;
    unsigned short* diagOffset = (unsigned short*) (&is_valid_array[3 * (minSize + 1) * sizeof(short)]);


// shared memory space for storing longer of the two strings

    memset(is_valid, 0, minSize);
    is_valid += minSize;
    memset(is_valid, 1, minSize);
    is_valid += minSize;
    memset(is_valid, 0, minSize);

    char myColumnChar;
    // the shorter of the two strings is stored in thread registers
    if(lengthSeqA < lengthSeqB)
    {
      if(thread_Id < lengthSeqA)
        myColumnChar = seqA[thread_Id];  // read only once
      longer_seq = seqB;
    }
    else
    {
      if(thread_Id < lengthSeqB)
        myColumnChar = seqB[thread_Id];
      longer_seq = seqA;
    }

    __syncthreads(); // this is required here so that complete sequence has been copied to shared memory

    int   i            = 1;
    int   j            = thread_Id;
    short thread_max   = 0; // to maintain the thread max score
    short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
    short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string
    int ind;

    if(thread_Id == 0 && block_Id == 0)
      printf("lengthSeqA = %d, lengthSeqB = %d, minSize = %d, maxSize = %d\n", lengthSeqA, lengthSeqB, minSize, maxSize);
    //set up the prefixSum for diagonal offset look up table for H_ptr, E_ptr, F_ptr
    int    locSum = 0;
     
    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++) 
    {
        
        int locDiagId = diag + 1;
        if(thread_Id == 0)
        {
            
            if(locDiagId <= minSize + 1)
            {
                locSum += locDiagId;
                diagOffset[locDiagId] = locSum;
            }
            else if(locDiagId > maxSize + 1)
            {
                locSum += (minSize + 1) - (locDiagId - (maxSize + 1));
                diagOffset[locDiagId] = locSum;
            }
            else
            {
                locSum += minSize + 1;
                diagOffset[locDiagId] = locSum;
            }
            diagOffset[lengthSeqA + lengthSeqB] = locSum + 2; //what is this for?
            
        }
    }
    
    __syncthreads(); //to make sure prefixSum is calculated before the threads start calculations.    


  //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
  //H, E and F)
    short _curr_H = 0, _curr_F = 0, _curr_E = 0;
    short _prev_H = 0, _prev_F = 0, _prev_E = 0;
    short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
    short _temp_Val = 0;

   __shared__ short sh_prev_E[32]; // one such element is required per warp
   __shared__ short sh_prev_H[32];
   __shared__ short sh_prev_prev_H[32];

   __shared__ short local_spill_prev_E[1024];// each threads local spill,
   __shared__ short local_spill_prev_H[1024];
   __shared__ short local_spill_prev_prev_H[1024];

    //unsigned      alignmentPad = 4 + (4 - totBytes % 4);

    __syncthreads(); // to make sure all shmem allocations have been initialized

    
    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {  // iterate for the number of anti-diagonals

        unsigned short diagId    = i + j;
        unsigned short locOffset = 0;
        if(diagId < maxSize + 1) 
        {
            locOffset = j;
        }
        else
        {
          unsigned short myOff = diagId - maxSize;
          locOffset            = j - myOff;
        }

        is_valid = is_valid - (diag < minSize || diag >= maxSize); //move the pointer to left by 1 if cnd true

	       _temp_Val = _prev_H; // value exchange happens here to setup registers for next iteration
	       _prev_H = _curr_H;
	       _curr_H = _prev_prev_H;
	       _prev_prev_H = _temp_Val;
         _curr_H = 0;

        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E = 0;

        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
        _curr_F = 0;


        if(laneId == 31)
        { // if you are the last thread in your warp then spill your values to shmem
          sh_prev_E[warpId] = _prev_E;
          sh_prev_H[warpId] = _prev_H;
		      sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(diag >= maxSize)
        { // if you are invalid in this iteration, spill your values to shmem
          local_spill_prev_E[thread_Id] = _prev_E;
          local_spill_prev_H[thread_Id] = _prev_H;
          local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
        }

        __syncthreads(); // this is needed so that all the shmem writes are completed.

        if(is_valid[thread_Id] && thread_Id < minSize)
        {
          unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));
          short fVal = _prev_F + extendGap;
          short hfVal = _prev_H + startGap;
          short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
          short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);
          short eVal=0, heVal = 0;

          if(diag >= maxSize) // when the previous thread has phased out, get value from shmem
          {
            eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
            heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
          }
          else
          {
            eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
            heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
          }

           if(warpId == 0 && laneId == 0) // make sure that values for lane 0 in warp 0 is not undefined
           {
              eVal = 0;
              heVal = 0;
           }
      		_curr_F = (fVal > hfVal) ? fVal : hfVal;
          F_ptr[maxMatrixSize+10] = '|';
          E_ptr[maxMatrixSize+10] = '-';
          H_ptr[maxMatrixSize+10] = '*';
          if (fVal > hfVal){
                F_ptr[diagOffset[diagId] + locOffset] = '|';
          } else {
                F_ptr[diagOffset[diagId] + locOffset] = 'H';
          }

      		_curr_E = (eVal > heVal) ? eVal : heVal;
          if (j!=0){
            if (eVal > heVal) {
              E_ptr[diagOffset[diagId] + locOffset] = '-';
            } else {
              E_ptr[diagOffset[diagId] + locOffset] = 'H';
            }
          }
          short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
          short final_prev_prev_H = 0;
          if(diag >= maxSize)
          {
            final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
          }
          else
          {
            final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
          }

          if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;
          short diag_score = final_prev_prev_H + ((longer_seq[i - 1] == myColumnChar) ? matchScore : misMatchScore);
          _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0, &ind);
          if (ind == 0) {
                H_ptr[diagOffset[diagId] + locOffset] = '\\';
            } else if (ind == 1) {
                H_ptr[diagOffset[diagId] + locOffset] = '|'; 
            } else if (ind == 2) {
                H_ptr[diagOffset[diagId] + locOffset] = '-'; 
            } else {
                H_ptr[diagOffset[diagId] + locOffset] = '*'; 
          }

          thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
          thread_max_j = (thread_max >= _curr_H) ? thread_max_j : thread_Id + 1;
          thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
          //if (block_Id == 0)
            //printf("i = %d, j = %d, _curr_H = %d, thread_max_i = %d, thread_max_j = %d, thread_max = %d\n",i, thread_Id, _curr_H, thread_max_i, thread_max_j, thread_max);
          i++;
       }
      __syncthreads(); // why do I need this? commenting it out breaks it
    }
    __syncthreads();

    thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j, minSize);  // thread 0 will have the correct values

    if(thread_Id == 0)
    {
        short current_i = thread_max_i;
        short current_j = thread_max_j;

        if(lengthSeqA < lengthSeqB)
        {
          seqB_align_end[block_Id] = thread_max_i;
          seqA_align_end[block_Id] = thread_max_j;
          top_scores[block_Id] = thread_max;
        }
        else
        {
        seqA_align_end[block_Id] = thread_max_i;
        seqB_align_end[block_Id] = thread_max_j;
        top_scores[block_Id] = thread_max;
        }

        //traceBack(current_i, current_j, seqA_array, seqB_array, prefix_lengthA, 
          //          prefix_lengthB, seqA_align_begin, seqA_align_end,
          //          seqB_align_begin, seqB_align_end, maxMatrixSize, maxCIGAR,
          //          longCIGAR, CIGAR, H_ptr, E_ptr, F_ptr, diagOffset);

    }
    __syncthreads();
}


__global__ void
gpu_bsw::sequence_dna_reverse(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, 
                    char* longCIGAR_array, char* CIGAR_array, char* H_ptr_array, char* E_ptr_array, char* F_ptr_array, 
                    int maxCIGAR, unsigned const maxMatrixSize, short matchScore, short misMatchScore, short startGap, short extendGap){

      int block_Id  = blockIdx.x;
      int thread_Id = threadIdx.x;
      short laneId = threadIdx.x%32;
      short warpId = threadIdx.x/32;
      // local pointers
      char*    seqA;
      char*    seqB;
      extern __shared__ char is_valid_array[];
      char*                  is_valid = &is_valid_array[0];
      char* longer_seq;

  // setting up block local sequences and their lengths.
      if(block_Id == 0)
      {
          seqA       = seqA_array;
          seqB       = seqB_array;
      }
      else
      {
          seqA       = seqA_array + prefix_lengthA[block_Id - 1];
          seqB       = seqB_array + prefix_lengthB[block_Id - 1];
      }


      int newlengthSeqA = seqA_align_end[block_Id];
      int newlengthSeqB = seqB_align_end[block_Id];

      unsigned maxSize = newlengthSeqA > newlengthSeqB ? newlengthSeqA : newlengthSeqB;
      unsigned minSize = newlengthSeqA < newlengthSeqB ? newlengthSeqA : newlengthSeqB;
      char myColumnChar;

      is_valid = &is_valid_array[0]; //reset is_valid array for second iter
      memset(is_valid, 0, minSize);
      is_valid += minSize;
      memset(is_valid, 1, minSize);
      is_valid += minSize;
      memset(is_valid, 0, minSize);
      __syncthreads(); // this is required because shmem writes


  //check if the new length of A is larger than B, if so then place the B string in registers and A in myLocString, make sure we dont do redundant copy by checking which string is located in myLocString before

      if(newlengthSeqA < newlengthSeqB)
      {
        if(thread_Id < newlengthSeqA)
        {
          myColumnChar = seqA[(newlengthSeqA - 1) - thread_Id];  // read only once
          longer_seq = seqB;
          }
      }
      else
      {
        if(thread_Id < newlengthSeqB)
        {
         myColumnChar = seqB[(newlengthSeqB - 1) - thread_Id];
           longer_seq = seqA;
        }
      }


      __syncthreads(); // this is required  because sequence has been re-written in shmem


      int   i            = 1;
      int   j            = thread_Id;
      short thread_max   = 0; // to maintain the thread max score
      short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
      short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string
      int ind;

    //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
    //H, E and F)
      short _curr_H = 0, _curr_F = 0, _curr_E = 0;
      short _prev_H = 0, _prev_F = 0, _prev_E = 0;
      short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
      short _temp_Val = 0;

     __shared__ short sh_prev_E[32]; // one such element is required per warp
     __shared__ short sh_prev_H[32];
     __shared__ short sh_prev_prev_H[32];

     __shared__ short local_spill_prev_E[1024];// each threads local spill,
     __shared__ short local_spill_prev_H[1024];
     __shared__ short local_spill_prev_prev_H[1024];


      __syncthreads(); // this is required because shmem has been updated
      for(int diag = 0; diag <  newlengthSeqA + newlengthSeqB - 1; diag++)
      {
        is_valid = is_valid - (diag < minSize || diag >= maxSize);

        _temp_Val = _prev_H;
      	_prev_H = _curr_H;
      	_curr_H = _prev_prev_H;
      	_prev_prev_H = _temp_Val;

      	_curr_H = 0;


        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E = 0;

        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
      	_curr_F = 0;

        if(laneId == 31)
        {
          sh_prev_E[warpId] = _prev_E;
          sh_prev_H[warpId] = _prev_H;
    		  sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(diag>= maxSize)
        { // if you are invalid in this iteration, spill your values to shmem
          local_spill_prev_E[thread_Id] = _prev_E;
          local_spill_prev_H[thread_Id] = _prev_H;
          local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
        }

        __syncthreads();

        if(is_valid[thread_Id] && thread_Id < minSize)
        {
      	    unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));
      	    short fVal = _prev_F + extendGap;
      	    short hfVal = _prev_H + startGap;
      	    short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
      	    short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);

            short eVal=0;
            short heVal = 0;
            if(diag >= maxSize)
            {
                eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
            }
            else
            {
              eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
            }

            if(diag >= maxSize)
            {
                heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
            }
            else
            {
              heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
            }


            if(warpId == 0 && laneId == 0)
            {
              eVal = 0;
              heVal = 0;
            }
            _curr_F = (fVal > hfVal) ? fVal : hfVal;
        		_curr_E = (eVal > heVal) ? eVal : heVal;
            short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
      	    short final_prev_prev_H =0;

            if(diag >= maxSize)
             {
               final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
             }
            else
             {
               final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
             }

            if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

            short diag_score = final_prev_prev_H + ((longer_seq[(maxSize - i )] == myColumnChar)? matchScore : misMatchScore);
            _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0, &ind);
            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : maxSize - i;
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : minSize - thread_Id -1;
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
            i++;
        }
          __syncthreads();
        }
        __syncthreads();

        thread_max = blockShuffleReduce_with_index_reverse(thread_max, thread_max_i, thread_max_j,  minSize);  // thread 0 will have the correct values
        if(thread_Id == 0)
        {
          if(newlengthSeqA < newlengthSeqB)
          {
            seqB_align_begin[block_Id] = (thread_max_i);
            seqA_align_begin[block_Id] = (thread_max_j);
          }
          else
          {
            seqA_align_begin[block_Id] = (thread_max_i);
            seqB_align_begin[block_Id] = (thread_max_j);
          }
        }
        __syncthreads();

}
__global__ void
gpu_bsw::sequence_aa_kernel(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix)
{
  int block_Id  = blockIdx.x;
  int thread_Id = threadIdx.x;
  short laneId = threadIdx.x%32;
  short warpId = threadIdx.x/32;

  unsigned lengthSeqA;
  unsigned lengthSeqB;
  // local pointers
  char*    seqA;
  char*    seqB;
  char* longer_seq;

  extern __shared__ char is_valid_array[];
  char*                  is_valid = &is_valid_array[0];

// setting up block local sequences and their lengths.
  if(block_Id == 0)
  {
      lengthSeqA = prefix_lengthA[0];
      lengthSeqB = prefix_lengthB[0];
      seqA       = seqA_array;
      seqB       = seqB_array;
  }
  else
  {
      lengthSeqA = prefix_lengthA[block_Id] - prefix_lengthA[block_Id - 1];
      lengthSeqB = prefix_lengthB[block_Id] - prefix_lengthB[block_Id - 1];
      seqA       = seqA_array + prefix_lengthA[block_Id - 1];
      seqB       = seqB_array + prefix_lengthB[block_Id - 1];
  }
  // what is the max length and what is the min length
  unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
  unsigned minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

// shared memory space for storing longer of the two strings
  memset(is_valid, 0, minSize);
  is_valid += minSize;
  memset(is_valid, 1, minSize);
  is_valid += minSize;
  memset(is_valid, 0, minSize);

  char myColumnChar;
  // the shorter of the two strings is stored in thread registers
  if(lengthSeqA < lengthSeqB)
  {
    if(thread_Id < lengthSeqA)
      myColumnChar = seqA[thread_Id];  // read only once
    longer_seq = seqB;
  }
  else
  {
    if(thread_Id < lengthSeqB)
      myColumnChar = seqB[thread_Id];
    longer_seq = seqA;
  }

  __syncthreads(); // this is required here so that complete sequence has been copied to shared memory

  int   i            = 1;
  int   j            = thread_Id;
  short thread_max   = 0; // to maintain the thread max score
  short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
  short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string
  int ind;

//initializing registers for storing diagonal values for three recent most diagonals (separate tables for
//H, E and F)
  short _curr_H = 0, _curr_F = 0, _curr_E = 0;
  short _prev_H = 0, _prev_F = 0, _prev_E = 0;
  short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
  short _temp_Val = 0;

 __shared__ short sh_prev_E[32]; // one such element is required per warp
 __shared__ short sh_prev_H[32];
 __shared__ short sh_prev_prev_H[32];

 __shared__ short local_spill_prev_E[1024];// each threads local spill,
 __shared__ short local_spill_prev_H[1024];
 __shared__ short local_spill_prev_prev_H[1024];

 __shared__ short sh_aa_encoding[ENCOD_MAT_SIZE];// length = 91
 __shared__ short sh_aa_scoring[SCORE_MAT_SIZE];

 int max_threads = blockDim.x;
 for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads){
    sh_aa_scoring[p] = scoring_matrix[p];
 }
 for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads){
   sh_aa_encoding[p] = encoding_matrix[p];
 }

  __syncthreads(); // to make sure all shmem allocations have been initialized

  for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
  {  // iterate for the number of anti-diagonals

      is_valid = is_valid - (diag < minSize || diag >= maxSize); //move the pointer to left by 1 if cnd true

       _temp_Val = _prev_H; // value exchange happens here to setup registers for next iteration
       _prev_H = _curr_H;
       _curr_H = _prev_prev_H;
       _prev_prev_H = _temp_Val;
       _curr_H = 0;

      _temp_Val = _prev_E;
      _prev_E = _curr_E;
      _curr_E = _prev_prev_E;
      _prev_prev_E = _temp_Val;
      _curr_E = 0;

      _temp_Val = _prev_F;
      _prev_F = _curr_F;
      _curr_F = _prev_prev_F;
      _prev_prev_F = _temp_Val;
      _curr_F = 0;


      if(laneId == 31)
      { // if you are the last thread in your warp then spill your values to shmem
        sh_prev_E[warpId] = _prev_E;
        sh_prev_H[warpId] = _prev_H;
        sh_prev_prev_H[warpId] = _prev_prev_H;
      }

      if(diag >= maxSize)
      { // if you are invalid in this iteration, spill your values to shmem
        local_spill_prev_E[thread_Id] = _prev_E;
        local_spill_prev_H[thread_Id] = _prev_H;
        local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
      }

      __syncthreads(); // this is needed so that all the shmem writes are completed.

      if(is_valid[thread_Id] && thread_Id < minSize)
      {
        unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));

        short fVal = _prev_F + extendGap;
        short hfVal = _prev_H + startGap;
        short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
        short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);

        short eVal=0, heVal = 0;

        if(diag >= maxSize) // when the previous thread has phased out, get value from shmem
        {
          eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
          heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
        }
        else
        {
          eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
          heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
        }

         if(warpId == 0 && laneId == 0) // make sure that values for lane 0 in warp 0 is not undefined
         {
            eVal = 0;
            heVal = 0;
         }
        _curr_F = (fVal > hfVal) ? fVal : hfVal;
        _curr_E = (eVal > heVal) ? eVal : heVal;

        short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
        short final_prev_prev_H = 0;
        if(diag >= maxSize)
        {
          final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
        }
        else
        {
          final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
        }


        if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

        short mat_index_q = sh_aa_encoding[(int)longer_seq[i-1]];//encoding_matrix
        short mat_index_r = sh_aa_encoding[(int)myColumnChar];

        short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

        short diag_score = final_prev_prev_H + add_score;

        _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0, &ind);
        thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
        thread_max_j = (thread_max >= _curr_H) ? thread_max_j : thread_Id + 1;
        thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
        i++;
     }

    __syncthreads(); // why do I need this? commenting it out breaks it

  }
  __syncthreads();

  thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j,
                                  minSize);  // thread 0 will have the correct values




  if(thread_Id == 0)
  {


      if(lengthSeqA < lengthSeqB)
      {
        seqB_align_end[block_Id] = thread_max_i;
        seqA_align_end[block_Id] = thread_max_j;
        top_scores[block_Id] = thread_max;
    }
      else
      {
      seqA_align_end[block_Id] = thread_max_i;
      seqB_align_end[block_Id] = thread_max_j;
      top_scores[block_Id] = thread_max;
      }

  }
}

__global__ void
gpu_bsw::sequence_aa_reverse(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix){

      int block_Id  = blockIdx.x;
      int thread_Id = threadIdx.x;
      short laneId = threadIdx.x%32;
      short warpId = threadIdx.x/32;
      // local pointers
      char*    seqA;
      char*    seqB;

      extern __shared__ char is_valid_array[];
      char*                  is_valid = &is_valid_array[0];
      char* longer_seq;

  // setting up block local sequences and their lengths.
      if(block_Id == 0)
      {
          seqA       = seqA_array;
          seqB       = seqB_array;
      }
      else
      {
          seqA       = seqA_array + prefix_lengthA[block_Id - 1];
          seqB       = seqB_array + prefix_lengthB[block_Id - 1];
      }
      int newlengthSeqA = seqA_align_end[block_Id];
      int newlengthSeqB = seqB_align_end[block_Id];
      unsigned maxSize = newlengthSeqA > newlengthSeqB ? newlengthSeqA : newlengthSeqB;
      unsigned minSize = newlengthSeqA < newlengthSeqB ? newlengthSeqA : newlengthSeqB;
      char myColumnChar;
      is_valid = &is_valid_array[0]; //reset is_valid array for second iter
      memset(is_valid, 0, minSize);
      is_valid += minSize;
      memset(is_valid, 1, minSize);
      is_valid += minSize;
      memset(is_valid, 0, minSize);
      __syncthreads(); // this is required because shmem writes

  //check if the new length of A is larger than B, if so then place the B string in registers and A in myLocString, make sure we dont do redundant copy by checking which string is located in myLocString before

      if(newlengthSeqA < newlengthSeqB)
      {
        if(thread_Id < newlengthSeqA)
        {
          myColumnChar = seqA[(newlengthSeqA - 1) - thread_Id];  // read only once
          longer_seq = seqB;
          }
      }
      else
      {
        if(thread_Id < newlengthSeqB)
        {
         myColumnChar = seqB[(newlengthSeqB - 1) - thread_Id];
           longer_seq = seqA;
        }
      }


      __syncthreads(); // this is required  because sequence has been re-written in shmem


      int   i            = 1;
      int   j            = thread_Id;
      short thread_max   = 0; // to maintain the thread max score
      short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
      short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string
      int ind;

    //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
    //H, E and F)
      short _curr_H = 0, _curr_F = 0, _curr_E = 0;
      short _prev_H = 0, _prev_F = 0, _prev_E = 0;
      short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
      short _temp_Val = 0;

     __shared__ short sh_prev_E[32]; // one such element is required per warp
     __shared__ short sh_prev_H[32];
     __shared__ short sh_prev_prev_H[32];

     __shared__ short local_spill_prev_E[1024];// each threads local spill,
     __shared__ short local_spill_prev_H[1024];
     __shared__ short local_spill_prev_prev_H[1024];

     __shared__ short sh_aa_encoding[ENCOD_MAT_SIZE];// length = 91
     __shared__ short sh_aa_scoring[SCORE_MAT_SIZE];

     int max_threads = blockDim.x;

     for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads){
       sh_aa_scoring[p] = scoring_matrix[p];
     }
     for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads){
       sh_aa_encoding[p] = encoding_matrix[p];
     }


      __syncthreads(); // this is required because shmem has been updated
      for(int diag = 0; diag <  newlengthSeqA + newlengthSeqB - 1; diag++)
      {
        is_valid = is_valid - (diag < minSize || diag >= maxSize);

        _temp_Val = _prev_H;
      	_prev_H = _curr_H;
      	_curr_H = _prev_prev_H;
      	_prev_prev_H = _temp_Val;

      	_curr_H = 0;


        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E = 0;

        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
      	_curr_F = 0;

        if(laneId == 31)
        {
          sh_prev_E[warpId] = _prev_E;
          sh_prev_H[warpId] = _prev_H;
    		  sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(diag>= maxSize)
        { // if you are invalid in this iteration, spill your values to shmem
          local_spill_prev_E[thread_Id] = _prev_E;
          local_spill_prev_H[thread_Id] = _prev_H;
          local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
        }

        __syncthreads();

        if(is_valid[thread_Id] && thread_Id < minSize)
        {

      	    unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));


      	    short fVal = _prev_F + extendGap;
      	    short hfVal = _prev_H + startGap;
      	    short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
      	    short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);

            short eVal=0;
            short heVal = 0;

            if(diag >= maxSize)
            {
                eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
            }
            else
            {
              eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
            }

            if(diag >= maxSize)
            {
                heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
            }
            else
            {
              heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
            }

            if(warpId == 0 && laneId == 0)
            {
              eVal = 0;
              heVal = 0;
            }
            _curr_F = (fVal > hfVal) ? fVal : hfVal;
        		_curr_E = (eVal > heVal) ? eVal : heVal;
            short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
      	    short final_prev_prev_H =0;

            if(diag >= maxSize)
             {
               final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
             }
            else
             {
               final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
             }

            if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

            short mat_index_q = sh_aa_encoding[(int)longer_seq[maxSize-i]];
            short mat_index_r = sh_aa_encoding[(int)myColumnChar];
            short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

            short diag_score = final_prev_prev_H + add_score;
            _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0, &ind);

            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : maxSize - i ;//i;// begin_A (longer string)
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : minSize - thread_Id -1; // begin_B (shorter string)
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
            i++;

        }
          __syncthreads();

        }
        __syncthreads();




        thread_max = blockShuffleReduce_with_index_reverse(thread_max, thread_max_i, thread_max_j,
                                        minSize);  // thread 0 will have the correct values
        if(thread_Id == 0)
        {
          if(newlengthSeqA < newlengthSeqB)
          {
            seqB_align_begin[block_Id] = /*newlengthSeqB - */(thread_max_i);
            seqA_align_begin[block_Id] = /*newlengthSeqA */ (thread_max_j);
          }
          else
          {
            seqA_align_begin[block_Id] = /*newlengthSeqA - */(thread_max_i);
            seqB_align_begin[block_Id] = /*newlengthSeqB -*/ (thread_max_j);
          }
        }
        __syncthreads();

}