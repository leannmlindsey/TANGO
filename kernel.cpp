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
    short array[4];
    array[0] = first; //diag_score
    array[1] = second; //_curr_F
    array[2] = third; //_curr_E
    array[3] = fourth; // 0

    short max;

    // to make sure that if the H score is 0, then * will be put in the H_ptr matrix for correct termination of traceback
    if (array[0] > 0 ) {
      max = array[0];
      *ind = 0;
    } else {
      max = 0;
      *ind = 3;
    }

    for (int i=1; i<4; i++){
      if (array[i] > max){
        max = array[i];
        *ind = i;
      }
    }

    return max;

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
    short cigar_position = 0;

    short beg_S;
    short end_S; 

   

    if (seqBShorter){
        
        beg_S = lengthShorterSeq - first_j-1;
        end_S = last_j; 
        if(seqB[lengthShorterSeq-1] == ' ') beg_S = beg_S - 1; //to fix if the original sequence was odd and a placeholder ' ' was put in 
    } else {
        beg_S = lengthLongerSeq - first_i-1; 
        end_S = last_i;
        if(seqA[lengthLongerSeq-1] == ' ') beg_S = beg_S -1;
    }
    
   
     //printf("beg_S = %d, end_S = %d\n", beg_S, end_S);
    if ( beg_S != 0){
        //printf("inside the beg_S loop\n");
        CIGAR[0]='S';
        cigar_position++ ; 
        cigar_position = intToCharPlusWrite(beg_S, CIGAR, cigar_position);
    }

    int p = 0;
    while(longCIGAR[p] != '\0'){
       
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
    //printf("end_S before loop is %d\n", end_S);
    if ( end_S != 0){
      //printf("inside the beg_S loop\n");
        CIGAR[cigar_position]='S';
        cigar_position++ ; 
        cigar_position = intToCharPlusWrite(end_S, CIGAR, cigar_position);
    }    
    cigar_position--;
    char temp;
    //code to reverse the cigar by swapping i and length of cigar - i
    for(int i = 0; i<(cigar_position)/2+1;i++){
        temp = CIGAR[i]; 
        CIGAR[i]=CIGAR[cigar_position-i]; 
        CIGAR[cigar_position-i] = temp; 
    }
    
    CIGAR[cigar_position+1]='\0';
}

__device__ void
gpu_bsw::traceBack(short current_i, short current_j, char* seqA_array, char* seqB_array, unsigned* prefix_lengthA, 
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, unsigned const maxMatrixSize, int maxCIGAR,
                    char* longCIGAR, char* CIGAR, char* H_ptr, uint32_t* diagOffset)
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
    unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    unsigned minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;
   
    const char* longerSeq = lengthSeqA < lengthSeqB ? seqB : seqA; 
    const char* shorterSeq = lengthSeqA < lengthSeqB ? seqA : seqB; 
    unsigned lengthShorterSeq = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;  
    unsigned lengthLongerSeq = lengthSeqA < lengthSeqB ? lengthSeqB : lengthSeqA;    
    bool seqBShorter = lengthSeqA < lengthSeqB ? false : true;  //need to keep track of whether query or ref is shorter for I or D in CIGAR 

    int binary_matrix_height = (maxSize%2==0) ? (maxSize/2) : (maxSize/2 + 1);//binary matrix is half the height of maxSize
    int binary_matrix_min, binary_matrix_max;
    if (binary_matrix_height > minSize) { 
	    binary_matrix_min = minSize;
	    binary_matrix_max = binary_matrix_height;
    } else { //sometimes binary matrix ends up shorter than minSize, this fixes the diagOffset in that case
	    binary_matrix_min = binary_matrix_height;
	    binary_matrix_max = minSize;
    };
    current_diagId    = current_i/2 + current_j;
    current_locOffset = 0;
    current_locOffset = (current_diagId < binary_matrix_max) ? current_j : current_j - (current_diagId - binary_matrix_max+1);

    char temp_H;
    //if (myId == BLOCK_TO_PRINT) printf("VALUES received by traceback: current_i = %d, current_j = %d, current_diagId = %d, binary_matrix_max = %d\n", current_i, current_j, current_diagId, binary_matrix_max);
    temp_H = H_ptr[diagOffset[current_diagId] + current_locOffset];
   
    short next_i;
    short next_j;
    

    short first_j = current_j; //recording the first_j, first_i values for use in calculating S
    short first_i = current_i;
    char matrix = 'H'; //initialize with H 

    int counter = 0;
    short prev_i;
    short prev_j;
    bool continueTrace = true;

    //printf("first_j = %d, first_i = %d\n", first_j, first_i);

   
    while(continueTrace && (current_i != 0) && (current_j !=0))
    {   
      //printf("BEFORE SHIFT: current_i = %d, current_j = %d, current_diagId = %d, temp_H = %02x\n", current_i, current_j, current_diagId, temp_H);
       temp_H = H_ptr[diagOffset[current_diagId] + current_locOffset];
       if (current_i%2==0){
          //printf("i=%d and is EVEN: H_ptr = %02x\n", current_i, H_ptr[diagOffset[current_diagId] + current_locOffset]);
        
          temp_H = H_ptr[diagOffset[current_diagId] + current_locOffset] >> 4 ;
       } else {
          //printf("i=%d and is ODD: H_ptr = %02x\n", current_i, H_ptr[diagOffset[current_diagId] + current_locOffset]);

          temp_H = H_ptr[diagOffset[current_diagId] + current_locOffset] & 0x0f;
       }

       //if(myId == BLOCK_TO_PRINT) printf("AFTER SHIFT: current_i = %d, current_j = %d, current_diagId = %d, index = %d, temp_H = %02x, diag_Id = %d, prefix = %d, locOffset = %d\n", current_i, current_j, current_diagId, diagOffset[current_diagId] + current_locOffset, temp_H, current_diagId, diagOffset[current_diagId], current_locOffset);

        //write the current value into longCIGAR then assign next_i
        if (matrix == 'H') { 
            //printf("H\n");
            switch (temp_H & 0b00001100){    
                case 0b00001100 :
                    matrix = 'H';
                    longCIGAR[counter] = shorterSeq[current_j] == longerSeq[current_i] ? '=' : 'X';
                    counter++;
                    next_i = current_i - 1;
                    next_j = current_j - 1;
                    break;
                case 0b00001000 :
                    matrix = 'F';
                    next_i = current_i;
                    next_j = current_j;
                    break;
                case 0b00000100 :
                    matrix = 'E';
                    next_i = current_i;
                    next_j = current_j;
                    break;
                 case 0b00000000 : 
                    continueTrace = false;
                    break;
            }
        } else if (matrix == 'E'){
            //printf("E\n");
            switch (temp_H & 0b00000010){
                case 0b00000010 :
		                longCIGAR[counter] = seqBShorter ? 'I' : 'D';
                    counter++;
                    next_i = current_i;
                    next_j = current_j - 1;
                    break;
                case 0b00000000 :
                    matrix = 'H';
                    longCIGAR[counter] = seqBShorter ? 'I' : 'D';
                    counter++;
                    next_i = current_i;
                    next_j = current_j - 1;
                    break;
            }
        } else if (matrix == 'F'){
            //printf("F\n");
            switch (temp_H & 0b00000001){
                case 0b00000001 :
		                longCIGAR[counter] = seqBShorter ? 'D' : 'I';
                    counter++;
                    next_i = current_i - 1;
                    next_j = current_j;
                    break;
                case 0b00000000 :
                    matrix = 'H';
                    longCIGAR[counter] = seqBShorter ? 'D' : 'I';
                    counter++;
                    next_i = current_i - 1;
                    next_j = current_j;
                    break;
            }
        }
        // if(myId == BLOCK_TO_PRINT){
        //     for (int i = 0; i <= counter; i++){
        //          printf("%c",longCIGAR[i]);
        //      }
        //     printf("\n");
        // }
       

        if (continueTrace != false){
          prev_i = current_i; //record current values in case this is the stop location
          prev_j = current_j;

          current_i = next_i;
          current_j = next_j;

          //current_diagId    = current_i + current_j;
          current_diagId    = current_i/2 + current_j;
          //printf("current_i = %d, current_j = %d, current_diagId = %d\n", current_i, current_j, current_diagId);
          current_locOffset = 0;
         
          current_locOffset = (current_diagId < binary_matrix_max) ? current_j : current_j - (current_diagId - binary_matrix_max+1);
        
          
        }
  }
   //handle edge cases
   if ((current_i == 0) || (current_j == 0)) {

      if (shorterSeq[current_j] == longerSeq[current_i] ){
        longCIGAR[counter] = '=';
        longCIGAR[counter+1] = '\0';
        prev_j = current_j;
        prev_i = current_i;
      }
      else {
        longCIGAR[counter]='\0';
      }
   } else {
    longCIGAR[counter] = '\0';
   }
   current_i ++; current_j++; next_i ++; next_j ++; 

    if(lengthSeqA < lengthSeqB){  
        seqB_align_begin[myId] = prev_i;
        seqA_align_begin[myId] = prev_j;
    }else{
        seqA_align_begin[myId] = prev_i;
        seqB_align_begin[myId] = prev_j;
    }

    if (myTId == 0){
     gpu_bsw::createCIGAR(longCIGAR, CIGAR, maxCIGAR, seqA, seqB, lengthShorterSeq, lengthLongerSeq, seqBShorter, first_j, prev_j, first_i, prev_i);
    }
}

__global__ void
gpu_bsw::sequence_dna_kernel_traceback(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, 
                    char* longCIGAR_array, char* CIGAR_array, char* H_ptr_array, 
                    int maxCIGAR, unsigned const maxMatrixSize, short matchScore, short misMatchScore, short startGap, short extendGap)
{  //test
  
    int block_Id  = blockIdx.x;
    int thread_Id = threadIdx.x;
    short laneId = threadIdx.x%32;
    short warpId = threadIdx.x/32;

    //if(block_Id == BLOCK_TO_PRINT && thread_Id == 0)printf("BLOCK = %d\n", block_Id);

    //if(thread_Id == 0) printf("THREAD PRINTING = %d\n",THREAD_TO_PRINT);
    unsigned lengthSeqA;
    unsigned lengthSeqB;
    // local pointers
    char*    seqA;
    char*    seqB;

    char* H_ptr;
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

    longCIGAR = longCIGAR_array + (block_Id * maxCIGAR);
    CIGAR = CIGAR_array + (block_Id * maxCIGAR);

     
    char* longer_seq;
    
    char myColumnChar;
    // the shorter of the two strings is stored in thread registers
    char H_temp = 0;  //temp value of H stored in register until H, E and F are set then written to global; set all bits to 0 initially

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

    int binary_matrix_height = (maxSize%2==0) ? (maxSize/2) : (maxSize/2 + 1);//binary matrix is half the height of maxSize
    int binary_matrix_min, binary_matrix_max;
    if (binary_matrix_height > minSize) { 
	    binary_matrix_min = minSize;
	    binary_matrix_max = binary_matrix_height;
    } else { //sometimes binary matrix ends up shorter than minSize, this fixes the diagOffset in that case
	    binary_matrix_min = binary_matrix_height;
	    binary_matrix_max = minSize;
    };
    uint32_t* diagOffset = (uint32_t*) (&is_valid_array[3 * maxSize * sizeof(uint32_t)]);
    //if(thread_Id == THREAD_TO_PRINT)printf("minSize = %d, maxSize = %d, binary_matrix_height = %d, binary_matrix_min = %d, binary_matrix_max = %d\n", minSize, maxSize, binary_matrix_height, binary_matrix_min, binary_matrix_max);
// shared memory space for storing longer of the two strings
    //if(thread_Id==0)printf("is_valid = %p, 0's start at this location, minSize = %d\n",is_valid, minSize);
    memset(is_valid, 0, minSize);
    is_valid += minSize;
    //if(thread_Id==0)printf("is_valid = %p, 1's start at this location, binary_matrix_height = %d\n",is_valid, binary_matrix_height);
    memset(is_valid, 1, binary_matrix_height);
    is_valid += binary_matrix_height;
     //if(thread_Id==0)printf("is_valid = %p, 0's start at this location, 1s end\n",is_valid);
    memset(is_valid, 0, minSize);

    __syncthreads(); // this is required here so that complete sequence has been copied to shared memory

    int   i            = 0;
    int   j            = thread_Id;
    short thread_max   = 0; // to maintain the thread max score
    short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
    short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string
    int ind;

    //set up the prefixSum for diagonal offset look up table for H_ptr
    int    locSum = 0;
    


    //if (thread_Id ==0) {
	    //printf("# cycles = %d, minSize = %d, binary_matrix_min + binary_matrix_max = %d\n", (binary_matrix_min + binary_matrix_max+1)/minSize + 1, minSize, binary_matrix_min + binary_matrix_max);
    //};
    int finalcell;
    for (int cyc = 0; cyc <= (binary_matrix_min + binary_matrix_max+1)/minSize + 1; cyc++){
    int locDiagId = thread_Id+cyc*minSize;
       if (locDiagId < binary_matrix_min + binary_matrix_max ){
         if(locDiagId < binary_matrix_min){
           locSum = (locDiagId) * (locDiagId + 1)/2; //fill in upper left triangle in matrix
           diagOffset[locDiagId]= locSum;
	         //printf("LEFT CORNER runs from 0 to %d\n", binary_matrix_min-1);
           //printf("LEFT CORNER inside loop thread_Id = %d cyc = %d locSum = %d locDiagId = %d\n", thread_Id, cyc, locSum, locDiagId);
         }
         else if (locDiagId > binary_matrix_max){
           int n = (binary_matrix_max+binary_matrix_min) - locDiagId-1;
           finalcell = (binary_matrix_max) * (binary_matrix_min); 
           locSum = finalcell - n*(n+1)/2; //fill in lower right triangle of the matrix
           diagOffset[locDiagId] = locSum;
	         //printf("RIGHT CORNER runs from binary_matrix_max = %d to finalcell = %d\n",binary_matrix_max,  finalcell);
          //printf("RIGHT CORNER inside loop thread_Id = %d cyc = %d locSum = %d locDiagId = %d\n", thread_Id, cyc, locSum, locDiagId);
         }
         else {
           locSum = ((binary_matrix_min)*(binary_matrix_min+1)/2) +(binary_matrix_min)*(locDiagId-binary_matrix_min);
           diagOffset[locDiagId] = locSum; //fill in constant diagonals of the matrix
           //printf("MIDDLE SECTION inside loop thread_Id = %d cyc = %d locSum = %d locDiagId = %d\n", thread_Id, cyc, locSum, locDiagId);
         }
       }
     }
        // if(thread_Id == 0){
          //printf("TABLE 1 : cycles = %d, binary_matrix_height = %d, minSize = %d, maxSize = %d, finalcell = %d\n",(minSize + binary_matrix_height+1)/minSize + 1, binary_matrix_height, minSize, maxSize, (binary_matrix_height) * (minSize)+1 );
          //for(int b = 0; b< minSize + binary_matrix_height-1; b++) printf("%d ", diagOffset[b]);
          //printf("\n");
         //}

    //   for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++) 
    //  {
        
    //      int locDiagId = diag;
    //      if(thread_Id == 0)
    //      {
            
    //          if(locDiagId <= binary_matrix_min + 1)
    //          {
    //              locSum += locDiagId;
    //              diagOffset[locDiagId] = locSum;
    //          }
    //          else if(locDiagId > binary_matrix_max + 1)
    //          {
    //              locSum += (binary_matrix_min + 1) - (locDiagId - (binary_matrix_max + 1));
    //              diagOffset[locDiagId] = locSum;
    //          }
    //          else
    //          {
    //              locSum += binary_matrix_min + 1;
    //              diagOffset[locDiagId] = locSum;
    //          }
    //          diagOffset[lengthSeqA + lengthSeqB] = locSum + 2; //what is this for?
            
    //          //printf("diag = %d, diagOffset = %d\n", locDiagId, diagOffset[locDiagId]);
    //      }
    //  }
   
     __syncthreads(); //to make sure prefixSum is calculated before the threads start calculations.    

  //initializing registers for storing diagonal values for three recent most diagonals (separate tables for H, E, F
    short _H0 = 0, _F0 = -100, _E0 = -100; //-100 acts as neg infinity
    short _H1 = 0, _F1 = -100, _E1 = -100;
    short _H2 = 0, _F2 = -100, _E2 = -100;
    short _H3 = 0, _E3 = -100;
    short _H4 = 0;

  
   __shared__ short sh_E2[32];// one such element is required per warp
   __shared__ short sh_E3[32];
  
   __shared__ short sh_H2[32];
   __shared__ short sh_H3[32];
   __shared__ short sh_H4[32];


   __shared__ short local_spill_E2[1024];// each threads local spill,
   __shared__ short local_spill_E3[1024];
  
   __shared__ short local_spill_H2[1024];
   __shared__ short local_spill_H3[1024];
   __shared__ short local_spill_H4[1024];


    __syncthreads(); // to make sure all shmem allocations have been initialized
 
    unsigned short diagId, locOffset;
    //if(thread_Id == 0 )printf("Number of diagonals to print = %d, minSize = %d, binary_matrix_height = %d\n", (minSize + binary_matrix_height-1), minSize, binary_matrix_height);
    for(int diag = 0; diag < (minSize + binary_matrix_height-1); diag++)
    {  // iterate for the number of anti-diagonals
    ///set up value exchange
       
        diagId = i/2 + j; 
        //if(thread_Id == THREAD_TO_PRINT) printf("diag= %d, i = %d, i/2 = %d, j = %d\n", diag, i, i/2, j);
        locOffset = 0;
        locOffset = (diagId < binary_matrix_max) ? j : j - (diagId - binary_matrix_max+1); //offset shrinks after you hit the max height of the matrix, +1 is because we index by 0
        is_valid = is_valid - 1; //move the pointer to left by 1
        //if(thread_Id == 0) printf("diag = %d, is_valid = %p, is_valid[0] = %d\n", diag, is_valid, is_valid[0]);
       
        H_temp = 0;
	      // value exchange happens here (iterating by 2) to setup registers for next iteration
	      _H4 = _H2;
        _H3 = _H1;
        _H2 = _H0;
        _H1 = 0;
        _H0 = 0;
      
        _E3 = _E1;
        _E2 = _E0;
        _E1 = -100;
        _E0 = -100;
       
        _F2 = _F0;
        _F1 = -100;
        _F0 = -100;

        if(laneId == 31)
        { // if you are the last thread in your warp then spill your values to shmem
        
          sh_E2[warpId] = _E2;
          sh_E3[warpId] = _E3;
      
		      sh_H2[warpId] = _H2;
          sh_H3[warpId] = _H3;
		      sh_H4[warpId] = _H4;
        }

        if(diag >= binary_matrix_height)
        { // if you are invalid in this iteration, spill your values to shmem
        
          local_spill_E2[thread_Id] = _E2;
          local_spill_E3[thread_Id] = _E3;
       
          local_spill_H2[thread_Id] = _H2;
          local_spill_H3[thread_Id] = _H3;
          local_spill_H4[thread_Id] = _H4;
        }

        __syncthreads(); // this is needed so that all the shmem writes are completed.
    
        if(is_valid[thread_Id] && thread_Id < minSize)
        {
          //printf("thread_Id = %d, i=%d\n", thread_Id, i);
          unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));

          // Calculations for i = even
          // First get the values we need from our own thread
          short fVal = _F2 + extendGap;
          short hfVal = _H2 + startGap;
          // Then use register shuffle to get the values from the previous thread
          short vale2Shfl = __shfl_sync(mask, _E2, laneId - 1, 32);
          short vale3Shfl = __shfl_sync(mask, _E3, laneId - 1, 32);
          short valh2Shfl = __shfl_sync(mask, _H2, laneId - 1, 32);
          short valh3Shfl = __shfl_sync(mask, _H3, laneId - 1, 32);
          short valh4Shfl = __shfl_sync(mask, _H4, laneId - 1, 32);

          //if(thread_Id == THREAD_TO_PRINT) printf("h2=%d, h3=%d, h4=%d\n", valh2Shfl, valh3Shfl, valh4Shfl);
         
          short eVal=0, heVal = 0, hdVal = 0;

          int mm_score = ((longer_seq[i] == myColumnChar) ? matchScore : misMatchScore);
         
          //if(thread_Id == THREAD_TO_PRINT) printf("%c, %c, %d, ", longer_seq[i],myColumnChar,mm_score );
          if(diag >=binary_matrix_height) // when the previous thread has phased out, get value from shmem
          {
            eVal = local_spill_E3[thread_Id - 1] + extendGap;
            heVal = local_spill_H3[thread_Id - 1]+ startGap;
            hdVal = local_spill_H4[thread_Id - 1]+ mm_score;
          }
          else
          {
            //potentially could combine these into  one if statement instead of 3
            eVal =((warpId !=0 && laneId == 0)?sh_E3[warpId-1]: vale3Shfl) + extendGap;
            heVal =((warpId !=0 && laneId == 0)?sh_H3[warpId-1]:valh3Shfl) + startGap;
            hdVal = ((warpId !=0 && laneId == 0)?sh_H4[warpId-1]:valh4Shfl) + mm_score; 
          }

          if(warpId == 0 && laneId == 0){
            eVal = 0 + extendGap;
            heVal = 0 + startGap;
            hdVal = 0 + mm_score;
          }


      		_F1 = (fVal > hfVal) ? fVal : hfVal;
          H_temp = (fVal > hfVal) ? ( H_temp = H_temp | 1) : (H_temp = H_temp & (~1)); //record F value in H_temp 0b00000001 or H_temp 0b00000000
          //if (thread_Id == 0) printf("Calculating _F1 - fVal = %d, hfVal = %d, _F1 = %d\n", fVal, hfVal, _F1);

      		_E1 = (eVal > heVal) ? eVal : heVal;
          if (j!=0) H_temp = (eVal > heVal) ? (H_temp = H_temp | 2) : (H_temp = H_temp & (~2));
          //if (thread_Id == 0) printf("Calculating _E1 - eVal = %d, heVal = %d, _E1 = %d\n", eVal, heVal, _E1);
          //short diag_score = hdVal + mm_score;
          _H1 = findMaxFour(hdVal, _F1, _E1, 0, &ind);
          //if (thread_Id == THREAD_TO_PRINT) printf("Calculating _H1 - _F1 = %d, _E1 = %d, diag_score = %d, _H1 = %d\n", _F1, _E1, hdVal, _H1);
          //if (thread_Id == THREAD_TO_PRINT && block_Id == BLOCK_TO_PRINT) printf("i = %d, %d\n",i, _H1);
          // can all go to one line of code

          if (ind == 0) { // diagonal cell is max, set bits to 0b00001100
                H_temp = H_temp | 4;     // set bit 0b00000100
                H_temp = H_temp | 8;     // set bit 0b00001000
                //printf("\\");
            } else if (ind == 1) {       // left cell is max, set bits to 0b00001000
                H_temp = H_temp & (~4);  // clear bit
                H_temp = H_temp | 8;     // set bit 0b00001000
                 //printf("-");
            } else if (ind == 2) {       // top cell is max, set bits to 0b00000100
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp | 4;     // set bit 0b00000100
                 //printf("|");
            } else {                     // score is 0, set bits to 0b00000000
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp & (~4);  //clear bit
                 //printf("*");
          }
          //if(thread_Id == THREAD_TO_PRINT) printf("H_temp before shift = %02x, i = %d, j = %d \n", H_temp, i, thread_Id);
          //Finished with calculations for i=even so move bits to upper half of H_temp 
          H_temp = H_temp << 4;
          //if(thread_Id == THREAD_TO_PRINT) printf("H_temp after shift = %02x, i = %d, j = %d \n", H_temp, i, thread_Id);
          //Check if the maximum score is at this location
          if (_H1 > thread_max) {
	  	      thread_max_i = i;
		        thread_max_j = thread_Id;
		        thread_max = _H1;
	        }
          //if(thread_Id == THREAD_TO_PRINT) printf("thread_Id = %d, _H1 = %d, i = %d, thread_max= %d\n", thread_Id, _H1, i, thread_max);
          //Increment i and repeat above for i=odd
          i++;
      
          // First get the values we have in our thread
          fVal = _F1 + extendGap;
          hfVal = _H1 + startGap;      

          eVal=0, heVal = 0, hdVal = 0;

          mm_score = ((longer_seq[i] == myColumnChar) ? matchScore : misMatchScore);
          //if(thread_Id == THREAD_TO_PRINT) printf("%c, %c, %d, ", longer_seq[i],myColumnChar,mm_score );
          if(diag >= binary_matrix_height) // when the previous thread has phased out, get value from shmem
          {
            eVal = local_spill_E2[thread_Id - 1] + extendGap;
            heVal = local_spill_H2[thread_Id - 1]+ startGap;
            hdVal = local_spill_H3[thread_Id - 1] + mm_score;
          }
          else
          {
            //potentially could combine these into  one if statement instead of 3 comparisons
            eVal =((warpId !=0 && laneId == 0)?sh_E2[warpId-1]: vale2Shfl) + extendGap;
            heVal =((warpId !=0 && laneId == 0)?sh_H2[warpId-1]:valh2Shfl) + startGap;
            hdVal = ((warpId !=0 && laneId == 0)?sh_H3[warpId-1]:valh3Shfl)+ mm_score; 
          }

          if(warpId == 0 && laneId == 0){
            eVal = 0 + extendGap;
            heVal = 0 + startGap;
            hdVal = 0 + mm_score;
          }

      		_F0 = (fVal > hfVal) ? fVal : hfVal;
          H_temp = (fVal > hfVal) ? ( H_temp = H_temp | 1) : (H_temp = H_temp & (~1)); //record F value in H_temp 0b00000001 or H_temp 0b00000000
          //if (thread_Id == 0) printf("Calculating _F0 - fVal = %d, hfVal = %d, _F0 = %d\n", fVal, hfVal, _F1);
      		_E0 = (eVal > heVal) ? eVal : heVal;
          if (j!=0) H_temp = (eVal > heVal) ? (H_temp = H_temp | 2) : (H_temp = H_temp & (~2));
           //if (thread_Id == 0) printf("Calculating _E0 - eVal = %d, heVal = %d, _E0 = %d\n", eVal, heVal, _E0);
         
          _H0 = findMaxFour(hdVal, _F0, _E0, 0, &ind);
          //if (thread_Id == THREAD_TO_PRINT) printf("Calculating _H1 - _F1 = %d, _E1 = %d, diag_score = %d, _H1 = %d\n", _F1, _E1, hdVal, _H1);
          //if (thread_Id == THREAD_TO_PRINT) printf("i = %d, %d\n",i, _H1);
          // can all go to one line of code

          if (ind == 0) { // diagonal cell is max, set bits to 0b00001100
                H_temp = H_temp | 4;     // set bit 0b00000100
                H_temp = H_temp | 8;     // set bit 0b00001000
                //printf("\\");
            } else if (ind == 1) {       // left cell is max, set bits to 0b00001000
                H_temp = H_temp & (~4);  // clear bit
                H_temp = H_temp | 8;     // set bit 0b00001000
                 //printf("-");
            } else if (ind == 2) {       // top cell is max, set bits to 0b00000100
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp | 4;     // set bit 0b00000100
                 //printf("|");
            } else {                     // score is 0, set bits to 0b00000000
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp & (~4);  //clear bit
                 //printf("*");
          }
          //if(thread_Id == THREAD_TO_PRINT) printf("H_temp before shift = %02x, i = %d, j = %d \n", H_temp, i, thread_Id);
          //if (thread_Id == THREAD_TO_PRINT  && block_Id == BLOCK_TO_PRINT) printf ("thread_max = %d, i = %d, j = %d\n", thread_max, thread_max_i, thread_max_j);
          //Finished with calculations for i=even so move bits to upper half of H_temp 
          //if(diagOffset[diagId] + locOffset > finalcell) printf("The index is outside the boundaries. block = %d, thread = %d, diagId = %d, prefix = %d, locOffset = %d, index = %d, finalcell = %d\n",block_Id, thread_Id, diagId, diagOffset[diagId],locOffset, diagOffset[diagId] + locOffset, finalcell);
           H_ptr[diagOffset[diagId] + locOffset] =  H_temp;
           //if(thread_Id == THREAD_TO_PRINT) printf("i = %d, j = %d, H_temp = %02x, index = %d, diagId = %d, prefix = %d, locOffset = %d, binary_matrix_max = %d\n", i, thread_Id, H_temp, diagOffset[diagId] + locOffset, diagId, diagOffset[diagId], locOffset, binary_matrix_max);
          //Check if the maximum score is at this location
        
          if (_H0 > thread_max) {
	  	      thread_max_i = i;
		        thread_max_j = thread_Id;
		        thread_max = _H0;
	        }
          //if(thread_Id == THREAD_TO_PRINT) printf("thread_Id = %d, _H0 = %d, thread_max= %d\n", thread_Id, _H0, thread_max);
          
          //Increment i 
          i++;
        }
        
      __syncthreads(); 
    }
    __syncthreads();
    //printf("thread_Id = %d, i = %d, thread_max_j = %d, thread_max_i = %d, thread_max= %d\n", thread_Id, i, thread_max_j, thread_max_i, thread_max);
    thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j, minSize);  // thread 0 will have the correct values

    
    
    if(thread_Id == 0)
    {
        
        short current_i = thread_max_i;
        short current_j = thread_max_j;
        //if(block_Id == BLOCK_TO_PRINT) printf("thread_max = %d, current_i = %d, current_j = %d\n", thread_max, current_i, current_j);
      
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
        //if(block_Id == BLOCK_TO_PRINT && thread_Id == 0) printf("DATA being passed to traceback: current_i = %d, current_j = %d, thread_max = %d\n",current_i, current_j, thread_max);
        gpu_bsw::traceBack(current_i, current_j, seqA_array, seqB_array, prefix_lengthA, 
                    prefix_lengthB, seqA_align_begin, seqA_align_end,
                    seqB_align_begin, seqB_align_end, maxMatrixSize, maxCIGAR,
                    longCIGAR, CIGAR, H_ptr, diagOffset);

        //if(block_Id == BLOCK_TO_PRINT) {
          //printf("CIGAR: ");
          //for(int b = 0; b < 100; b++){
            //printf("%c",CIGAR[b]);
          //}
          //printf("\n");
        //}
    }
    __syncthreads();
}

__global__ void
gpu_bsw::sequence_aa_kernel_traceback(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, char* longCIGAR_array, 
                    char* CIGAR_array, char* H_ptr_array, int maxCIGAR, unsigned const maxMatrixSize, 
                    short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix)
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

  char* H_ptr;
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

  longCIGAR = longCIGAR_array + (block_Id * maxCIGAR);
  CIGAR = CIGAR_array + (block_Id * maxCIGAR);

  uint32_t* diagOffset = (uint32_t*) (&is_valid_array[3 * (minSize + 1) * sizeof(uint32_t)]);

 
// shared memory space for storing longer of the two strings
  memset(is_valid, 0, minSize);
  is_valid += minSize;
 
  memset(is_valid, 1, minSize);
  is_valid += minSize;
  
  memset(is_valid, 0, minSize);
   

  char myColumnChar;
  char H_temp = 0;
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

  int   i            = 0;
  int   j            = thread_Id;
  short thread_max   = 0; // to maintain the thread max score
  short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
  short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string
  int ind;

  //set up the prefixSum for diagonal offset look up table for H_ptr, E_ptr, F_ptr
  int    locSum = 0;
  
  for (int cyc = 0; cyc <= (lengthSeqA + lengthSeqB+1)/minSize + 1; cyc++){
      
      int locDiagId = thread_Id+cyc*minSize;
      if (locDiagId < lengthSeqA + lengthSeqB ){
        if(locDiagId <= minSize){
          locSum = (locDiagId) * (locDiagId + 1)/2;
          diagOffset[locDiagId]= locSum;
          //printf("LEFT CORNER inside loop thread_Id = %d cyc = %d locSum = %d locDiagId = %d\n", thread_Id, cyc, locSum, locDiagId);
        }
        else if (locDiagId > maxSize + 1){
          int n = (maxSize+minSize) - locDiagId-1;
          int finalcell = (maxSize) * (minSize)+1;
          locSum = finalcell - n*(n+1)/2;
          diagOffset[locDiagId] = locSum;
          //printf("RIGHT CORNER inside loop thread_Id = %d cyc = %d locSum = %d locDiagId = %d\n", thread_Id, cyc, locSum, locDiagId);
        }
        else {
          locSum = ((minSize)*(minSize+1)/2) +(minSize)*(locDiagId-minSize);
          diagOffset[locDiagId] = locSum;
          //printf("MIDDLE SECTION inside loop thread_Id = %d cyc = %d locSum = %d locDiagId = %d\n", thread_Id, cyc, locSum, locDiagId);
        }
      }
    }
     
    
    __syncthreads(); //to make sure prefixSum is calculated before the threads start calculations.  


//initializing registers for storing diagonal values for three recent most diagonals (separate tables for
//H, E and F)
  short _curr_H = 0, _curr_F = -100, _curr_E = -100;
  short _prev_H = 0, _prev_F = -100, _prev_E = -100;
  short _prev_prev_H = 0, _prev_prev_F = -100, _prev_prev_E = -100;
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

      unsigned short diagId    = i + j;
      unsigned short locOffset = 0;
      if(diagId < maxSize) 
      {
          locOffset = j;
      }
      else
      {
        unsigned short myOff = diagId - maxSize+1;
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
      _curr_E = -100;

      _temp_Val = _prev_F;
      _prev_F = _curr_F;
      _curr_F = _prev_prev_F;
      _prev_prev_F = _temp_Val;
      _curr_F = -100;


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

        if (fVal > hfVal){
                H_temp = H_temp | 1;
        } else {
                H_temp = H_temp & (~1);
        }

        _curr_E = (eVal > heVal) ? eVal : heVal;

        if (j!=0){
            if (eVal > heVal) {
              H_temp = H_temp | 2;
            } else {
              H_temp = H_temp & (~2);
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

        short mat_index_q = sh_aa_encoding[(int)longer_seq[i]];//encoding_matrix
        short mat_index_r = sh_aa_encoding[(int)myColumnChar];

        short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

        short diag_score = final_prev_prev_H + add_score;

        _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0, &ind);

        switch (ind) {
              case 0: // diagonal cell is max, set bits to 0b00001100
                H_temp |= 4; // set bit 0b00000100
                H_temp |= 8; // set bit 0b00001000
                //printf("\\");
                break;
              case 1: // left cell is max, set bits to 0b00001000
                H_temp &= (~4); // clear bit
                H_temp |= 8; // set bit 0b00001000
                //printf("-");
                break;
              case 2: // top cell is max, set bits to 0b00000100
                H_temp &= (~8); //clear bit
                H_temp |= 4; // set bit 0b00000100
                //printf("|");
                break;
              default: // score is 0, set bits to 0b00000000
                H_temp &= (~8); //clear bit
                H_temp &= (~4); //clear bit
                //printf("*");
                break;
            }
	H_ptr[diagOffset[diagId] + locOffset] =  H_temp;

        thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
        thread_max_j = (thread_max >= _curr_H) ? thread_max_j : thread_Id;
        thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
        i++;
     }

    __syncthreads(); 

  }
  __syncthreads();

  thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j,
                                  minSize);  // thread 0 will have the correct values

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

      gpu_bsw::traceBack(current_i, current_j, seqA_array, seqB_array, prefix_lengthA, 
                    prefix_lengthB, seqA_align_begin, seqA_align_end,
                    seqB_align_begin, seqB_align_end, maxMatrixSize, maxCIGAR,
                    longCIGAR, CIGAR, H_ptr, diagOffset);


  }
  __syncthreads();
}


