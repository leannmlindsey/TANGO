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
    
   
   
    if ( beg_S != 0){
       
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
   
    if ( end_S != 0){
     
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
gpu_bsw::traceBack_8bit(short current_i, short current_j, char* seqA_array, char* seqB_array, unsigned* prefix_lengthA, 
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
                    //longCIGAR[counter] = (temp_H >> 4) & 1 == 1 ? '=' : 'X'; 
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
             //for (int i = 0; i <= counter; i++){
             //     printf("%c",longCIGAR[i]);
             // }
             //printf("\n");
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

__device__ void
gpu_bsw::traceBack_4bit(short current_i, short current_j, char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
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

    const char* longerSeq = lengthSeqA < lengthSeqB ? seqB : seqA;
    const char* shorterSeq = lengthSeqA < lengthSeqB ? seqA : seqB;
    unsigned lengthShorterSeq = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;
    unsigned lengthLongerSeq = lengthSeqA < lengthSeqB ? lengthSeqB : lengthSeqA;
    bool seqBShorter = lengthSeqA < lengthSeqB ? false : true;  //need to keep track of whether query or ref is shorter for I or D in CIGAR

    current_diagId    = current_i + current_j;
    current_locOffset = 0;
    if(current_diagId < maxSize)
    {
        current_locOffset = current_j;
    }
    else
    {
        unsigned short myOff = current_diagId - maxSize+1;
        current_locOffset    = current_j - myOff;
    }

    char temp_H;
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

    while(continueTrace && (current_i != 0) && (current_j !=0))
    {
       temp_H = H_ptr[diagOffset[current_diagId] + current_locOffset];

        //write the current value into longCIGAR then assign next_i
        if (matrix == 'H') {

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

        if (continueTrace != false){
          prev_i = current_i; //record current values in case this is the stop location
          prev_j = current_j;

          current_i = next_i;
          current_j = next_j;

          current_diagId    = current_i + current_j;
          current_locOffset = 0;

          if(current_diagId < maxSize)
          {
            current_locOffset = current_j;
          } else {
            unsigned short myOff2 = current_diagId - maxSize+1;
            current_locOffset     = current_j - myOff2;
          }
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
                    int maxCIGAR, unsigned const maxMatrixSize, 
                    short matchScore, short misMatchScore, short startGap, short extendGap)
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
  
    memset(is_valid, 0, minSize);
    is_valid += minSize;
 
    memset(is_valid, 1, binary_matrix_height);
    is_valid += binary_matrix_height;
   
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
    
    int finalcell;
    for (int cyc = 0; cyc <= (binary_matrix_min + binary_matrix_max+1)/minSize + 1; cyc++){
    int locDiagId = thread_Id+cyc*minSize;
       if (locDiagId < binary_matrix_min + binary_matrix_max ){
         if(locDiagId < binary_matrix_min){
           locSum = (locDiagId) * (locDiagId + 1)/2; //fill in upper left triangle in matrix
           diagOffset[locDiagId]= locSum;
         }
         else if (locDiagId > binary_matrix_max){
           int n = (binary_matrix_max+binary_matrix_min) - locDiagId-1;
           finalcell = (binary_matrix_max) * (binary_matrix_min); 
           locSum = finalcell - n*(n+1)/2; //fill in lower right triangle of the matrix
           diagOffset[locDiagId] = locSum;
         }
         else {
           locSum = ((binary_matrix_min)*(binary_matrix_min+1)/2) +(binary_matrix_min)*(locDiagId-binary_matrix_min);
           diagOffset[locDiagId] = locSum; //fill in constant diagonals of the matrix
         }
       }
     }
   
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
 
    for(int diag = 0; diag < (minSize + binary_matrix_height-1); diag++)
    {  // iterate for the number of anti-diagonals
    ///set up value exchange
       
        diagId = i/2 + j; 
  
        locOffset = 0;
        locOffset = (diagId < binary_matrix_max) ? j : j - (diagId - binary_matrix_max+1); //offset shrinks after you hit the max height of the matrix, +1 is because we index by 0
        is_valid = is_valid - 1; //move the pointer to left by 1
       
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
        if(thread_Id <= diag && diag < binary_matrix_height + thread_Id && thread_Id < minSize) 
        //if(is_valid[thread_Id] && thread_Id < minSize)
        {
          //unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));
          unsigned mask = __ballot_sync(__activemask(), (thread_Id <= diag && diag < binary_matrix_height + thread_Id && thread_Id < minSize));
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

          short eVal=0, heVal = 0, hdVal = 0;

          int mm_score = ((longer_seq[i] == myColumnChar) ? matchScore : misMatchScore);
         
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
         
      		_E1 = (eVal > heVal) ? eVal : heVal;
          if (j!=0) H_temp = (eVal > heVal) ? (H_temp = H_temp | 2) : (H_temp = H_temp & (~2));
         
          //short diag_score = hdVal + mm_score;
          _H1 = findMaxFour(hdVal, _F1, _E1, 0, &ind);
         
          // can all go to one line of code

          if (ind == 0) { // diagonal cell is max, set bits to 0b00001100
                H_temp = H_temp | 4;     // set bit 0b00000100
                H_temp = H_temp | 8;     // set bit 0b00001000
          
            } else if (ind == 1) {       // left cell is max, set bits to 0b00001000
                H_temp = H_temp & (~4);  // clear bit
                H_temp = H_temp | 8;     // set bit 0b00001000
                 
            } else if (ind == 2) {       // top cell is max, set bits to 0b00000100
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp | 4;     // set bit 0b00000100
                
            } else {                     // score is 0, set bits to 0b00000000
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp & (~4);  //clear bit
              
          }
          
          //Finished with calculations for i=even so move bits to upper half of H_temp 
          H_temp = H_temp << 4;
         
          //Check if the maximum score is at this location
          if (_H1 > thread_max) {
	  	      thread_max_i = i;
		        thread_max_j = thread_Id;
		        thread_max = _H1;
	        }
         
          //Increment i and repeat above for i=odd
          i++;
      
          // First get the values we have in our thread
          fVal = _F1 + extendGap;
          hfVal = _H1 + startGap;      

          eVal=0, heVal = 0, hdVal = 0;

          mm_score = ((longer_seq[i] == myColumnChar) ? matchScore : misMatchScore);
        
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
         
      		_E0 = (eVal > heVal) ? eVal : heVal;
          if (j!=0) H_temp = (eVal > heVal) ? (H_temp = H_temp | 2) : (H_temp = H_temp & (~2));
          
         
          _H0 = findMaxFour(hdVal, _F0, _E0, 0, &ind);
          
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
          
          //Finished with calculations for i=even so move bits to upper half of H_temp 
         
           H_ptr[diagOffset[diagId] + locOffset] =  H_temp;
           
          //Check if the maximum score is at this location
        
          if (_H0 > thread_max) {
	  	      thread_max_i = i;
		        thread_max_j = thread_Id;
		        thread_max = _H0;
	        }
         
          //Increment i 
          i++;
        }
        
      __syncthreads(); 
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
       
        gpu_bsw::traceBack_8bit(current_i, current_j, seqA_array, seqB_array, prefix_lengthA, 
                    prefix_lengthB, seqA_align_begin, seqA_align_end,
                    seqB_align_begin, seqB_align_end, maxMatrixSize, maxCIGAR,
                    longCIGAR, CIGAR, H_ptr, diagOffset);

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
    short matchScore = 3;
    short misMatchScore = -3;
    int block_Id  = blockIdx.x;
    int thread_Id = threadIdx.x;
    short laneId = threadIdx.x%32;
    short warpId = threadIdx.x/32;

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
  
    memset(is_valid, 0, minSize);
    is_valid += minSize;
 
    memset(is_valid, 1, binary_matrix_height);
    is_valid += binary_matrix_height;
   
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
    
    int finalcell;
    for (int cyc = 0; cyc <= (binary_matrix_min + binary_matrix_max+1)/minSize + 1; cyc++){
    int locDiagId = thread_Id+cyc*minSize;
       if (locDiagId < binary_matrix_min + binary_matrix_max ){
         if(locDiagId < binary_matrix_min){
           locSum = (locDiagId) * (locDiagId + 1)/2; //fill in upper left triangle in matrix
           diagOffset[locDiagId]= locSum;
         }
         else if (locDiagId > binary_matrix_max){
           int n = (binary_matrix_max+binary_matrix_min) - locDiagId-1;
           finalcell = (binary_matrix_max) * (binary_matrix_min); 
           locSum = finalcell - n*(n+1)/2; //fill in lower right triangle of the matrix
           diagOffset[locDiagId] = locSum;
         }
         else {
           locSum = ((binary_matrix_min)*(binary_matrix_min+1)/2) +(binary_matrix_min)*(locDiagId-binary_matrix_min);
           diagOffset[locDiagId] = locSum; //fill in constant diagonals of the matrix
         }
       }
     }
   
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
 
    unsigned short diagId, locOffset;
 
    for(int diag = 0; diag < (minSize + binary_matrix_height-1); diag++)
    {  // iterate for the number of anti-diagonals
    ///set up value exchange
       
        diagId = i/2 + j; 
  
        locOffset = 0;
        locOffset = (diagId < binary_matrix_max) ? j : j - (diagId - binary_matrix_max+1); //offset shrinks after you hit the max height of the matrix, +1 is because we index by 0
        is_valid = is_valid - 1; //move the pointer to left by 1
       
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
        if(thread_Id <= diag && diag < binary_matrix_height + thread_Id && thread_Id < minSize) 
        //if(is_valid[thread_Id] && thread_Id < minSize)
        {
          //unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));
          unsigned mask = __ballot_sync(__activemask(), (thread_Id <= diag && diag < binary_matrix_height + thread_Id && thread_Id < minSize));
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

          short eVal=0, heVal = 0, hdVal = 0;

          short mat_index_q = sh_aa_encoding[(int)longer_seq[i]];//encoding_matrix
          short mat_index_r = sh_aa_encoding[(int)myColumnChar];

          short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical
         
          if(diag >=binary_matrix_height) // when the previous thread has phased out, get value from shmem
          {
            eVal = local_spill_E3[thread_Id - 1] + extendGap;
            heVal = local_spill_H3[thread_Id - 1]+ startGap;
            hdVal = local_spill_H4[thread_Id - 1]+ add_score;
          }
          else
          {
            //potentially could combine these into  one if statement instead of 3
            eVal =((warpId !=0 && laneId == 0)?sh_E3[warpId-1]: vale3Shfl) + extendGap;
            heVal =((warpId !=0 && laneId == 0)?sh_H3[warpId-1]:valh3Shfl) + startGap;
            hdVal = ((warpId !=0 && laneId == 0)?sh_H4[warpId-1]:valh4Shfl) + add_score; 
          }

          if(warpId == 0 && laneId == 0){
            eVal = 0 + extendGap;
            heVal = 0 + startGap;
            hdVal = 0 + add_score;
          }


      		_F1 = (fVal > hfVal) ? fVal : hfVal;
          H_temp = (fVal > hfVal) ? ( H_temp = H_temp | 1) : (H_temp = H_temp & (~1)); //record F value in H_temp 0b00000001 or H_temp 0b00000000
         
      		_E1 = (eVal > heVal) ? eVal : heVal;
          if (j!=0) H_temp = (eVal > heVal) ? (H_temp = H_temp | 2) : (H_temp = H_temp & (~2));
         
          //short diag_score = hdVal + mm_score;
          _H1 = findMaxFour(hdVal, _F1, _E1, 0, &ind);
         
          // can all go to one line of code

          if (ind == 0) { // diagonal cell is max, set bits to 0b00001100
                H_temp = H_temp | 4;     // set bit 0b00000100
                H_temp = H_temp | 8;     // set bit 0b00001000
          
            } else if (ind == 1) {       // left cell is max, set bits to 0b00001000
                H_temp = H_temp & (~4);  // clear bit
                H_temp = H_temp | 8;     // set bit 0b00001000
                 
            } else if (ind == 2) {       // top cell is max, set bits to 0b00000100
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp | 4;     // set bit 0b00000100
                
            } else {                     // score is 0, set bits to 0b00000000
                H_temp = H_temp & (~8);  //clear bit
                H_temp = H_temp & (~4);  //clear bit
              
          }
          
          //Finished with calculations for i=even so move bits to upper half of H_temp 
          H_temp = H_temp << 4;
         
          //Check if the maximum score is at this location
          if (_H1 > thread_max) {
	  	      thread_max_i = i;
		        thread_max_j = thread_Id;
		        thread_max = _H1;
	        }
         
          //Increment i and repeat above for i=odd
          i++;
      
          // First get the values we have in our thread
          fVal = _F1 + extendGap;
          hfVal = _H1 + startGap;      

          eVal=0, heVal = 0, hdVal = 0;

          mat_index_q = sh_aa_encoding[(int)longer_seq[i]];//encoding_matrix
          mat_index_r = sh_aa_encoding[(int)myColumnChar];

          add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical
         
        
          if(diag >= binary_matrix_height) // when the previous thread has phased out, get value from shmem
          {
            eVal = local_spill_E2[thread_Id - 1] + extendGap;
            heVal = local_spill_H2[thread_Id - 1]+ startGap;
            hdVal = local_spill_H3[thread_Id - 1] + add_score;
          }
          else
          {
            //potentially could combine these into  one if statement instead of 3 comparisons
            eVal =((warpId !=0 && laneId == 0)?sh_E2[warpId-1]: vale2Shfl) + extendGap;
            heVal =((warpId !=0 && laneId == 0)?sh_H2[warpId-1]:valh2Shfl) + startGap;
            hdVal = ((warpId !=0 && laneId == 0)?sh_H3[warpId-1]:valh3Shfl)+ add_score; 
          }

          if(warpId == 0 && laneId == 0){
            eVal = 0 + extendGap;
            heVal = 0 + startGap;
            hdVal = 0 + add_score;
          }

      		_F0 = (fVal > hfVal) ? fVal : hfVal;
          H_temp = (fVal > hfVal) ? ( H_temp = H_temp | 1) : (H_temp = H_temp & (~1)); //record F value in H_temp 0b00000001 or H_temp 0b00000000
         
      		_E0 = (eVal > heVal) ? eVal : heVal;
          if (j!=0) H_temp = (eVal > heVal) ? (H_temp = H_temp | 2) : (H_temp = H_temp & (~2));
          
         
          _H0 = findMaxFour(hdVal, _F0, _E0, 0, &ind);
          
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
          
          //Finished with calculations for i=even so move bits to upper half of H_temp 
         
           H_ptr[diagOffset[diagId] + locOffset] =  H_temp;
           
          //Check if the maximum score is at this location
        
          if (_H0 > thread_max) {
	  	      thread_max_i = i;
		        thread_max_j = thread_Id;
		        thread_max = _H0;
	        }
         
          //Increment i 
          i++;
        }
        
      __syncthreads(); 
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
       
        gpu_bsw::traceBack_8bit(current_i, current_j, seqA_array, seqB_array, prefix_lengthA, 
                    prefix_lengthB, seqA_align_begin, seqA_align_end,
                    seqB_align_begin, seqB_align_end, maxMatrixSize, maxCIGAR,
                    longCIGAR, CIGAR, H_ptr, diagOffset);

    }
    __syncthreads();
}
