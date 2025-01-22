
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <stdlib.h>
#include <iostream>

#include <fstream>
#include <sstream>
#include <time.h>
#include <chrono>

#define MAX_BLOCK_SIZE 1024 // 每個 block 的最大執行緒數量

// 計算 Hamming 距離的 Kernel
__global__ void computeHammingDistances(
    const unsigned long long* bitText,       // 位元串陣列
    const int* posTemp,            // 字串位置陣列
    int numberInputString,         // 總字串數量
    int parameterL,                // 漢明距離的參數 L
    unsigned long long bitReference,         // 參考字串的位元串
    unsigned long long clearWitnessBit,
    unsigned long long keepWitnessBit,
    int* distance                  // 輸出：每個字串的漢明距離
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberInputString) return; // 超出範圍則退出

    // 計算字串的位元串
    int aux1 = numberSymbolInUll - posTemp[idx] % numberSymbolInUll;
    int aux2 = posTemp[idx] / numberSymbolInUll;
    unsigned long long bitString;

    if (aux1 >= parameterL) {
        bitString = (bitText[aux2] >> ((aux1 - parameterL) * numberBitForSymbol)) & filterUselessBit;
    }
    else {
        bitString = ((bitText[aux2] << ((parameterL - aux1) * numberBitForSymbol)) |
            (bitText[aux2 + 1] >> ((numberSymbolInUll - parameterL + aux1) * numberBitForSymbol))) &
            filterUselessBit;
    }

    // 漢明距離計算 (popcount 方法) 
    distance[idx] = __popcll(((bitString ^ bitReference) + clearWitnessBit)& keepWitnessBit);
}

__global__ void countingSort(
    const int* distance,  // 輸入：漢明距離陣列
    int* counting,        // 輸出：計數排序表
    int numberInputString,
    int parameterL
) {
    __shared__ int localCounting[128]; // 共享記憶體（根據最大 L 設置）
    int idx = threadIdx.x;

    // 初始化共享記憶體
    if (idx < parameterL + 1) localCounting[idx] = 0;
    __syncthreads();

    // 累計漢明距離的出現次數
    for (int i = threadIdx.x; i < numberInputString; i += blockDim.x) {
        atomicAdd(&localCounting[distance[i]], 1);
    }
    __syncthreads();

    // 將結果寫回全域記憶體
    if (idx < parameterL + 1) atomicAdd(&counting[idx], localCounting[idx]);
}

void parallelGroupStrings(
    const uint64_t* bitText,
    const int* posTemp,
    int numberInputString,
    int parameterL,
    uint64_t bitReference
) {
    int* d_distance, * d_counting;
    uint64_t* d_bitText;
    int* d_posTemp;

    // 記憶體分配
    cudaMalloc(&d_distance, sizeof(int) * numberInputString);
    cudaMalloc(&d_counting, sizeof(int) * (parameterL + 1));
    cudaMalloc(&d_bitText, sizeof(uint64_t) * textSize);
    cudaMalloc(&d_posTemp, sizeof(int) * numberInputString);

    cudaMemcpy(d_bitText, bitText, sizeof(uint64_t) * textSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posTemp, posTemp, sizeof(int) * numberInputString, cudaMemcpyHostToDevice);
    cudaMemset(d_counting, 0, sizeof(int) * (parameterL + 1));

    // 設定執行緒和區塊數量
    int blockSize = 256;
    int numBlocks = (numberInputString + blockSize - 1) / blockSize;

    // 啟動 CUDA 核心
    computeHammingDistances << <numBlocks, blockSize >> > (d_bitText, d_posTemp, numberInputString, parameterL, bitReference, d_distance);
    countingSort << <1, 128 >> > (d_distance, d_counting, numberInputString, parameterL);

    // 複製結果回主機
    int* counting = new int[parameterL + 1];
    cudaMemcpy(counting, d_counting, sizeof(int) * (parameterL + 1), cudaMemcpyDeviceToHost);

    // 清理記憶體
    cudaFree(d_distance);
    cudaFree(d_counting);
    cudaFree(d_bitText);
    cudaFree(d_posTemp);

    // 輸出計數表
    for (int i = 0; i <= parameterL; i++) {
        printf("Distance %d: %d\n", i, counting[i]);
    }
    delete[] counting;
}


/* Define the maximum size of alphabet */
#define MAX_ALPHABET_SIZE 255
/* Define boolean values */
#define true 1
#define false 0

/* Define the abbreviated names */
typedef unsigned char UC;
typedef unsigned long long ULL;

/* Define the struct for a node in the reference tree */
/* Note that this struct is for internal and leaf nodes */
typedef struct _REFERENCE_TREE_NODE
{
    int numberReference;                     /* The number of input strings */
    int reference;                           /* The reference string */
    bool order;                              /* The flag for accessing memory */
    UC children;                             /* The number of child nodes (which will be expanded)*/
    UC* distance;                            /* The child nodes' distances (with the reference string) */
    struct _REFERENCE_TREE_NODE** childNode; /* pointers for child nodes */
} ReferenceTreeNode;

/* Define the struct for an element in job stack */
typedef struct _STACK
{
    int numberInputString;                   /* The number of input strings */
    int start;                               /* The start position for the memory */
    bool order;                              /* The flag for accessing memory */
    void* pointerNode;                       /* The child node needed to be expaned */
    struct _STACK* previous;                 /* The previous element in the job stack */
    struct _STACK* next;                     /* The next element in the job stack */
} Stack;

/* Calculate the number of bits whose values are 1 in a computer word */
int ComputeHD_64(ULL i)
{
    i = i - ((i >> 1) & 0x5555555555555555UL);
    i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
    return (int)((((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56);
}

/* Free the memory of the node in the reference tree */
void FreeTree(ReferenceTreeNode** node)
{
    /* If any subtree belongs to this node, the memory for the subtree will be free recurrsively. */
    if ((*node)->children > 1)
    {
        int i = 0;
        for (; i < (*node)->children - 1; i++)
            FreeTree(&((*node)->childNode[i]));
        free((*node)->childNode);
        free((*node)->distance);
    }
    /* After the recursive free, the memory of this node is freed */
    free(*node);
}

int main(int argc, char** argv)
{
    int iterations = 5;
    double totalTime1 = 0.0;
    double totalTime2 = 0.0;
    double totalTime3 = 0.0;
    for (int kk = 0; kk < iterations; kk++) {
        UC** pattern = NULL;                 /* The array for recording patterns */
        int* lengthPattern = NULL;           /* The array for recording lengths of pattern */
        int numberPattern = 0;            /* The number of patterns */
        //FILE* pointerPatternFile = NULL;     /* The FILE pointer for opening the file of patterns */
        int lmin = 0;                        /* Recording the mininum length among all patterns */
        int parameterL = 0;                  /* Parameter l */
        int parameterK = 0;                  /* Parameter k */
        int i = 0;                           /* for loop index */
        clock_t preprocessingStart;   /* Time stamp for the start of preprocecssing */
        clock_t preprocessingEnd;     /* Time stamp for the end of preprocessing */
        clock_t searchingStart;       /* Time stamp for the start of searching */
        clock_t searchingEnd;         /* Time stamp for the end of searching */
        int abb, acc;
        /* Checking the number of input arguments */
        /* When the input number is incorrect, the help message for using this program is displayed. */
        if (argc != 5)
        {
            printf("Usage: %s [text file] [pattern file] [parameter l] [parameter k]\n", argv[0]);
            printf("       [text file]: the file name of the input text string\n");
            printf("       [pattern file]: the file name of the input pattern strings\n");
            printf("       [parameter l]: the user's assigned length of prefixes\n");
            printf("       [parameter k]: the user's assigned size for determining whether a node is a leaf in reference tree\n\n");
            return 0;
        }

        /* Checking whether the opening file for patterns is successfully */
        FILE* pointerPatternFile = NULL;

        // 正確使用 fopen_s
        errno_t err1 = fopen_s(&pointerPatternFile, argv[2], "rb");
        if (err1 != 0 || pointerPatternFile == NULL) {
            /* 如果無法開啟檔案，停止程式 */
            printf("ERROR: Cannot read the pattern file (\"%s\").\n", argv[2]);
            return 0;
        }



        /* Note that the format of patterns is described as follows:
            * The first line is the total number of patterns.
            * Patterns are recorded in the rest of lines.
            * Each line (from 2nd to the last lines) contains two elements.
            * The length is recorded in the first element, and its corresponding pattern is in the second element.
            * Two elements are separated by a comma.
            * */
            /* Read the number of pattern */
        fscanf_s(pointerPatternFile, "%d\n", &numberPattern);
        /* Request memory for the first dimension of pattern array */
        pattern = (UC**)malloc(sizeof(UC*) * numberPattern); // 顯式轉型為 UC**
        if (!pattern) {
            printf("Memory allocation failed for pattern\n");
            return -1;
        }
        /* Request memory for the lengths of patterns */
        lengthPattern = (int*)malloc(sizeof(int) * numberPattern); // 顯式轉型為 int*
        if (!lengthPattern) {
            printf("Memory allocation failed for lengthPattern\n");
            return -1;
        }
        /* Read each line to obtain the length and the pattern by for loop */
        for (i = 0; i < numberPattern; i++)
        {
            /* Obtain the length of the pattern first, and then request the memory of this pattern array (the second dimension) */
            fscanf_s(pointerPatternFile, "%d,", &lengthPattern[i]);
            /* Record the minimum length among all patterns */
            if (i == 0)
                lmin = lengthPattern[i];
            else if (lmin > lengthPattern[i])
                lmin = lengthPattern[i];
            pattern[i] = (UC*)malloc(lengthPattern[i] * sizeof(UC)); // 根據 lengthPattern[i] 的值來分配內存
            if (!pattern[i]) {
                printf("Memory allocation failed for pattern[i]\n");
                return -1;
            }
            /* Read the pattern and store into pattern array */
            fread(pattern[i], 1, lengthPattern[i], pointerPatternFile);
            /* Prepare for next line in the file */
            fscanf_s(pointerPatternFile, "\n");
        }
        /* close the file of patterns */
        fclose(pointerPatternFile);

        /* obtain the parameters l and k */
        parameterL = atoi(argv[3]);
        parameterK = atoi(argv[4]);

        /* set the time stamp of the start for preprocessing */
        preprocessingStart = clock();

        //FILE* pointerTextFile = NULL;          /* The FILE pointer for opening the file of text */
        int lengthText = 0;                    /* The length of text */
        UC alphabet[MAX_ALPHABET_SIZE];        /* The alphabet of strings */
        UC ENV = sizeof(ULL) * 8;              /* The size of a computer word */
        UC sizeAlphabet = 0;                   /* The size of alphabet */
        UC inputChar = 0;                      /* A temporary character for reading text string */
        UC numberBitForSymbol = 1;             /* A Symbol need how many bits*/
        UC numberSymbolInUll = 1;              /* maximum number of characters in an ULL 64/3=21 */
        ULL filterUselessBit = 0x0ULL;         /* The mask for filtering unused bits in a ULL */
        ULL clearWitnessBit = 0x0ULL;          /* The mask for filtering the witness bits */
        ULL keepWitnessBit = 0x0ULL;           /* The mask for filtering the bits whic are not witness bits */
        int totalNumberUllForText = 1;         /* The number of ULLs that text string needs */
        int numberInputString = 0;             /* The number of input strings */
        int aux1 = 0;                          /* The temporary variable for obtaining the bit string */
        int aux2 = 0;
        ULL bitReference = 0x0ULL;             /* The bit string with respect to reference string */
        ULL bitString = 0x0ULL;                /* The bit string with respect to input string */
        ULL* bitText = NULL;                   /* The bit string with respect to text string */
        int* posTemp[2];                       /* The memory for recording whole input strings (use double space) */
        UC* distance = NULL;                   /* The Hamming distances of child nodes */
        int* counting = NULL;                  /* The counting table for counting sort */
        int sizeofCounting = 0;                /* The size of counting table */
        ReferenceTreeNode* root = NULL;        /* Root node pointer */
        ReferenceTreeNode* currentNode = NULL; /* The node pointer which is ready to be expanded */
        Stack* head = NULL;                    /* The head pointer of the stack */
        Stack* currentJob = NULL;              /* The current pointer which is ready to be processing */
        int start = 0;                         /* The start position in memory for input string */
        int end = 0;                           /* The end position in memory for input string */
        int reference = 0;                     /* The position for reference string */
        UC children = 0;                       /* The number of child nodes */
        bool order = true;                     /* The flag for accessing memory */
        bool newOrder = false;                 /* The new flag for accessing memory */
        bool flag = true;
        UC HD = 0;
        int j = 0;
        int k = 0;

        /* This program is only run in 64-bit operating system. If it is not 64-bit OS, stop this program. */
        if (ENV < 64)
        {
            printf("ERROR: Please run this program in 64-bit environment.\n");
            return 0;
        }

        /* Check whether the opening file for text is successful */
        FILE* pointerTextFile = NULL;
        errno_t err2;

        if ((err2 = fopen_s(&pointerTextFile, argv[1], "rb")) != 0)
        {
            /* If fail to open the file, free the used memory and stop. */
            printf("ERROR: Cannot read the text file (\"%s\"). Error code: %d\n", argv[1], err2);
            for (i = 0; i < numberPattern; i++)
                free(pattern[i]);
            free(pattern);
            free(lengthPattern);
            return 0;
        }

        /* Read the text string and record the alphabet of text */
        memset(alphabet, 0xff, MAX_ALPHABET_SIZE);
        while ((char)(inputChar = fgetc(pointerTextFile)) != EOF)
            alphabet[inputChar] = 0;
        lengthText = ftell(pointerTextFile);  /* Obtain the length of text string */

        for (i = 0; i != MAX_ALPHABET_SIZE; i++)
            if (!alphabet[i])
                alphabet[i] = sizeAlphabet++; /* Assign an unique number for this symbol */

        for (i = 2; i < sizeAlphabet; i *= 2)
            numberBitForSymbol++;             /* Calculate the number of bits used to record a symbol */
        numberBitForSymbol++;                 /* An extra leading witness bit is needed */

        /* Note:
            * This program is only used one computer word to implement reference tree approach.
            * If the parameter l is too large that one computer word cannot deal with it, stop.
            * */
        if (parameterL * numberBitForSymbol > ENV)
        {
            printf("ERROR: The parameter l is too large! (A unit (unsigned long long) cannot handle it.)\n");
            return 0;
        }

        /* Compute the maximum number of characters in an ULL */
        numberSymbolInUll = ENV / numberBitForSymbol;
        /* Compute the number of total input strings */
        numberInputString = lengthText - (lmin > parameterL ? lmin : parameterL) + 1;
        /* Compute the number of ULLs that text string needs */
        if (lengthText % numberSymbolInUll)
            totalNumberUllForText = lengthText / numberSymbolInUll;
        else
            totalNumberUllForText = lengthText / numberSymbolInUll + 1;

        /* If characters can exactly fill up in an ULL, no bit will be filted;
            * Otherwise, compute the mask for filtering the unused bits in the high positions.
            * */
        if (parameterL * numberBitForSymbol == ENV)
            filterUselessBit = ~0x0ULL;
        else
            filterUselessBit = (0x1ULL << (parameterL * numberBitForSymbol)) - 1;

        /* Compute the masks for filtering witness bits and keeping witness bits */
        keepWitnessBit = clearWitnessBit = 0x1ULL << (numberBitForSymbol - 1);
        for (i = 1; i != parameterL; i++)
            keepWitnessBit = (keepWitnessBit << numberBitForSymbol) | clearWitnessBit;
        clearWitnessBit = (~keepWitnessBit) & filterUselessBit;

        /* Request memory for bit string with respect to text string */
        bitText = (ULL*)malloc(sizeof(ULL) * totalNumberUllForText); // 顯式轉型為 ULL*
        if (!bitText) {
            printf("Memory allocation failed for bitText\n");
            return -1;
        }
        /* Request memory for whole input strings (double space for swaping) */
        posTemp[0] = (int*)malloc(sizeof(int) * lengthText);
        posTemp[1] = (int*)malloc(sizeof(int) * lengthText);
        if (!posTemp[0]) {
            printf("Memory allocation failed for posTemp[0]\n");
            return -1;
        }
        if (!posTemp[1]) {
            printf("Memory allocation failed for posTemp[1]\n");
            return -1;
        }
        /* Request memory for recording the distances between input string and reference string */
        distance = (UC*)malloc(numberInputString);
        if (!distance) {
            printf("Memory allocation failed for distance \n");
            return -1;
        }
        /* Request memory for the table of counting sort */
        sizeofCounting = sizeof(int) * (parameterL + 1);
        counting = (int*)malloc(sizeofCounting); // 分配內存
        if (!distance) {
            printf("Memory allocation failed for distance \n");
            return -1;
        }
        printf("\n");
        //上面那行不知道是怎樣刪掉就壞了

        /* Read text string again and transfer each character into its corresponding number */
        fseek(pointerTextFile, 0, SEEK_SET);
        for (i = 0; i != lengthText; i++)
        {
            posTemp[0][i] = i;
            bitText[i / numberSymbolInUll] = (bitText[i / numberSymbolInUll] << numberBitForSymbol) | alphabet[fgetc(pointerTextFile)];
        }

        /* left shift the valid bits to the corresponding positions */
        bitText[(lengthText - 1) / numberSymbolInUll] <<= ((numberSymbolInUll - ((lengthText - 1) % numberSymbolInUll) - 1) * numberBitForSymbol);
        /* Close the file of text */
        fclose(pointerTextFile);

        /* Request memory for root node */
        root = (ReferenceTreeNode*)malloc(sizeof(ReferenceTreeNode));
        /* Request memory for the first element of stack and then initialize it */
        head = (Stack*)malloc(sizeof(Stack));
        head->start = 0;
        head->numberInputString = numberInputString;
        head->order = false;
        head->pointerNode = (void*)root;
        head->previous = NULL;
        /* Point the first element of stack to be the element ready to be processing */
        currentJob = head;

        /* Tackle the stack until the stack is empty */
        while (currentJob != NULL)
        {
            /* Dealing with the current element */
            start = currentJob->start;
            numberInputString = currentJob->numberInputString;
            order = currentJob->order;
            currentNode = (ReferenceTreeNode*)(currentJob->pointerNode);

            /* Point to next element */
            if (currentJob->previous == NULL)
            {
                free(currentJob);
                currentJob = NULL;
            }
            else
            {
                currentJob = currentJob->previous;
                free(currentJob->next);
            }

            /* Consider the number of input strings for the current node.
                * If the number is greater than parameter k, this node is an internal node needed to be expanded.
                * Otherwise, this node is a leaf node.
                * */
            if (numberInputString > parameterK) /* Internal node */
            {
                /* Calculate the ending position in the memory */
                end = start + numberInputString;
                /* Pick the first element to be reference string */
                reference = start;

                /* Initialize the table for counting sort */
                memset(counting, 0, sizeofCounting);

                /* Compute the bit string with respect the reference string */
                aux1 = numberSymbolInUll - posTemp[order][reference] % numberSymbolInUll;
                aux2 = posTemp[order][reference] / numberSymbolInUll;
                /* If all the bits of the reference string are in an ULL of the bit string with respect to text, bit string can be obtained directly.
                    * Otherwise, these bits will be extracted from two ULL of the bit string with respect to text.
                    * */
                if (aux1 >= parameterL)
                    bitReference = (bitText[aux2] >> ((aux1 - parameterL) * numberBitForSymbol)) & filterUselessBit;
                else
                    bitReference = ((bitText[aux2] << ((parameterL - aux1) * numberBitForSymbol)) | (bitText[aux2 + 1] >> ((numberSymbolInUll - parameterL + aux1) * numberBitForSymbol))) & filterUselessBit;

                /* Compute the Hamming distances between each input string and the reference string */
                for (i = start; i != end; i++)
                {
                    /* Obtain the bit string with respect to the input string
                        * This process is similar to that of reference string as shown above */
                    aux1 = numberSymbolInUll - posTemp[order][i] % numberSymbolInUll;
                    aux2 = posTemp[order][i] / numberSymbolInUll;
                    if (aux1 >= parameterL)
                        bitString = (bitText[aux2] >> ((aux1 - parameterL) * numberBitForSymbol)) & filterUselessBit;
                    else
                        bitString = ((bitText[aux2] << ((parameterL - aux1) * numberBitForSymbol)) | (bitText[aux2 + 1] >> ((numberSymbolInUll - parameterL + aux1) * numberBitForSymbol))) & filterUselessBit;
                    /* popcount method for computing Hamming distance */
                    parallelGroupStrings(bitText,  // 位元串陣列
                        lengthText,               // 文字總長度
                        parameterL,               // 參數 L
                        numberInputString        // 總字串數量
                    );
                    //HD = ComputeHD_64(((bitString ^ bitReference) + clearWitnessBit) & keepWitnessBit);
                    distance[i] = HD;
                    counting[HD]++;
                }

                /* Comuting the number of child nodes and update the table of counting sort */
                children = 0;
                for (i = 1; i <= parameterL; i++)
                {
                    if (counting[i]) children++;
                    counting[i] += counting[i - 1];
                }

                currentNode->children = children + 1;
                /* If all input string are identical, this node becomes leaf node */
                if (!children)
                {
                    currentNode->numberReference = numberInputString;
                    currentNode->reference = start;
                    currentNode->order = order;

                    continue;
                }

                /* The flag for accessingg memory of whole input strings.
                    * If the flag is 0, the first piece memory will be used.
                    * If the flag is 1, the second piece memory will be used.
                    * This approach will only copy the memory used in the further expanding easily.
                    * */
                newOrder = (order + 1) % 2;
                for (i = end - 1; i >= start; i--)
                    posTemp[newOrder][start + (--counting[distance[i]])] = posTemp[order][i];

                /* Update this internal node.
                    * 1. Record the reference string
                    * 2. Record the start position of input string in memory
                    * 3. Record the new flag of accessing memory
                    * 4. Request memory for child nodes (include their corresponding distance)
                    * */
                currentNode->numberReference = counting[1];
                currentNode->reference = start;
                currentNode->order = newOrder;

                currentNode->distance = (UC*)malloc(children);
                currentNode->childNode = (ReferenceTreeNode**)malloc(sizeof(ReferenceTreeNode*) * children);

                /* Store the processing of expanding child nodes into stack */
                for (i = 1, j = 0; i <= parameterL; i++)
                {
                    k = (i == parameterL) ? numberInputString - counting[i] : counting[i + 1] - counting[i];
                    if (!k) continue;
                    /* Request memory for the pointer of child nodes and the corrersponding distances */
                    currentNode->distance[j] = i;
                    currentNode->childNode[j] = (ReferenceTreeNode*)malloc(sizeof(ReferenceTreeNode));

                    /* Request memory for this element of stack */
                    if (currentJob == NULL)
                    {
                        currentJob = (Stack*)malloc(sizeof(Stack));
                        currentJob->previous = NULL;
                    }
                    else
                    {
                        currentJob->next = (Stack*)malloc(sizeof(Stack));
                        currentJob->next->previous = currentJob;
                        currentJob = currentJob->next;
                    }
                    /* Sotre the relative data of this element into stack */
                    currentJob->start = start + counting[i];
                    currentJob->numberInputString = k;
                    currentJob->order = newOrder;
                    currentJob->pointerNode = (void*)(currentNode->childNode[j++]);
                }
            }
            else /* Leaf node */
            {
                /* Store the input strings, the start position in the memory, and the flag of accessing memory */
                currentNode->numberReference = numberInputString;
                currentNode->reference = start;
                currentNode->order = order;
                currentNode->children = 0;
            }
        }

        /* Prepare the memory for pattern */
        ULL filterBitForSingleSymbol = (0x1ULL << numberBitForSymbol) - 0x1ULL;
        ULL bitPattern = 0x0ULL;

        /* Set the time stamp of the end for preprocessing */
        preprocessingEnd = clock();
        /* Set the time stamp of the start for searching */
        searchingStart = clock();

        /* Consider the patterns one by one */
        for (i = 0; i != numberPattern; i++)
        {
            /* Initialize the bit string for pattern */
            bitPattern = 0x0ULL;
            /* Compute the bit string with respect to this pattern */
            for (j = 0; j != parameterL; j++)
                bitPattern = (bitPattern << numberBitForSymbol) | alphabet[pattern[i][j]];

            /* Search the reference tree started from root node */
            currentNode = root;
            while (1)
            {   /* 1. 開啟檔案
                FILE* file = fopen("output.txt", "w");
                if (file == NULL) {
                    perror("無法開啟檔案");
                    return 1;
                }*/
                /* If this is an internal node, compute the Hamming distance between l-prefix of this pattern and the reference string */
                if (currentNode->children != 0)
                {
                    /* Obtain the bit string with respect to the reference string */
                    aux1 = numberSymbolInUll - posTemp[currentNode->order][currentNode->reference] % numberSymbolInUll;
                    aux2 = posTemp[currentNode->order][currentNode->reference] / numberSymbolInUll;
                    if (aux1 >= parameterL)
                        bitReference = (bitText[aux2] >> ((aux1 - parameterL) * numberBitForSymbol)) & filterUselessBit;
                    else
                    {
                        //                    if (aux2 + 2 == totalNumberUllForText)
                        //                        bitReference = ((bitText[aux2] << ((parameterL - aux1) * numberBitForSymbol)) | (bitText[aux2 + 1] >> ((numberSymbolInLastUllForText - parameterL + aux1) * numberBitForSymbol))) & filterUselessBit;
                        //                    else
                        bitReference = ((bitText[aux2] << ((parameterL - aux1) * numberBitForSymbol)) | (bitText[aux2 + 1] >> ((numberSymbolInUll - parameterL + aux1) * numberBitForSymbol))) & filterUselessBit;
                    }
                    

                    /* Compute the Hamming distance by popcount method */
                    HD = ComputeHD_64(((bitPattern ^ bitReference) + clearWitnessBit) & keepWitnessBit);
                    if (HD) /* If the distance is not zero, the corresponding subtree will be searched */
                    {
                        start = currentNode->children - 1;
                        /* If the child node whose distance is equal to the current Hamming distance exists, continue the search.
                            * Otherwise (such child node does not exist), the search for this pattern stops.
                            * */
                        flag = false;
                        for (j = 0; j != start; j++)
                            if (currentNode->distance[j] == HD)
                            {
                                flag = true;
                                break;
                            }
                        if (flag == true)
                        {
                            currentNode = currentNode->childNode[j];
                            continue;
                        }
                        else
                            break;
                    }
                    else /* If the distance is zero, search go to the special leaf node whose strings all are identical to the l-prefix of this pattern */
                    {
                        /* For each string in this node, linear compare with the remaining suffixes of patterns one by one */
                        end = currentNode->reference + currentNode->numberReference;
                        order = currentNode->order;
                        /* If the length of pattern is equal to parameter l, exact matches are found */
                        if (parameterL == lengthPattern[i])
                        {
                            for (j = currentNode->reference; j != end; j++)
                                //fprintf(file, "%d(%d),", posTemp[order][j] + parameterL, i + 1);
                                printf("%d(%d),", posTemp[order][j] + parameterL, i + 1);
                                abb = 0;
                        }
                        else
                        {
                            for (start = currentNode->reference; start != end; start++)
                            {
                                flag = true;
                                for (j = parameterL, k = posTemp[order][start] + parameterL; j < lengthPattern[i]; j++, k++)
                                    if (alphabet[pattern[i][j]] != ((bitText[k / numberSymbolInUll] >> ((numberSymbolInUll - k % numberSymbolInUll - 1) * numberBitForSymbol)) & filterBitForSingleSymbol))
                                    {
                                        flag = false;
                                        break;
                                    }
                                if (flag == true)
                                {
                                    if (k <= lengthText)
                                        //fprintf(file, "%d(%d),", k, i + 1);
                                        printf("%d(%d),", k, i + 1);
                                        acc = 0;
                                }
                            }
                        }

                        break;
                    }
                }
                /* If this node is leaf node, compare the string in this node with pattern one by one */
                else
                {
                    end = currentNode->reference + currentNode->numberReference;
                    order = currentNode->order;
                    for (start = currentNode->reference; start != end; start++)
                    {
                        flag = true;
                        for (j = 0, k = posTemp[order][start]; j < lengthPattern[i]; j++, k++)
                            if (alphabet[pattern[i][j]] != ((bitText[k / numberSymbolInUll] >> ((numberSymbolInUll - k % numberSymbolInUll - 1) * numberBitForSymbol)) & filterBitForSingleSymbol))
                            {
                                flag = false;
                                break;
                            }
                        if (flag == true)
                        {
                            if (k <= lengthText)
                                //fprintf(file, "%d(%d),", k, i + 1);
                                printf("%d(%d),", k, i + 1);
                                acc = 0;
                        }
                    }

                    break;
                }
            }
        }

        /* 預處理(建樹)、搜尋、全部時間*/
        searchingEnd = clock();
        double preprocessing_clock = (static_cast<double>(preprocessingEnd - preprocessingStart) / CLOCKS_PER_SEC) * 1000.0;
        double searching_clock = (static_cast<double>(searchingEnd - searchingStart) / CLOCKS_PER_SEC) * 1000.0;
        double all_clock = (static_cast<double>(searchingEnd - preprocessingStart) / CLOCKS_PER_SEC) * 1000.0;

        totalTime1 += preprocessing_clock;
        totalTime2 += searching_clock;
        totalTime3 += all_clock;

        printf("%.2f, %.2f, %.2f\n", preprocessing_clock, searching_clock, all_clock);
        /* Free the memories*/
        for (i = 0; i < numberPattern; i++)
            free(pattern[i]);
        free(pattern);
        free(lengthPattern);
        free(bitText);
        free(posTemp[0]);
        free(posTemp[1]);
        free(distance);
        free(counting);
        FreeTree(&root);

    }
    double averageTime1 = totalTime1 / iterations;
    double averageTime2 = totalTime2 / iterations;
    double averageTime3 = totalTime3 / iterations;
    printf("average:\n");
    printf("%f, %f ,%f\n", averageTime1, averageTime2, averageTime3);
    /* Return normal ending to system */
    return 0;
}
