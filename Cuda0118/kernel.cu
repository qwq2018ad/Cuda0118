#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <iostream>
#include <cstring>
#include <cmath>

#define MAX_SYMBOLS 64
#define filterUselessBit 0xFFFFFFFFFFFFFFFFULL

struct ReferenceTreeNode {
    int numberReference;             // 字串數量
    int reference;                   // 參考字串位置
    int order;                       // 當前的存取順序
    int children;                    // 子節點數量
    int* distance;                   // 各子群的距離
    ReferenceTreeNode** childNode;   // 子節點指針陣列
};

__global__ void computeHammingAndGroup(
    const unsigned long long* bitText,
    const int* posTemp,
    int numStrings,
    int numSymbolsInULL,
    int parameterL,
    unsigned long long bitReference,
    int* distances,
    int* counting
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numStrings) return;

    int aux1 = numSymbolsInULL - posTemp[i] % numSymbolsInULL;
    int aux2 = posTemp[i] / numSymbolsInULL;

    unsigned long long bitString;
    if (aux1 >= parameterL) {
        bitString = (bitText[aux2] >> ((aux1 - parameterL) * MAX_SYMBOLS)) & filterUselessBit;
    }
    else {
        bitString = ((bitText[aux2] << ((parameterL - aux1) * MAX_SYMBOLS)) |
            (bitText[aux2 + 1] >> ((numSymbolsInULL - parameterL + aux1) * MAX_SYMBOLS))) & filterUselessBit;
    }

    unsigned long long xorResult = bitString ^ bitReference;
    distances[i] = __popcll(xorResult);

    atomicAdd(&counting[distances[i]], 1);
}

__global__ void groupAndBuildTree(
    const int* distances,
    const int* counting,
    int* newPosTemp,
    int* prefixSum,
    int numStrings,
    int numGroups,
    ReferenceTreeNode* currentNode,
    ReferenceTreeNode** childNodes   // childNodes 改為 ReferenceTreeNode**
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numStrings) return;

    int groupIndex = distances[i];
    int pos = atomicAdd(&prefixSum[groupIndex], 1);
    newPosTemp[pos] = i;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int totalChildren = 0;
        for (int j = 1; j <= numGroups; j++) {
            if (counting[j] > 0) {
                currentNode->distance[totalChildren] = j;
                currentNode->childNode[totalChildren] = childNodes[totalChildren]; // 這裡的修改
                totalChildren++;
            }
        }
        currentNode->children = totalChildren;
    }
}


void buildReferenceTreeCUDA(
    unsigned long long* bitText,
    int* posTemp[2],
    int numStrings,
    int numSymbolsInULL,
    int parameterL,
    int parameterK,
    ReferenceTreeNode* root
) {
    int* distances;
    int* counting;
    int* prefixSum;
    cudaMalloc(&distances, sizeof(int) * numStrings);
    cudaMalloc(&counting, sizeof(int) * (parameterL + 1));
    cudaMalloc(&prefixSum, sizeof(int) * (parameterL + 1));

    root->numberReference = numStrings;
    root->reference = 0;
    root->order = 0;

    ReferenceTreeNode* currentNode = root;
    int currentOrder = 0;

    while (currentNode != NULL) {
        cudaMemset(counting, 0, sizeof(int) * (parameterL + 1));

        computeHammingAndGroup << <(numStrings + 255) / 256, 256 >> > (
            bitText,
            posTemp[currentOrder],
            numStrings,
            numSymbolsInULL,
            parameterL,
            bitText[currentNode->reference],
            distances,
            counting
            );

        groupAndBuildTree << <(numStrings + 255) / 256, 256 >> > (
            distances,
            counting,
            posTemp[1 - currentOrder],
            prefixSum,
            numStrings,
            parameterL,
            currentNode,
            currentNode->childNode  // 傳遞 childNode 作為 ReferenceTreeNode** 類型
            );

        currentOrder = 1 - currentOrder;
        currentNode = NULL;  // 需要加入邏輯取得下一個節點
    }

    cudaFree(distances);
    cudaFree(counting);
    cudaFree(prefixSum);
}

void printTree(ReferenceTreeNode* node, int depth = 0) {
    if (node == nullptr) return;

    // 列印當前節點的詳細資料
    std::cout << std::string(depth * 2, ' ') << "Node at depth " << depth << std::endl;
    std::cout << std::string(depth * 2, ' ') << "Reference: " << node->reference << std::endl;
    std::cout << std::string(depth * 2, ' ') << "Number of References: " << node->numberReference << std::endl;
    std::cout << std::string(depth * 2, ' ') << "Children: " << node->children << std::endl;

    // 如果有子節點，遞迴列印
    if (node->children > 0) {
        for (int i = 0; i < node->children; ++i) {
            printTree(node->childNode[i], depth + 1);
        }
    }
}


ReferenceTreeNode* findNodeByReference(ReferenceTreeNode* node, int reference) {
    if (node == nullptr) return nullptr;

    if (node->reference == reference) {
        return node;
    }

    // 遞迴搜尋子節點
    if (node->children > 0) {
        for (int i = 0; i < node->children; ++i) {
            ReferenceTreeNode* result = findNodeByReference(node->childNode[i], reference);
            if (result != nullptr) return result;
        }
    }

    return nullptr;  // 如果沒有找到對應的節點
}

int getTotalChildren(ReferenceTreeNode* node) {
    if (node == nullptr) return 0;

    int totalChildren = 0;
    if (node->children > 0) {
        for (int i = 0; i < node->children; ++i) {
            totalChildren += getTotalChildren(node->childNode[i]);  // 遞迴統計
        }
    }
    else {
        totalChildren = 1;  // 如果是葉節點，則計為1
    }

    return totalChildren;
}


int main() {
    const int numStrings = 6;
    const int parameterL = 3;
    const int parameterK = 2;

    unsigned long long bitText[] = {
        // 6 個測試字串，這些字串可以隨意調整長度和內容
        0b1010101010101010,        // 字串 1
        0b1110101010101110,        // 字串 2
        0b1011101010111010,        // 字串 3
        0b1111111111111111,        // 字串 4
        0b0000000000000000,        // 字串 5
        0b1110001110001110,        // 字串 6

        // 更多字串
        0b0101010101010101,        // 字串 7
        0b1101101101101101,        // 字串 8
        0b0001110001110001,        // 字串 9
        0b1111110000001111,        // 字串 10
        0b1010101010101011,        // 字串 11
        0b1100110011001100,        // 字串 12

        // 更多的隨機字串，適合用於測試
        0b1001001001001001,        // 字串 13
        0b1100001110000111,        // 字串 14
        0b0110101010101010,        // 字串 15
        0b1111001100110011,        // 字串 16
        0b0000000000000000,        // 字串 17
        0b0000000000000011         // 字串 18
    };


    int posTemp1[] = { 0, 1, 2, 3, 4, 5 };
    int posTemp2[numStrings];
    int* posTemp[2] = { posTemp1, posTemp2 };

    ReferenceTreeNode root;
    buildReferenceTreeCUDA(bitText, posTemp, numStrings, MAX_SYMBOLS, parameterL, parameterK, &root);

    // 列印樹的結構
    printTree(&root);

    // 查找指定 reference 的節點
    unsigned long long queryReference = 3;
    ReferenceTreeNode* foundNode = findNodeByReference(&root, queryReference);
    if (foundNode != nullptr) {
        std::cout << "Node found with reference " << queryReference << std::endl;
    }
    else {
        std::cout << "Node with reference " << queryReference << " not found." << std::endl;
    }

    // 計算總子節點數量
    int totalChildren = getTotalChildren(&root);
    std::cout << "Total children in the tree: " << totalChildren << std::endl;

    return 0;
}

