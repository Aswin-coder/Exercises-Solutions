__kernel void logicalAND(__global const int* arrayA,
                             __global const int* arrayB,
                             __global int* result,
                             const unsigned int numRows,
                             const unsigned int numCols) 
    {
        int globalId = get_global_id(0);
        int row = globalId / numCols;
        int col = globalId % numCols;

        if (row < numRows && col < numCols) {
            result[row * numCols + col] = 0;
            if(arrayA[row * numCols + col]!=0)
            {
                if(arrayB[row * numCols + col]!=0)
                {
                    result[row * numCols + col]=1;
                }
            }
        }
    }