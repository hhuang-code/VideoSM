import pdb

"""
dynamic programming for 0-1 knapsack problem, based on https://www.geeksforgeeks.org/knapsack-problem/ 
Arg:
    W: maximum weight
    wt: weight (duration) list
    val: value (score) list
    n: length of wt (or val)
"""
def knapSack(W, wt, val, n):
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    sln = [[[] for x in range(W + 1)] for x in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                if val[i - 1] + K[i - 1][w - wt[i - 1]] > K[i - 1][w]:
                    K[i][w] = val[i - 1] + K[i - 1][w - wt[i - 1]]
                    sln[i][w] = list(sln[i - 1][w - wt[i - 1]])
                    if len(sln[i][w]) != 0:
                        sln[i][w].append(i - 1)
                    else:
                        sln[i][w] = [i - 1]
                else:
                    K[i][w] = K[i - 1][w]
                    sln[i][w] = list(sln[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
                sln[i][w] = list(sln[i - 1][w])

    return K[n][W], sln[n][W]

val = [4, 3, 6]
wt = [1, 3, 5]
W = 6

opt, sln = knapSack(W, wt, val, len(val))
print(opt)
print(sln)