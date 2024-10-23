import numpy as np

def argsort_iter(v):
    return sorted(range(len(v)), key=lambda i: v[i])

def augmenting_path(nc, cost, u, v, path, row4col, shortestPathCosts, i, SR, SC, remaining):
    minVal = 0

    num_remaining = nc
    for it in range(nc):
        remaining[it] = nc - it - 1

    SR.fill(False)
    SC.fill(False)
    shortestPathCosts.fill(np.inf)

    sink = -1
    while sink == -1:
        index = -1
        lowest = np.inf
        SR[i] = True

        for it in range(num_remaining):
            j = remaining[it]
            r = minVal + cost[i, j] - u[i] - v[j]
            if r < shortestPathCosts[j]:
                path[j] = i
                shortestPathCosts[j] = r

            if shortestPathCosts[j] < lowest or (shortestPathCosts[j] == lowest and row4col[j] == -1):
                lowest = shortestPathCosts[j]
                index = it

        minVal = lowest
        if minVal == np.inf:
            return -1, minVal

        j = remaining[index]
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        SC[j] = True
        remaining[index] = remaining[num_remaining - 1]
        num_remaining -= 1

    return sink, minVal

def solve(nr, nc, cost, maximize):
    if nr == 0 or nc == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    transpose = nc < nr

    if transpose:
        cost = cost.T
        nr, nc = cost.shape

    if maximize:
        cost = -cost

    if np.any(np.isnan(cost)) or np.any(cost == -np.inf):
        raise ValueError("Invalid cost matrix")

    u = np.zeros(nr)
    v = np.zeros(nc)
    shortestPathCosts = np.zeros(nc)
    path = np.full(nc, -1, dtype=int)
    col4row = np.full(nr, -1, dtype=int)
    row4col = np.full(nc, -1, dtype=int)
    SR = np.zeros(nr, dtype=bool)
    SC = np.zeros(nc, dtype=bool)
    remaining = np.zeros(nc, dtype=int)

    for curRow in range(nr):
        sink, minVal = augmenting_path(nc, cost, u, v, path, row4col, shortestPathCosts, curRow, SR, SC, remaining)
        if sink < 0:
            raise ValueError("Cost matrix is infeasible")

        u[curRow] += minVal
        for i in range(nr):
            if SR[i] and i != curRow:
                u[i] += minVal - shortestPathCosts[col4row[i]]

        for j in range(nc):
            if SC[j]:
                v[j] -= minVal - shortestPathCosts[j]

        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == curRow:
                break

    if transpose:
        return np.argsort(col4row), np.array(col4row)
    else:
        return np.arange(nr), col4row

def linear_sum_assignment(cost_matrix, maximize=False):
    nr, nc = cost_matrix.shape
    row_ind, col_ind = solve(nr, nc, cost_matrix, maximize)
    return row_ind, col_ind


def calculate_cost_matrix(teammate_positions, formation_positions):
    # Calculate the cost matrix (distances) between teammate positions and formation positions
    cost_matrix = np.zeros((len(teammate_positions), len(formation_positions)))
    
    for i, t_pos in enumerate(teammate_positions):
        for j, f_pos in enumerate(formation_positions):
            cost_matrix[i][j] = np.linalg.norm(np.array(t_pos) - np.array(f_pos))
    
    return cost_matrix

def role_assignment(teammate_positions, formation_positions):
    # Step 1: Calculate the cost matrix
    cost_matrix = calculate_cost_matrix(teammate_positions, formation_positions)
    
    # Step 2: Apply the Hungarian algorithm using scipy's linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the mapping from teammate (unum) to formation position
    assignment = {}
    for i in range(len(row_ind)):
        assignment[row_ind[i] + 1] = formation_positions[col_ind[i]]
    
    return assignment

def pass_reciever_selector(player_unum, teammate_positions, final_target):

    #Input : Locations of all teammates and a final target you wish the ball to finish at
    #Output : Target Location in 2d of the player who is recieveing the ball
    #-----------------------------------------------------------#

    # Example
    pass_reciever_unum = player_unum + 1                  #This starts indexing at 1, therefore player 1 wants to pass to player 2

    if pass_reciever_unum != 12:
        target = teammate_positions[pass_reciever_unum-1] #This is 0 indexed so we actually need to minus 1 
    else:
        target = final_target 

    return target

