using LinearAlgebra
using DataFrames
using Random
using CSV

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# DESCRIPTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# 
# States: (n^2 + 6 values)
# Represented as (grid, 'lives' left, sunk ship (agent), sunk ships (opp), special shots (agent), special shots (opp)) where:
#   Grid:
#   n x n matrix of integers where each integer value represents:
#       -1 : Missed
#       0 : Unexplored
#       1 : Hit
#
#   'Lives' left:
#   Integer representing how many grid elements our opponent still has to hit to kill us
#   I.e. the number equalling:
#       # of beginning free ship slots - # of times opponent has hit us
#       17 - # of times opponent has hit us (assuming standard fleet; 2 + 3 + 3 + 4 + 5)
#
#   Sunk ships (agent):
#   Integer representing the total number of our remaining ships 
# 
#   Sunk ships (opp):
#   Integer representing the total number of our opponent's remaning ships
#
#   Special Shots (agent):
#   Tuple of integers representing how many uses of each special shot type we have left
#   (bomb shots left, line shots left) where each integer value is 'hopefully' within the interval [0, 2]
#   Rewards will be all negative for any interval violating this condition
#
#   Special Shots (opp): 
#   Same as above but for the opponent's special shots
#
# Actions: (3 values)
# Represented as tuple of integers:
#   (pos1, pos2, type) where:
#       pos1 is the row of the selected shot
#       pos2 is the column of the selected shot
#       type is the type of the shot; options include:
#           0: 'normal', -1: 'bomb', 1: 'line'
#
# Model: (highly variable number of values, depends on feature function)
# Represented as a tuple of tuples:
#   ((weight1, bias1, activation1), (weight2, bias2, activation2), ...) where:
#       weight1 is the weights for layer 1
#       bias1 is the bias for layer 1
#       activation1 is the activation immediately proceeding layer 1
#           (NOTE: activation_N is ignored)
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# DESCRIPTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# HYPERPARAMETERS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Size of the battleship board (n)
# Classic battleship has 10 x 10 board size
board_size = 10

# Fleet of ships
# Assumed standard fleet (1x2, 1x3, 1x3, 1x4, 1x5)
fleet = [2, 3, 3, 4, 5]

# Discount Factor
# Since we can never return to a state, we'll never have loops 
# Thus, discount factor of 1 is fine
gamma = 1

# Which feature vector to use
# Possibilities include 'concatenation', 'fourier'
feature = "concatenation"

# Only if using feature 'fourier'
# Defines the max number of fourier permutations to consider
maxFourier = 20 

# Initial step size
initial_step_size = 0.005

# Toggle initial step size toggles off calculate_step_size
# True keeps step size constant at initial_step_size
toggle_initial_step_size = true

# The number of games we consider during training
num_games_to_try = 1#1000

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# HYPERPARAMETERS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CONSTANTS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Directions
left_direction = 1
right_direction = 2
up_direction = 3
down_direction = 4

# Shot Types
bomb_shot = -1
normal_shot = 0
line_shot = 1

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CONSTANTS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# GAME DYNAMICS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Functionality:
#   Prints the board to the terminal
#  
# Inputs:
#   Board (n x n matrix) representing the current state of the game board
#
# Outputs: 
#   Prints the board to the terminal
#   Outputs nothing
function print_board(board)
    for row in 1:board_size
        for column in 1:board_size
            print(board[row, column])
        end
        println(' ')
    end
    println(' ')
end

# Functionality:
#   Helper function for initialize_board
#   Tests whether a ship can be placed in a specified direction from a starting position
#   If update_board is true, it returns the updated board; otherwise, it returns a boolean indicating if the direction is valid
#
# Inputs:
#   Board (n x n matrix) representing the current state of the game board
#   Row (integer) representing the starting row for placing the ship
#   Column (integer) representing the starting column for placing the ship
#   Direction (integer)  representing the direction to place the ship (1: left, 2: right, 3: up, 4: down)
#   Ship (integer)  representing the length of the ship to be placed
#   Ship Index (integer) representing a particular ship's index
#   Update Board (boolean) that determines whether to update the board or just check if the direction is valid
#
# Outputs:
#   If update_board is true, returns a copy of the board with the ship placed in the specified direction
#   If update_board is false, returns a boolean indicating whether the ship can be placed in the specified direction
function direction_works(board, row, column, direction, ship, ship_index, update_board)
    # Setup
    slots_filled = ship - 1
    values_remaining = ship - 1
    current_row = row
    current_column = column

    board_copy = copy(board)
    board_copy[row, column] = (ship_index, 0)

    # Test the specified direction
    for i in 1:ship - 1
        if direction == left_direction
            current_column -= 1
        elseif direction == right_direction
            current_column += 1
        elseif direction == up_direction
            current_row -= 1
        else
            current_row += 1
        end 

        # Check that the new position is within the board
        if current_row < 1 || current_row > board_size || current_column < 1 || current_column > board_size 
            break
        end

        # Check that the position isn't already occupied by another ship
        if board_copy[current_row, current_column][1] != 0
            break
        end

        # Update the current position
        board_copy[current_row, current_column] = (ship_index, 0)
        values_remaining -= 1
        slots_filled -= 1
    end

    current_row = row
    current_column = column
    # Test the next direction
    for i in 1:values_remaining
        if direction == left_direction
            current_column += 1
        elseif direction == right_direction
            current_column -= 1
        elseif direction == up_direction
            current_row += 1
        else
            current_row -= 1
        end 

        # Check that the new position is within the board
        if current_row < 1 || current_row > board_size || current_column < 1 || current_column > board_size 
            break
        end

        # Check that the position isn't already occupied by another ship
        if board[current_row, current_column][1] != 0
            break
        end

        board_copy[current_row, current_column] = (ship_index, 0)
        slots_filled -= 1
    end

    if update_board
        return board_copy
    else
        return slots_filled == 0
    end
end

# Functionality:
#   Initializes an n x n Battleship board with random fleet assortment
#   Only used to provide proper game mechanics;
#       the model never sees the adversary's board
# 
# Inputs:
#   Board Size (integer) describing the size of the width of the board 
#
# Outputs:
#   Board Size x Board Size sized matrix with tupled entries:
#       (fleet_position, status) where:
#           Fleet positions are defined as integers with:
#               0 representing no ship
#               i representing ship where i is some unique integer for each ship
#           Status is whether the specified position has been shot at:
#               0 represents hasn't yet been shot
#               1 represents shot
function initialize_board(board_size)
    board = fill((0, 0), (board_size, board_size))
    
    shuffled_fleet = shuffle(fleet)

    # Iterate over all ships
    ship_index = 1
    for ship in shuffled_fleet
        while true
            random_row = rand(1:board_size)
            random_column = rand(1:board_size)

            # Check if the initial placement is on another ship
            if board[random_row, random_column][1] != 0
                continue
            end

            # Pick a random direction
            direction = rand(1:4)

            if !direction_works(board, random_row, random_column, direction, ship, ship_index, false)
                if direction == left_direction || direction == right_direction
                    direction = rand(up_direction:down_direction)
                else
                    direction = rand(left_direction:right_direction)
                end

                # If both directions don't work, then continue trying initial starting positions
                if !direction_works(board, random_row, random_column, direction, ship, ship_index, false)
                    continue
                end
            end

            # If we're here, then direction is a valid direction and we should populate the board with this ship
            board = direction_works(board, random_row, random_column, direction, ship, ship_index, true)
            ship_index += 1
            break
        end
    end

    return board
end

# Functionality:
#   Performs a bomb shot at a specific index
#       Bomb shots are 3x3 area shots
#   
# Inputs:
#   Board (n x n board) capturing the board state
#   Position 1 (integer) is the column of the bomb shot
#   Position 2 (integer) is the row of the bomb shot
# 
# Outputs:
#   Board (n x n board) representing the updated board with the bomb shot
function bomb_shot(board, pos1, pos2)
    # Define the range for the bomb shot
    row_range = max(1, pos2 - 1):min(board_size, pos2 + 1)
    col_range = max(1, pos1 - 1):min(board_size, pos1 + 1)

    # Apply the bomb shot to the 3x3 area
    for i in row_range
        for j in col_range
            board[i, j] = (board[i, j][1], 1)
        end
    end
    return board
end

# Functionality:
#   Performs a line shot at a specific index
#       Line shots explore an entire row or column
#   
# Inputs:
#   Board (n x n board) capturing the board state
#   Position 1 (integer) is the column of the line shot
#   Position 2 (integer) is the row of the line shot
#   Direction (integer) specifying which direction to take the line shot in
# 
# Outputs:
#   Board (n x n board) representing the updated board with the line shot
function line_shot(board, pos1, pos2, direction)
    # Horizontal line shot
    if direction == left_direction || direction == right_direction
        for j in 1:board_size
            board[pos2, j] = (board[pos2, j][1], 1)
        end
    # Vertical line shot
    elseif direction == up_direction || direction == down_direction
        for i in 1:board_size
            board[i, pos1] = (board[i, pos1][1], 1)
        end
    end
    return board
end

# Functionality:
#   Converts a game board to a state board
#   Should only be used on the opponent's board
#   
# Inputs:
#   Board (n x n board) capturing a player's board
#
# Outputs:
#   State board (n x n matrix) described above
function board_to_state_board(opponents_board)
    board = fill(0, (board_size, board_size))

    for i in 1:board_size
        for j in 1:board_size
            # If we've explored a grid element
            if opponents_board[i, j][2] == 1
                # If that grid element has a ship
                if opponents_board[i, j][1] != 0
                    # Mark a hit
                    board[i, j] = 1
                else
                    # Otherwise, mark a miss
                    board[i, j] = -1
                end
            end
        end
    end

    return board
end

# Functionality:
#   Returns the next state given the current state and chosen action
#   Assumes an opponent that follows a random policy
#       Will randomly assign a shot type at a random placement
#   Also returns the reward 
#
# Inputs:
#   State (tuple) described above
#   Action (tuple) described above
#   Agent's Board, an (n x n board) capturing the agent's board state
#   Opponent's Board, an (n x n board) capturing the opponent's board state
# 
# Outputs:
#   State (tuple) described above with updated values
#   New Agent's Board (n x n board) capturing the agent's board
#   New Opponent's Board (n x n board) capturing the Opponent's board
#   Reward for this particular action
function next_state(state, action, agents_board, opponents_board)
    action_row, action_col, action_type = action

    if action_type == bomb_shot
        opponents_board = bomb_shot(opponents_board, action_row, action_col)
    elseif action_type == line_shot
        opponents_board = line_shot(opponents_board, action_row, action_col)
    elseif action_type == normal_shot
        opponents_board[action_row, action_col] = (opponents_board[action_row, action_col][1], 1)
    end

    agents_board = random_opponent_action(agents_board)

    # Convert updated agents_board, opponents_board to state
    state = nothing

    # Calculate the reward
    reward = nothing

    return state, agents_board, opponents_board, reward
end

# Functionality:
#   Determines if the game is over based on the agent's and opponent's board
#   An ended game means one of the player has no remaining ships
#
# Inputs:
#   Agent's Board, an (n x n board) capturing the agent's board state
#   Opponent's Board, an (n x n board) capturing the opponent's board state
# 
# Outputs:
#   Boolean which is true if the game has ended
function is_game_ended(agents_board, opponents_board)
    agents_board_finished = true

    for cell in agents_board
        # If there's a ship that hasn't been shot at, this board still has 'lives'
        if cell[1] != 0 && cell[2] == 0
            agents_board_finished = false
        end
    end

    opponents_board_finished = true

    for cell in opponents_board
        if cell[1] != 0 && cell[2] == 0
            opponents_board_finished = false
        end
    end

    return agents_board_finished || opponents_board_finished
end

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# GAME DYNAMICS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# FEATURE VECTORS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Functionality:
#   Defines our feature given our agent's current state and chosen action
#   Uses Fourier transform (sum of cosines) as features to increase generality 
# 
# Inputs:
#   State (tuple) described above
#   Action (tuple) described above
#   maxFourier (integer) describing the maximum sum of terms used
#
# Outputs:
#   Cosine for each permuted value
#   EX: (for integer state action (s, a))
#       [1, cos(pi * (s)), cos(pi * (a)), cos(pi * (s + a)), cos(pi * (2s)), ...]
function feature_fourier_vector(state, action, maxFourier=maxFourier)

end

# Functionality:
#   Simple concatenation feature vector 
#
# Inputs:
#   State (tuple) described above
#   Action (tuple) described above
# 
# Outputs:
#   Concatenated features for all inputted values
function feature_concatenate_vector(state, action)

end

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# FEATURE VECTORS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CALCULATION FUNCTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Functionality:
#   Forward pass of predictor function (calculates Q)
#   Uses feature vector to make prediction 
#
# Inputs:
#   Model (tuple) described above
#   Feature () described above
#       Can be any of the options defined above
# 
# Outputs:
#   Predicted Q given a state, action and the current model's weights
function forward(model, feature)

end

# Functionality:
#   Backpropagation of our predictor function (updating weights)
#   Uses Q-learning update function with function approximation given by:
#       w <- w - eta * ( Q(s,a) - (r - gamma * V(s',a'))) * gradient where:
#           w : weights
#           eta : step size / learning rate
#               Given by calculate_step_size below
#           Q(s,a) : Q value for a specific state and action
#               Given by forward() defined above
#           r : reward from current state to future state 
#               Given by reward() defined below
#           gamma : discount factor
#           V(s',a') : V value (future value) for future state s' and action a'
#               Given by forward() defined above
#           gradient : Gradient/backpropogation through the model
#
# Inputs:
#   Model (tuple) described above
#   Feature () described above
#       Can be any of the options defined above
#   Move Index (integer) for use in calculating the step size
# 
# Outputs:
#   Predicted Q given a state, action and the current model's weights
function backprop(model, feature, move_index) 

end

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# REINFORCEMENT LEARNING FUNCTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Functionality:
#   Generates reward from a given state and action
#   Uses next_state to generate the future state given the action
#   Rewards for possible scenarios are:
#       Agent marked a hit on an opponent's ship: +2
#       Agent missed all opponent's ship: -0.5
#       Agent sinks opponent's ship: +3
#       Agent uses a special shot when it can't: -100000
#       Opponent marked a hit on an agent's ship: -1
#       Opponent missed all agent's ship: 0
#       Opponent sinks agent's ship: -2
#       (Note: these are stackable; if we hit an opponent's ship twice then +4)
#
# Inputs:
#   State (tuple) described above
#   Action (tuple) described above
# 
# Outputs:
#   Integer value representing reward of that particular action from the specified state
function reward(state, action)

end

# Functionality:
#   Reduces the step size depending on how far we are into the game to improve convergence
#   Uses initial_step_size
#   Toggleable using toggle_initial_step_size, which if is true we only return initial_step_size
#
# Inputs:
#   Move Index (integer) specifying how many moves we are in within the game 
# 
# Outputs:
#   Integer value representing the step size
function calculate_step_size(move_index)
    if toggle_initial_step_size
        return initial_step_size
    else
        return initial_step_size * (0.99) ^ (move_index / 20)
    end
end

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# REINFORCEMENT LEARNING FUNCTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# MAIN
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

function main()
    # Iterate over all games
    for _ in 1:num_games_to_try

        # Generate Boards
        agents_board = initialize_board(board_size)
        opponents_board = initialize_board(board_size)

        println("AGENT'S BOARD")
        print_board(agents_board)

        println("OPPONENT'S BOARD")
        print_board(opponents_board)

        println(is_game_ended(agents_board, opponents_board))

        #move_index = 1
        # Iterate until the game is over
        #while !is_game_ended(agents_board, opponents_board)

            # Generate Features
            #feature_vector = nothing
            #if feature == "concatenation"
                #feature_vector = 
            #elseif feature == "fourier"

            #move_index += 1
        #end
    end
end

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# MAIN
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CALL THE RL ALGORITHM 
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Usage: julia final.jl
main()