1) Creating the TicTacToe game :
   - writting the constructor
   - have a function "get_initial_state" to return the board with zeros
   - write a function "get_next_state" that take as arg state + action => next_state
   - function "valid_moves" that return the empty positions like [0,1,0,...,1]
   - function "check_win" that take state + action and return if the board is a winning one
   - function "get_value_and_terminated" : value = 1 if won else 0
   - function "get_opponent" that returns the opponent player ( here -player )
   - function "change_perspective" that takes as arg state + player
   - function "get_encoded_state" that takes as input a state of shape (3,3) and returns the encoded state of shape (3,3,3) // rgb to feed our NN

2) Creating the ResNet NN :
   -  start block
   -  backbone of ResBlock ( check the image )
   -  policyhead NN // valuehead NN

3) Create the Node class ( for MCTS ) :
   - constructor ( game, args, state, parent=None, action_taken=None, prior=0, visit_count=0 )
   - function "is_fully_expanded" : expandind the node is adding his futur children to the three we are building
   - function "select" that selects the best node using its ucb score
   - function "get_ucb" that returns the ucb score of a child. we play a 0sum game, so the parents wants to minimize the Q value of their children.
   - function "expand" that takes a policy as argument ( policy correspond to a probability distribution PI )
   - function "backpropagate"

4) Create a MCTS class:
   - constructor : game, args , model
   - function "search" that takes a state as a root and then applies the mcts algo using a model (NN)

5) AlphaZero class:
   - Constructor : game, optimizer, args, model
   - function "self_play" that takes into arg a state. The algo always plays as the player 1 and it's the state that changes to - state each step.
     it returns the historic of the play.
   - function "train" that takes a historic.
   - function "learn"
