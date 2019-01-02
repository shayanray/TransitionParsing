# Task

Transition Parsing with Neural Networks: implement the Neural Network based Transition
Parsing system described in [Chen and Manning, 2014]. Implement the training and test functionality for the transition classifier. Generate the features given the (sentence, tree) pairs and write the loss function to use for training the network. 

## Config and Results

Best Configuration:
• Activation function: Cube non-linearity in forwardPass(…) method

• Number of hidden layers: 2

• Std_dev = 0.1

• Config.py

1. UNKNOWN = "UNK"

2. ROOT = "ROOT"

3. NULL = "NULL"

4. NONEXIST = -1

5. max_iter = 5001

6. batch_size = 10000

7. hidden_size = 200 # using same size across multiple layers as applicable

8. embedding_size = 50

9. learning_rate = 0.1 

10. display_step = 100

11. validation_step = 200

12. n_Tokens = 48

13. lam = 1e-8

Best Configuration Results num_hidden = 2, max_iter=5001
Average loss at step 5000 >>  0.179273202

## Implementation


a sentence(sent) = (w1 w2 w3 w4 ... wn)
Init Config : S = [ROOT], B = [w1,w2,w3 ,w4,...,wn] ,A = [].
Final Config : S = [ROOT], B = [] ,A = [final set of dependency arcs].


Get the top word and second top word from stack 

capture labels

check based on tag whether a left shift, right shift or just a shift makes sense.

Accordingly add the arc and remove the word to which the arc points to 

Return the updated Configuration c

2. DependencyParser.py: getFeatures(...)
Based on the paper the set of features are accumulated as follows:
# set 1: get top 3 words of stack and buffer
# set 2: the first and second leftmost/rightmost children of the top 2 words on the stack
# set 3: the leftmost of the leftmost/rightmost of rightmost children of the top two words on the stack
append all the sets 1,2,3
get the word ids for 18 words, pos ids for 18 words and labelids for last 12 words (as per paper). append
all of these IDs together and form the feature set and return the collected features.

2. DependencyParser.py: forward_pass(...) [ for 1 hidden layer with default configuration ]
Simply compute the matrix multiplication of the input weights and the embeddings. To that
product/matrix add the bias.
Apply the non-linear cube activation function to it. Mark that as 'h'
Make predictions on data with the product of 'h' and weights output.
Return the predictions back to the calling function.

2. DependencyParser.py: build_graph(...)
setup training placeholders for input weights, biases and output
Lookup training embeddings and reshape them as applicable
make predictions on dev using the forwardpass method
calculate the cross entropy loss and l2 loss and apply the reduce mean.
continue with the function as already pre-defined with applying the optimizers and applying the
forwardpass on the test data to get the predictions on the test data.