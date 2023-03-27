# Forward forward 

Forward forward is a backpropgation alternative proposed in [this paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf) utilising 2 forward passes in a contrastive learning like scenario for training a neural network without the need to cache the entire computational graph like in backward propagation.

This is a minimal implementation of the paper.

`final_sup_checkpoint.pt` is the best checkpoint in a 90 epoch run using the forward forward algorithm and achieves 92% accuracy on MNIST.

## Resources
- [The Forward-Forward Algorithm: Some Preliminary Investigations](https://www.cs.toronto.edu/~hinton/FFA13.pdf)