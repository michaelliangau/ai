# RNN-T

Recurrent neural network transducer architecture for sequence to sequence model, commonly used in speech recognition.

This implementation works on text problem inserting missing vowels into a sentence.

## Resources
- [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
- [Sequence-to-sequence learning with Transducers](https://lorenlugosch.github.io/posts/2020/11/transducer/)
- [PyTorch implementation](https://github.com/lorenlugosch/transducer-tutorial/blob/main/transducer_tutorial_example.ipynb)

## How does RNN-T work.

`compute_loss` fn in the Trainer is the main function that runs the RNN-T algorithm. Learn this to learn everything.

RNN-T works like this:
- Dataloader feeds in data into the main train function in the form of (x, y, T, U) where
    - x: input text (B, T)
    - y: label (B, U)
    - T: length of input text (B)
    - U: length of label (B)
    And T doesn't have to equal U. In the notebook example, T is less than you and represents
    the length of the input text (vowels removed). U is the length of the label (original string).
- Encoder output: Input text is encoded by a Encoder network at each timestep (RNN, LSTM, GRU, etc). Shape: (batch, T, encoder_dim)
- Predictor output: Label text is encoded by Predictor network at eeach timestep (RNN, LSTM, GRU, etc). Shape: (batch, U, predictor_dim)
- Encoder and Predictor output are unsqueezed at the proper dimensions. Shape: (batch, T, 1, encoder_dim) and (batch, 1, U, predictor_dim)
- Joiner output: E and P outputs are fed into the Joiner network which adds the two tensors and pushes it through a ReLU non-linearity and linear network. Shape: (batch, T, U, num_outputs). num_outputs is the number of letters in the vocabulary (101 in the notebook example).
- Log softmax is run on the joiner_dim to get the log output probabilities for each letter at each encoder/predictor permutation. Negative numbers. Shape: (batch, T, U, num_outputs)
- Alignment algorithm (compute_forward_prob): acts on the log softmaxed Joiner output to get the summed forward probabilities of the label given all possible permutations of the joiner output (considering all permutations of the E/P network outputs). High negative number (sum logs). We want this number to be higher (negative number moving towards zero) to be a better network. Shape: (batch)
- Loss: Negative mean of the output of the alignment algorithm to get a mean (across batch) and positive number (representative of the badness of the probabilities of the output) to minimize through training.