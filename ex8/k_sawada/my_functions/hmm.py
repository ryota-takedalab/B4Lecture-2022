import numpy as np


class HMM:
    def __init__(self, states, initial_state_probability,
                 transition_probability, output_probability):
        """HMM

        Args:
            states (int):
                number of states
            initial_state_probability (ndarray, axis=(state)):
                initial state probability array
            transition_probability (ndarray, axis=(before, after)):
                transition probability matrix
            output_probability (ndarray, axix=(state, output label)):
                output probability matrix
        """
        self.states = states
        self.initial_state_probability = initial_state_probability
        self.transition_probability = transition_probability
        self.output_probability = output_probability
    
    def forward(self, outputs):
        """likelihoods based on Forward algorithm
        likelihoods is equal to joint probability

        Args:
            outputs (ndarray, axis=(time)):
                outputs series

        Returns:
            float:
                likelihoods
        """
        length = len(outputs)
        alpha = np.zeros((length, self.states))
        
        # base stage
        alpha[0] = self.initial_state_probability * \
            self.output_probability[:, outputs[0]]
        
        # recursion stage
        for t in range(1, length):
            alpha[t] = (self.transition_probability @
                        alpha[t - 1].reshape((self.states, 1))).flatten() * \
                self.output_probability[:, outputs[t]]
        return np.sum(alpha[-1])
    
    def viterbi(self, outputs):
        """likelihoods based on Viterbi algorithm
        likelihoods is equal to joint probability

        Args:
            outputs (ndarray, axis=(time)):
                outputs series

        Returns:
            float:
                likelihoods
        """
        length = len(outputs)
        psi_probabiolity = np.zeros((length, self.states))
        psi_states = np.zeros((length, self.states))

        # base stage
        psi_probabiolity[0] = self.initial_state_probability * \
            self.output_probability[:, outputs[0]]
        # psi_states[0] is already zeros

        # recursion stage
        for t in range(1, length):
            psi_probabiolity[t] = \
                np.max(
                    self.transition_probability *
                    psi_probabiolity[t - 1].reshape(self.states, 1), axis=1) * \
                self.output_probability[:, [outputs[t]]].flatten()
            psi_states[t] = \
                np.argmax(
                    self.transition_probability *
                    psi_probabiolity[t - 1].reshape(self.states, 1), axis=1)
        return np.max(psi_probabiolity[-1])
