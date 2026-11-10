# Extra credit: a GRU cell written gate by gate

The implementation computes the reset gate, update gate, candidate and hidden-state blend directly. It does not call nn.GRU or nn.GRUCell in its forward pass. nn.GRU is used only as the reference in the verification script.

## Run

With PyTorch installed, from the repository root:

```bash
python extra_credit/verify_gru.py
```

New results are written to extra_credit/rerun_results, preserving the recorded Colab results.

## Recorded Colab results

All 18 cases passed on CPU with PyTorch 2.14.1+cu130 and float64. Tests combine seeds 42, 1337 and 2026; sequence lengths 1, 7 and 32; and zero/random initial hidden states. Batch size is 3, input size 5 and hidden size 8. Both implementations receive identical copied weights, biases, inputs and initial states.

- Maximum output difference: 3.331e-16.
- Maximum final-hidden-state difference: 2.220e-16.
- Assertion tolerances: atol=1e-12, rtol=1e-10.

The CSV is transcribed from the Colab output shared in the conversation, retaining the displayed precision. The original full-precision files remain in /content/gru_extra_credit in that Colab session and should be downloaded if needed. The scripts reproduce the tested calculations as standalone Python files.

This verifies forward equivalence for the tested single-layer, unidirectional configuration with biases. It does not test gradients, GPU execution, multiple layers, packed sequences, or bidirectionality, and it does not retrain the language model.

## Why the reset placement matters

PyTorch computes the candidate as tanh(input_projection + reset * hidden_projection), where hidden_projection includes its bias. Applying reset to the hidden state before its matrix multiplication is a different GRU variant and generally will not match nn.GRU.

Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
