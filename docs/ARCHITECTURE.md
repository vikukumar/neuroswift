# Architecture Notes

NeuroSwift is designed as a CPU-first language modeling experiment that favors linear sequence processing and sparse conditional computation.

Created by Vikash Kumar.

## Main Components

### 1. LinearSSM

`LinearSSM` replaces standard self-attention with a recurrent state-space scan.

Benefits:

- linear sequence complexity
- lower memory pressure than full attention
- more predictable CPU behavior on longer contexts

### 2. SparseMoE

`SparseMoE` routes each token to the top 2 experts out of 8 total experts.

Benefits:

- lower FLOPs per token
- conditional specialization
- better CPU efficiency than dense expert execution

### 3. HebbianUpdater

The plasticity module stores a small fast-weight state and updates it online using a Hebbian-style rule.

Benefits:

- short-term context adaptation during inference
- no full backpropagation step required
- a simple mechanism for dynamic memory

## Block Composition

Each `NeuroSwiftBlock` applies:

1. linear state-space mixing
2. sparse expert routing and expert execution
3. online plasticity adaptation

## Model Flow

1. tokens are embedded
2. features pass through a stack of NeuroSwift blocks
3. each block updates SSM state and optional plastic state
4. normalized hidden states are projected to logits

## Design Goals

- fast CPU experimentation
- minimal code for architecture research
- readable implementation suitable for extension
- explicit support for adaptive inference behavior
