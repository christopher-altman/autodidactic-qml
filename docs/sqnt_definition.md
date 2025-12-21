# Superpositional Quantum Network Topologies (SQNT)

**Superpositional Quantum Network Topologies (SQNT)** refers to a formalism for adaptive network evolution where the network topology itself is treated as a quantum observable subject to superposition and measurement-induced collapse.

## Definition

In the context of this repository, SQNT provides the theoretical lineage for the **autodidactic loop**: a system that updates its own internal state (parameters/topology) based on self-measurement.

The term originates from the following canonical references:

1. **Altman, Pykacz & Zapatrin (2004)**: *"Superpositional Quantum Network Topologies"*. Defined the mathematical framework for networks where adjacency matrices are replaced by quantum operators, allowing for superposition of topological states.
2. **Altman & Zapatrin (2010)**: *"Backpropagation in Adaptive Quantum Networks"*. Extended the framework to include learning dynamics (backpropagation) on these superpositional substrates.

## Mapping to This Codebase

While the original SQNT formalism deals with quantum observables, this "Glass-Box Counterexample" implements the **logic** of the SQNT update loop using classical surrogates to test falsifiable claims about locality:

*   **Superpositional State**: Represented here by the **Matrix Model** (trace-class Hermitian matrices), which can be viewed as the density matrix or observable of the network topology.
*   **Measurement**: The **Correspondence Map** functions as the measurement operator, collapsing the matrix state into a specific recurrent neural network (RNN) wiring diagram for evaluation.
*   **Autodidactic Update**: The learning rule is driven by self-consistency (CI metric) rather than external supervision, mirroring the "self-measurement" dynamics of the original SQNT proposal.

## Usage in Code

In the codebase, you may encounter `sqnt` modules or variable names. These refer to:

*   `sqnt/`: Directory containing topology update logic and falsification protocols derived from the SQNT framework.
*   `sqnt_update`: Functions implementing the recursive self-modification steps.

**Note on Terminology:** Earlier versions of this work may have used "Superposed" or "Self-Quantized" as synonyms. The correct canonical term is **Superpositional Quantum Network Topologies**.
