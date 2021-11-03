from pyquest_wrapper.qcircuit import QCircuit


class Hamiltonian:
    """
    Needs to be usable for appending to a circuit to compute expectation values.

    In this representation, a Hamiltonian is represented by a list of gates with associated weights.
    The hamiltonian is then the weighted sum of all the matrices the gates represent.
    """
    def __init__(self, num_qubits, weighted_gate_list: list):
        """
        :param num_qubits: the number of qubits this Hamiltonian acts on
        :param weighted_gate_list: [([gate, ..., gate], weight), ..., ([gate, ..., gate], weight)]
        """
        self.num_qubits = num_qubits
        self.gates = []
        self.weights = []
        for gatelist_weight_tuple in weighted_gate_list:
            weight = gatelist_weight_tuple[1]
            gatelist = gatelist_weight_tuple[0]
            assert len(gatelist) > 0, "[ERROR] Empty gate list given."
            for gate in gatelist:
                self.gates.append(gate)
                self.weights.append(weight)

        assert len(self.gates) == len(self.weights), "[ERROR] A weight must be specified for each gate."

    def compute_appended_inner_product(self, state1: QCircuit, state2: QCircuit):
        """
        :param state1: QCircuit representation of the ket
        :param state2: QCircuit representation of the bra
        :return: Computes <state1|H|state2>
        """
        output = 0
        assert len(self.gates) > 0, "[ERROR] Hamiltonian contains no terms."
        for i, gate in enumerate(self.gates):
            reg1 = Register(self.num_qubits)
            reg2 = Register(self.num_qubits)
            state2_temp = state2.copy()
            state2_temp.append(gate)
            reg1.apply_circuit(state1.get_pyquest_circ())
            reg2.apply_circuit(state2_temp.get_pyquest_circ())
            output += self.weights[i] * state2_temp.get_global_phase() * state1.get_global_phase() * (reg1 * reg2)
        return output

    def get_hamiltonian_matrix(self):
        """
        NOTE: This method should only be used for debugging small systems, and DOES NOT SCALE.
        :return: a numpy matrix object corresponding to the Hamiltonian's matrix.
        """
        raise NotImplementedError





