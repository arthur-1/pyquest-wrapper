"""
- Each gate is one layer of the quantum circuit, may have an arbitrary number of controls (< n - 1)
- Each gate targets one qubit
-
[Gate1, Gate2, ...]


TODO =====================
-Support for non-parameterized gates in your ITE circuit
-Support for more complex Hamiltonians (e.g. sums of two basis circuit elements)


# IDEA
-Parameter iterator?
"""
from pyquest import Register
from pyquest.unitaries import *
from pyquest.unitaries import H, X, Y, Z, Rx, Ry, Rz, Unitary
from pyquest import Circuit


class GateTypes:
    X = 0
    Y = 1
    Z = 2
    H = 3
    Rx = 4
    Ry = 5
    Rz = 6

    @staticmethod
    def is_gate_type_parameterized(gate_type):
        return gate_type == GateTypes.Rx or gate_type == GateTypes.Ry or gate_type == GateTypes.Rz

    @staticmethod
    def rotation_gate_to_generator_gate(gate_type):
        if gate_type == GateTypes.Rx:
            return GateTypes.X
        elif gate_type == GateTypes.Ry:
            return GateTypes.Y
        elif gate_type == GateTypes.Rz:
            return GateTypes.Z
        else:
            raise RuntimeError

    @staticmethod
    def generator_gate_to_rotation_gate(gate_type):
        if gate_type == GateTypes.X:
            return GateTypes.Rx
        elif gate_type == GateTypes.Y:
            return GateTypes.Ry
        elif gate_type == GateTypes.Z:
            return GateTypes.Rz
        else:
            raise RuntimeError


class Gate:
    def __init__(self, qubit: int, gate_type: int, param: float=None, controls=None):
        self.qubit = qubit
        self.gate_type = gate_type
        self.param = param
        self.controls = controls

    def get_pyquest_gate(self):
        if self.gate_type == GateTypes.X:
            if self.controls is None:
                return X(self.qubit)
            else:
                return X(self.qubit, controls=self.controls)
        elif self.gate_type == GateTypes.Y:
            if self.controls is None:
                return Y(self.qubit)
            else:
                return Y(self.qubit, controls=self.controls)
        elif self.gate_type == GateTypes.Z:
            if self.controls is None:
                return Z(self.qubit)
            else:
                return Z(self.qubit, controls=self.controls)
        elif self.gate_type == GateTypes.Rx:
            if self.controls is None:
                return Rx(self.qubit, self.param)
            else:
                return Rx(self.qubit, self.param, controls=self.controls)
        elif self.gate_type == GateTypes.Ry:
            if self.controls is None:
                return Ry(self.qubit, self.param)
            else:
                return Ry(self.qubit, self.param, controls=self.controls)
        elif self.gate_type == GateTypes.Rz:
            if self.controls is None:
                return Rz(self.qubit, self.param)
            else:
                return Rz(self.qubit, self.param, controls=self.controls)
        raise NotImplementedError

    def is_parameterized(self):
        return GateTypes.is_gate_type_parameterized(self.gate_type)

    def copy(self):
        return Gate(self.qubit, self.gate_type, self.param, self.controls)


class QCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: [Gate] = []
        self.reg = Register(self.num_qubits)
        # The global phase parameter is used to track the global phase introduced by differentiating params in a circ
        self.__global_phase = 1

    def append(self, gate: Gate):
        self.gates.append(gate)

    def derivative_with_respect_to(self, param_index: int):
        """
        :param param_index: the index of the gate to differentiate.
        :return: A QCircuit object with the specified gate differentiated.
                 Currently, only support gates represented by exponentials whose derivative only produces one term.
        """
        assert 0 <= param_index < len(self.gates), "[ERROR] Invalid parameter index."
        # Check if the specified gate has a type whose derivative is currently supported

        gate = self.gates[param_index]
        assert gate.controls is None, "[UNSUPPORTED] Differentiating gates with controls is currently unsupported."
        assert gate.is_parameterized(), "[UNSUPPORTED] Differentiating nonparameterized gates is currently unsupported."
        assert gate.gate_type == GateTypes.Rx or gate.gate_type == GateTypes.Ry or gate.gate_type == GateTypes.Rz, \
            "[UNSUPPORTED] Parameterized gate type is currently unsupported."

        # TODO, IF Rx, X; if Ry, Y; if Rz, Z;
        new_gate = Gate(gate.qubit, GateTypes.rotation_gate_to_generator_gate(gate.gate_type))
        # self.gates.insert(param_index, new_gate)
        # Create a deep copy of the circuit object
        new_circ = self.copy()
        new_circ.gates.insert(param_index, new_gate)
        new_circ.__global_phase = -1j  # TODO Phase might be 1j instead of -1j depending on QuEST conventions
        return new_circ

    def copy(self):
        new_circ = QCircuit(self.num_qubits)
        for gate in self.gates:
            new_circ.append(gate.copy())
        return new_circ

    def get_pyquest_circ(self):
        gates = []
        for gate in self.gates:
            gates.append(gate.get_pyquest_gate())
        return Circuit(gates)

    def get_global_phase(self):
        return self.__global_phase

    def get_num_parameters(self):
        num_params = 0
        for gate in self.gates:
            if GateTypes.is_gate_type_parameterized(gate.gate_type):
                num_params += 1
        return num_params

    def get_parameter_vector(self):
        """
        :return: A numpy dx1 matrix, where d is the number of parameters in the circuit.
        """
        num_params = self.get_num_parameters()
        assert num_params > 0, "[ERROR] Invalid number of parameters in circuit."
        param_vec = np.zeros((num_params, 1)).astype(np.complex_)
        param_index = 0
        for gate in self.gates:
            if GateTypes.is_gate_type_parameterized(gate.gate_type):
                param_vec[param_index, 0] = gate.param
                param_index += 1
        return param_vec

    def get_i_th_paramterized_gate(self, i):
        assert 0 <= i <= self.get_num_parameters(), "[ERROR] Invalid parameter index given."
        j = 0
        for gate in self.gates:
            if gate.is_parameterized():
                if i == j:
                    return gate
                j += 1
        raise RuntimeError

    def set_parameter_vector(self, param_vec: np.array):
        assert param_vec.shape[0] == self.get_num_parameters(), "[ERROR] Given parameter vector of incorrect dimension."
        param_index = 0
        for gate in self.gates:
            if GateTypes.is_gate_type_parameterized(gate.gate_type):
                assert param_vec[param_index, 0].imag == 0, "[ERROR] Imaginary parameter."
                # The .real suppresses a warning that is caught by the above assertion
                gate.param = param_vec[param_index, 0].real
                param_index += 1


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





