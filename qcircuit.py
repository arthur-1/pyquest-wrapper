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
from pyquest import Circuit
from pyquest_wrapper.gate import Gate, GateTypes


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


