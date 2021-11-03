from pyquest.unitaries import H, X, Y, Z, Rx, Ry, Rz, Unitary


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
