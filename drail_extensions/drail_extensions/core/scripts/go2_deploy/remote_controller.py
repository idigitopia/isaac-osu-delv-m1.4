import struct


class RemoteController:
    def __init__(self):
        # Joystick states
        self.Lx = 0.0
        self.Ly = 0.0
        self.Rx = 0.0
        self.Ry = 0.0

        # Button states
        self.L1 = 0
        self.L2 = 0
        self.R1 = 0
        self.R2 = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.Up = 0
        self.Down = 0
        self.Left = 0
        self.Right = 0
        self.Start = 0
        self.Select = 0
        self.F1 = 0
        self.F3 = 0

    def parse_buttons(self, data1, data2):
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    def parse_axes(self, data):
        lx_offset = 4
        self.Lx = round(struct.unpack("<f", data[lx_offset : lx_offset + 4])[0], 2)  # noqa: E203
        rx_offset = 8
        self.Rx = round(struct.unpack("<f", data[rx_offset : rx_offset + 4])[0], 2)  # noqa: E203
        ry_offset = 12
        self.Ry = round(struct.unpack("<f", data[ry_offset : ry_offset + 4])[0], 2)  # noqa: E203
        # L2_offset = 16
        # L2 = struct.unpack('<f', data[L2_offset:L2_offset + 4])[0] # Placeholder，unused
        ly_offset = 20
        self.Ly = round(struct.unpack("<f", data[ly_offset : ly_offset + 4])[0], 2)  # noqa: E203

    def parse(self, data):
        self.parse_axes(data)
        self.parse_buttons(data[2], data[3])
