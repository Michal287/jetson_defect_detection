import snap7


class Connector:
    def __init__(self, ip, port, data_block):
        self.client = snap7.client.Client()
        self.client.connect(ip, port, data_block)

    def is_available(self):
        return self.client.get_connected()

    def read(self, db_number, start, length=1, inverted=False):
        byte_data = self.client.db_read(db_number, start, length)
        bits_list = []

        for byte in byte_data:
            byte_bits = [int(bit) for bit in f'{byte:08b}']

            if inverted:
                byte_bits = byte_bits[::-1]

            bits_list.append(byte_bits)
        return bits_list

    @staticmethod
    def bits_to_bytes(bits_list):
        # Funkcja, która przekształca listę bitów na bajty.
        bytes_data = []
        for bits in bits_list:
            byte_str = ''.join(map(str, bits))
            byte_value = int(byte_str, 2)
            bytes_data.append(byte_value.to_bytes(1, byteorder='big'))
        return b''.join(bytes_data)

    def write(self, db_number, start, data, inverted=False):
        if inverted:
            data = [bits[::-1] for bits in data]

        bytes_to_write = self.bits_to_bytes(data)
        self.client.db_write(db_number, start, bytes_to_write)


# Create custom function for your problem
class StabConnector(Connector):
    def __init__(self, ip, port=0, data_block=1):
        super(StabConnector, self).__init__(ip, port, data_block)
        self.data = None

    def enableVisionCheck(self):
        self.data = self.read(db_number=1, start=0, length=2, inverted=True)
        return self.data[0][4] == 1

    def elementIsValid(self):
        # Sygnał czy jest paletka
        self.data[0][1] = 1
        self.data[0][2] = 0
        # Czy element jest wadliwy
        self.data[0][5] = 0

        self.write(db_number=1, start=0, data=self.data, inverted=True)

    def elementIsInvalid(self):
        # Sygnał czy jest paletka
        self.data[0][1] = 0
        self.data[0][2] = 1
        # Czy element jest wadliwy
        self.data[0][5] = 1

        self.write(db_number=1, start=0, data=self.data, inverted=True)



