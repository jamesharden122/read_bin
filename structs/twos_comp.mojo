from std.collections import InlineArray

struct Int64TwosComp[N: Int]:
    var simd_dec_list: InlineArray[SIMD[DType.uint8, 8], Self.N]

    def __init__(out self, simd_dec_list: InlineArray[SIMD[DType.uint8, 8], Self.N]):
        self.simd_dec_list = simd_dec_list
        
    def int64_conversion(mut self) -> InlineArray[Int64, Self.N]:
        var out = InlineArray[Int64, Self.N](uninitialized=True)
        for i in range(Self.N):
            var bytes = self.simd_dec_list[i]
            var raw = (
                (UInt64(bytes[0]) << 56)
                | (UInt64(bytes[1]) << 48)
                | (UInt64(bytes[2]) << 40)
                | (UInt64(bytes[3]) << 32)
                | (UInt64(bytes[4]) << 24)
                | (UInt64(bytes[5]) << 16)
                | (UInt64(bytes[6]) << 8)
                | UInt64(bytes[7])
            )
            var sign_mask = UInt64(1) << 63
            if (raw & sign_mask) == 0:
                out[i] = Int64(raw)
            else:
                var magnitude = (~raw) + 1
                if magnitude == sign_mask:
                    out[i] = -9223372036854775807 - 1
                else:
                    out[i] = -Int64(magnitude)
        return out   

# Method to convert a 64-bit binary representation to IEEE 754 double precision
    def binary_to_int(self, simd_bool: SIMD[DType.bool, 64]) -> SIMD[DType.int64, 1]:
        if simd_bool[0] == False:
            sign_bit = 1.0
        else:
            sign_bit = -1.0
        var temp2 = simd_bool.cast[DType.float64]().__mul__(self.create_powers_of_2_simd())
        var dt = sign_bit * temp2
        return(dt.cast[DType.int64]().reduce_add())     
 
    # Method to process InlineArray of SIMD[uint8, 8] values
    def process_simd_list(mut self, simd_list: InlineArray[SIMD[DType.uint8, 8], Self.N]) -> InlineArray[SIMD[DType.bool, 64], Self.N]:
        var temp_dts = InlineArray[SIMD[DType.bool, 64], Self.N](uninitialized=True)
        for i in range(Self.N):
            var temp_val = self.uint8_to_bin(simd_list[i])
            temp_dts[i] = temp_val
        return temp_dts
   
    def uint8_to_bin(self, simd_value: SIMD[DType.uint8, 8]) -> SIMD[DType.bool, 64]:
        var main_array = SIMD[DType.bool, 64]()
        for i in range(8):
            var tmp = bin(simd_value[i])
            #print("Decimal to Binary Representation", tmp)#," ", len(tmp))
            var temp_simd = SIMD[DType.bool, 8]()
            # Handle various lengths of binary strings and convert them into SIMD vectors
            if len(tmp) == 8:
                temp_simd = SIMD[DType.bool, 8](
                    False, False,
                    self.try_convert_int(String(tmp[byte=2])), self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])), self.try_convert_int(String(tmp[byte=5])),
                    self.try_convert_int(String(tmp[byte=6])), self.try_convert_int(String(tmp[byte=7]))
                )
            elif len(tmp) == 9:
                temp_simd = SIMD[DType.bool, 8](
                    False, self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3])), self.try_convert_int(String(tmp[byte=4])),
                    self.try_convert_int(String(tmp[byte=5])), self.try_convert_int(String(tmp[byte=6])),
                    self.try_convert_int(String(tmp[byte=7])), self.try_convert_int(String(tmp[byte=8]))
                )
            elif len(tmp) == 10:
                temp_simd = SIMD[DType.bool, 8](
                    self.try_convert_int(String(tmp[byte=2])), self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])), self.try_convert_int(String(tmp[byte=5])),
                    self.try_convert_int(String(tmp[byte=6])), self.try_convert_int(String(tmp[byte=7])),
                    self.try_convert_int(String(tmp[byte=8])), self.try_convert_int(String(tmp[byte=9]))
                )
            elif len(tmp) == 3:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    False, False, self.try_convert_int(String(tmp[byte=2]))
                )
            elif len(tmp) == 4:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, 
                    False, False, False, 
                    self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3]))
                )
            elif len(tmp) == 5:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])),
                )
            elif len(tmp) == 6:  # Condition for 6-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, 
                    self.try_convert_int(String(tmp[byte=2])), self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])), self.try_convert_int(String(tmp[byte=5]))
                )
            elif len(tmp) == 7:  # Condition for 7-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3])), self.try_convert_int(String(tmp[byte=4])),
                    self.try_convert_int(String(tmp[byte=5])), self.try_convert_int(String(tmp[byte=6]))
                )

            if i == 0:
                main_array = main_array.insert[offset=0](temp_simd)
            if i == 1:
                main_array = main_array.insert[offset=8](temp_simd)
            if i == 2:
                main_array = main_array.insert[offset=16](temp_simd)
            if i == 3:
                main_array = main_array.insert[offset=24](temp_simd)
            if i == 4:
                main_array = main_array.insert[offset=32](temp_simd)
            if i == 5:
                main_array = main_array.insert[offset=40](temp_simd)
            if i == 6:
                main_array = main_array.insert[offset=48](temp_simd)
            if i == 7:
                main_array = main_array.insert[offset=56](temp_simd)
        return main_array


    # Method to create the mantissa powers of 2 SIMD
    def create_powers_of_2_simd(self) -> SIMD[DType.float64, 64]:
        var powers_of_2_decreasing = SIMD[DType.float64, 64](
            2.0**63,  2.0**62,  2.0**61,  2.0**60,  2.0**59,  2.0**58,  2.0**57,  2.0**56,  
            2.0**55,  2.0**54,  2.0**53,  2.0**52,  2.0**51,  2.0**50,  2.0**49,  2.0**48,  
            2.0**47,  2.0**46,  2.0**45,  2.0**44,  2.0**43,  2.0**42,  2.0**41,  2.0**40,
            2.0**39,  2.0**38,  2.0**37,  2.0**36,  2.0**35,  2.0**34,  2.0**33,  2.0**32,
            2.0**31,  2.0**30,  2.0**29,  2.0**28,  2.0**27,  2.0**26,  2.0**25,  2.0**24, 
            2.0**23,  2.0**22,  2.0**21,  2.0**20,  2.0**19,  2.0**18,  2.0**17,  2.0**16,
            2.0**15,  2.0**14,  2.0**13,  2.0**12,  2.0**11,  2.0**10,  2.0**9,   2.0**8,  
            2.0**7,   2.0**6,   2.0**5,   2.0**4,   2.0**3,   2.0**2,   2.0**1,   2.0**0
        )
        #print("Powers of2: ", powers_of_2_mantissa)
        return powers_of_2_decreasing

    # Method to try converting a string to an integer
    def try_convert_int(self, input_str: String) -> Bool:
        try:
            var value = input_str.__int__()
            return Bool(value)
        except:
            print("An error occurred")
            return False


struct Int32TwosComp[N: Int]:
    var simd_dec_list: InlineArray[SIMD[DType.uint8, 4], Self.N]

    def __init__(out self, simd_dec_list: InlineArray[SIMD[DType.uint8, 4], Self.N]):
        self.simd_dec_list = simd_dec_list
        
    def int32_conversion(mut self) -> InlineArray[Int32, Self.N]:
        var simd_processed_bits = self.process_simd_list(self.simd_dec_list)
        var out = InlineArray[Int32, Self.N](uninitialized=True)
        for i in range(Self.N):
            var temp_int = self.binary_to_int(simd_processed_bits[i])
            out[i] = temp_int[0]
        return out   

# Method to convert a 64-bit binary representation to IEEE 754 double precision
    def binary_to_int(self, simd_bool: SIMD[DType.bool, 32]) -> SIMD[DType.int32, 1]:
        if simd_bool[0] == False:
            sign_bit = 1.0
        else:
            sign_bit = -1.0
        var temp2 = simd_bool.cast[DType.float64]().__mul__(self.create_powers_of_2_simd())
        var dt = sign_bit * temp2
        #print("Full Bit: ", simd_bool.cast[DType.uint8]())
        #print("Sign Bit: ", extended_simd[0].cast[DType.uint8]())
        #print("Exponent: ", temp1, " ", exponent_simd.cast[DType.uint8]())
        #print("Mantissa: ",temp2," ", mantissa_simd.cast[DType.uint8]( ))
        return(dt.cast[DType.int32]().reduce_add())     
 
    # Method to process InlineArray of SIMD[uint8, 4] values
    def process_simd_list(self, simd_list: InlineArray[SIMD[DType.uint8, 4], Self.N]) -> InlineArray[SIMD[DType.bool, 32], Self.N]:
        var temp_dts = InlineArray[SIMD[DType.bool, 32], Self.N](uninitialized=True)
        for i in range(Self.N):
            var temp_val = self.uint8_to_bin(simd_list[i])
            temp_dts[i] = temp_val
        return temp_dts
   
    def uint8_to_bin(self, simd_value: SIMD[DType.uint8, 4]) -> SIMD[DType.bool, 32]:
        var main_array = SIMD[DType.bool, 32]()
        for i in range(4):
            var tmp = bin(simd_value[i])
            var temp_simd = SIMD[DType.bool, 8]()
            # Handle various lengths of binary strings and convert them into SIMD vectors
            if len(tmp) == 8:
                temp_simd = SIMD[DType.bool, 8](
                    False, False,
                    self.try_convert_int(String(tmp[byte=2])), self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])), self.try_convert_int(String(tmp[byte=5])),
                    self.try_convert_int(String(tmp[byte=6])), self.try_convert_int(String(tmp[byte=7]))
                )
            elif len(tmp) == 9:
                temp_simd = SIMD[DType.bool, 8](
                    False, self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3])), self.try_convert_int(String(tmp[byte=4])),
                    self.try_convert_int(String(tmp[byte=5])), self.try_convert_int(String(tmp[byte=6])),
                    self.try_convert_int(String(tmp[byte=7])), self.try_convert_int(String(tmp[byte=8]))
                )
            elif len(tmp) == 10:
                temp_simd = SIMD[DType.bool, 8](
                    self.try_convert_int(String(tmp[byte=2])), self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])), self.try_convert_int(String(tmp[byte=5])),
                    self.try_convert_int(String(tmp[byte=6])), self.try_convert_int(String(tmp[byte=7])),
                    self.try_convert_int(String(tmp[byte=8])), self.try_convert_int(String(tmp[byte=9]))
                )
            elif len(tmp) == 3:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    False, False, self.try_convert_int(String(tmp[byte=2]))
                )
            elif len(tmp) == 6:  # Condition for 6-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, 
                    self.try_convert_int(String(tmp[byte=2])), self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])), self.try_convert_int(String(tmp[byte=5]))
                )
            elif len(tmp) == 7:  # Condition for 7-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3])), self.try_convert_int(String(tmp[byte=4])),
                    self.try_convert_int(String(tmp[byte=5])), self.try_convert_int(String(tmp[byte=6]))
                )
            elif len(tmp) == 4:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, 
                    False, False, False, 
                    self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3]))
                    )
            elif len(tmp) == 5:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    self.try_convert_int(String(tmp[byte=2])),
                    self.try_convert_int(String(tmp[byte=3])),
                    self.try_convert_int(String(tmp[byte=4])),
                    )
            else:
                print("Decimal to Binary Representation", tmp,":", len(tmp))
            if i == 0:
                main_array = main_array.insert[offset=0](temp_simd)
            if i == 1:
                main_array = main_array.insert[offset=8](temp_simd)
            if i == 2:
                main_array = main_array.insert[offset=16](temp_simd)
            if i == 3:
                main_array = main_array.insert[offset=24](temp_simd)
        return main_array

    # Method to try converting a string to an integer
    def try_convert_int(self, input_str: String) -> Bool:
        try:
            var value = input_str.__int__()
            return Bool(value)
        except:
            print("An error occurred")
            return False

    def create_powers_of_2_simd(self) -> SIMD[DType.float64, 32]:
        var powers_of_2_decreasing = SIMD[DType.float64, 32](   
            
            2.0**31,  2.0**30,  2.0**29,  2.0**28,  2.0**27,  2.0**26,  2.0**25,  2.0**24, 
            2.0**23,  2.0**22,  2.0**21,  2.0**20,  2.0**19,  2.0**18,  2.0**17,  2.0**16,
            2.0**15,  2.0**14,  2.0**13,  2.0**12,  2.0**11,  2.0**10,  2.0**9,   2.0**8,  
            2.0**7,   2.0**6,   2.0**5,   2.0**4,   2.0**3,   2.0**2,   2.0**1,  2.0**0
         )
        #print("Powers of2: ", powers_of_2_mantissa)
        return powers_of_2_decreasing
