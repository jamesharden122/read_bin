from max.tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index

struct Uint64TwosComp:
    var simd_dec_list: List[SIMD[DType.uint8,8]]
    fn __init__(out self, simd_dec_list: List[SIMD[DType.uint8,8]]):
        self.simd_dec_list = simd_dec_list
        
    fn uint64_conversion(self) -> Tensor[DType.uint64]:
        var simd_processed_bits = self.process_simd_list(self.simd_dec_list)
        print(len(simd_processed_bits))
        var spec = TensorSpec(DType.uint64,len(simd_processed_bits),1)
        var tensor_col = Tensor[DType.uint64](spec)
        for i in range(len(self.simd_dec_list)):
            var temp_float = self.binary_to_int(simd_processed_bits[i])
            tensor_col[Index(i)] = temp_float
        return tensor_col    

# Method to convert a 64-bit binary representation to IEEE 754 double precision
    fn binary_to_int(self, simd_bool: SIMD[DType.bool, 64]) -> SIMD[DType.uint64,1]:
        var dt = simd_bool.cast[DType.float64]().__mul__(self.create_powers_of_2_simd())
        #print("Full Bit: ", simd_bool.cast[DType.uint8]())
        #print("Sign Bit: ", extended_simd[0].cast[DType.uint8]())
        #print("Exponent: ", temp1, " ", exponent_simd.cast[DType.uint8]())
        #print("Mantissa: ",temp2," ", mantissa_simd.cast[DType.uint8]( ))
        return(dt.cast[DType.uint64]().reduce_add())     
 
    # Method to process a list of SIMD[uint8, 8] values
    fn process_simd_list(self, simd_list: List[SIMD[DType.uint8, 8]]) -> List[SIMD[DType.bool, 64]]:
        var temp_dts = List[SIMD[DType.bool, 64]]()
        for i in range(len(simd_list)):
            var temp_val = self.uint8_to_bin(simd_list[i])
            temp_dts.append(temp_val)
        return temp_dts
   
    fn uint8_to_bin(self, simd_value: SIMD[DType.uint8, 8]) -> SIMD[DType.bool, 64]:
        var main_array = SIMD[DType.bool, 64]()
        for i in range(8):
            var tmp = bin(simd_value[i])
            var temp_simd = SIMD[DType.bool, 8]()
            # Handle various lengths of binary strings and convert them into SIMD vectors
            if len(tmp) == 8:
                temp_simd = SIMD[DType.bool, 8](
                    False, False,
                    self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5]),
                    self.try_convert_int(tmp[6]), self.try_convert_int(tmp[7])
                )
            elif len(tmp) == 9:
                temp_simd = SIMD[DType.bool, 8](
                    False, self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3]), self.try_convert_int(tmp[4]),
                    self.try_convert_int(tmp[5]), self.try_convert_int(tmp[6]),
                    self.try_convert_int(tmp[7]), self.try_convert_int(tmp[8])
                )
            elif len(tmp) == 10:
                temp_simd = SIMD[DType.bool, 8](
                    self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5]),
                    self.try_convert_int(tmp[6]), self.try_convert_int(tmp[7]),
                    self.try_convert_int(tmp[8]), self.try_convert_int(tmp[9])
                )
            elif len(tmp) == 3:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    False, False, self.try_convert_int(tmp[2])
                )
            elif len(tmp) == 5:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]),
                    )
            elif len(tmp) == 6:  # Condition for 6-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, 
                    self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5])
                )
            elif len(tmp) == 7:  # Condition for 7-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3]), self.try_convert_int(tmp[4]),
                    self.try_convert_int(tmp[5]), self.try_convert_int(tmp[6])
                )
            elif len(tmp) == 4:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, 
                    False, False, False, 
                    self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3])
                    )
            else:
                print("Decimal to Binary Representation", tmp)#," ", len(tmp))
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

    # Method to try converting a string to an integer
    fn try_convert_int(self, input_str: String) -> Bool:
        try:
            var value = input_str.__int__()
            return Bool(value)
        except:
            print("An error occurred")
            return False

    # Method to create the mantissa powers of 2 SIMD
    fn create_powers_of_2_simd(self) -> SIMD[DType.float64, 64]:
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



struct Uint32TwosComp:
    var simd_dec_list: List[SIMD[DType.uint8,4]]
    fn __init__(out self, simd_dec_list: List[SIMD[DType.uint8,4]]):
        self.simd_dec_list = simd_dec_list
        
    fn uint32_conversion(self) -> Tensor[DType.uint32]:
        var simd_processed_bits = self.process_simd_list(self.simd_dec_list)
        var spec = TensorSpec(DType.uint32,len(simd_processed_bits),1)
        var tensor_col = Tensor[DType.uint32](spec)
        for i in range(len(self.simd_dec_list)):
            var temp_float = self.binary_to_int(simd_processed_bits[i])
            tensor_col[Index(i,0)] = temp_float
        return tensor_col    

# Method to convert a 64-bit binary representation to IEEE 754 double precision
    fn binary_to_int(self, simd_bool: SIMD[DType.bool, 32]) -> SIMD[DType.uint32,1]:
        var dt = simd_bool.cast[DType.float64]().__mul__(self.create_powers_of_2_simd())
        #print("Full Bit: ", simd_bool.cast[DType.uint8]())
        #print("Sign Bit: ", extended_simd[0].cast[DType.uint8]())
        #print("Exponent: ", temp1, " ", exponent_simd.cast[DType.uint8]())
        #print("Mantissa: ",temp2," ", mantissa_simd.cast[DType.uint8]( ))
        return(dt.cast[DType.uint32]().reduce_add())     
 
    # Method to process a list of SIMD[uint8, 8] values
    fn process_simd_list(self, simd_list: List[SIMD[DType.uint8, 4]]) -> List[SIMD[DType.bool, 32]]:
        var temp_dts = List[SIMD[DType.bool, 32]]()
        for i in range(len(simd_list)):
            var temp_val = self.uint8_to_bin(simd_list[i])
            temp_dts.append(temp_val)
        return temp_dts
   
    fn uint8_to_bin(self, simd_value: SIMD[DType.uint8, 4]) -> SIMD[DType.bool, 32]:
        var main_array = SIMD[DType.bool, 32]()
        for i in range(4):
            var tmp = bin(simd_value[i])
            var temp_simd = SIMD[DType.bool, 8]()
            # Handle various lengths of binary strings and convert them into SIMD vectors
            if len(tmp) == 8:
                temp_simd = SIMD[DType.bool, 8](
                    False, False,
                    self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5]),
                    self.try_convert_int(tmp[6]), self.try_convert_int(tmp[7])
                )
            elif len(tmp) == 9:
                temp_simd = SIMD[DType.bool, 8](
                    False, self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3]), self.try_convert_int(tmp[4]),
                    self.try_convert_int(tmp[5]), self.try_convert_int(tmp[6]),
                    self.try_convert_int(tmp[7]), self.try_convert_int(tmp[8])
                )
            elif len(tmp) == 10:
                temp_simd = SIMD[DType.bool, 8](
                    self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5]),
                    self.try_convert_int(tmp[6]), self.try_convert_int(tmp[7]),
                    self.try_convert_int(tmp[8]), self.try_convert_int(tmp[9])
                )
            elif len(tmp) == 3:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    False, False, self.try_convert_int(tmp[2])
                )
            elif len(tmp) == 5:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]),
                    )
            elif len(tmp) == 6:  # Condition for 6-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, 
                    self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                    self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5])
                )
            elif len(tmp) == 7:  # Condition for 7-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3]), self.try_convert_int(tmp[4]),
                    self.try_convert_int(tmp[5]), self.try_convert_int(tmp[6])
                )
            elif len(tmp) == 4:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, 
                    False, False, False, 
                    self.try_convert_int(tmp[2]),
                    self.try_convert_int(tmp[3])
                    )
            else:
                print("Decimal to Binary Representation", tmp)#," ", len(tmp))
            if i == 0:
                main_array = main_array.insert[offset=0](temp_simd)
            if i == 1:
                main_array = main_array.insert[offset=8](temp_simd)
            if i == 2:
                main_array = main_array.insert[offset=16](temp_simd)
            if i == 3:
                main_array = main_array.insert[offset=24](temp_simd)
        return main_array


    # Method to create the mantissa powers of 2 SIMD
    fn create_powers_of_2_simd(self) -> SIMD[DType.float64, 32]:
        var powers_of_2_decreasing = SIMD[DType.float64, 32](   
            
            2.0**31,  2.0**30,  2.0**29,  2.0**28,  2.0**27,  2.0**26,  2.0**25,  2.0**24, 
            2.0**23,  2.0**22,  2.0**21,  2.0**20,  2.0**19,  2.0**18,  2.0**17,  2.0**16,
            2.0**15,  2.0**14,  2.0**13,  2.0**12,  2.0**11,  2.0**10,  2.0**9,   2.0**8,  
            2.0**7,   2.0**6,   2.0**5,   2.0**4,   2.0**3,   2.0**2,   2.0**1,  2.0**0
         )
        #print("Powers of2: ", powers_of_2_mantissa)
        return powers_of_2_decreasing

    # Method to try converting a string to an integer
    fn try_convert_int(self, input_str: String) -> Bool:
        try:
            var value = input_str.__int__()
            return Bool(value)
        except:
            print("An error occurred")
            return False

struct Uint8TwosComp:
    var simd_dec_list: List[SIMD[DType.uint8,1]]
    fn __init__(out self, simd_dec_list: List[SIMD[DType.uint8,1]]):
        self.simd_dec_list = simd_dec_list
        
    fn uint8_conversion(self) -> Tensor[DType.uint8]:
        var spec = TensorSpec(DType.uint64,len(self.simd_dec_list),1)
        var tensor_col = Tensor[DType.uint8](spec)
        for i in range(len(self.simd_dec_list)):
            tensor_col[Index(i)] = self.simd_dec_list[i]
        return tensor_col    

    # Method to convert a 64-bit binary representation to IEEE 754 double precision
    fn binary_to_int(self, simd_bool: SIMD[DType.bool, 8]) -> SIMD[DType.uint8,1]:
        var twos_vec = SIMD[DType.float64, 8](
            2.0**7,   2.0**6,   2.0**5,   2.0**4,   2.0**3,   2.0**2,   2.0**1,  2.0**0
         )
        var dt = simd_bool.cast[DType.float64]().__mul__(twos_vec) 
        #print("Full Bit: ", simd_bool.cast[DType.uint8]())
        #print("Sign Bit: ", extended_simd[0].cast[DType.uint8]())
        #print("Exponent: ", temp1, " ", exponent_simd.cast[DType.uint8]())
        #print("Mantissa: ",temp2," ", mantissa_simd.cast[DType.uint8]( ))
        return(dt.cast[DType.uint8]().reduce_add())     

    fn uint8_to_bin(self, simd_value: SIMD[DType.uint8, 1]) -> SIMD[DType.bool, 8]:
        var tmp = bin(simd_value)
        var temp_simd = SIMD[DType.bool, 8]()
          # Handle various lengths of binary strings and convert them into SIMD vectors
        if len(tmp) == 8:
             temp_simd = SIMD[DType.bool, 8](
                False, False,
                self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5]),
                self.try_convert_int(tmp[6]), self.try_convert_int(tmp[7])
            )
        elif len(tmp) == 9:
             temp_simd = SIMD[DType.bool, 8](
                False, self.try_convert_int(tmp[2]),
                self.try_convert_int(tmp[3]), self.try_convert_int(tmp[4]),
                self.try_convert_int(tmp[5]), self.try_convert_int(tmp[6]),
                self.try_convert_int(tmp[7]), self.try_convert_int(tmp[8])
            )
        elif len(tmp) == 10:
             temp_simd = SIMD[DType.bool, 8](
                self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5]),
                self.try_convert_int(tmp[6]), self.try_convert_int(tmp[7]),
                self.try_convert_int(tmp[8]), self.try_convert_int(tmp[9])
           )
        elif len(tmp) == 3:  # Condition for 3-bit binary string
             temp_simd = SIMD[DType.bool, 8](
                False, False, False, False, False,
                False, False, self.try_convert_int(tmp[2])
          )
        elif len(tmp) == 5:  # Condition for 3-bit binary string
            temp_simd = SIMD[DType.bool, 8](
                False, False, False, False, False,
                self.try_convert_int(tmp[2]),
                self.try_convert_int(tmp[3]),
                self.try_convert_int(tmp[4]),
                )
        elif len(tmp) == 6:  # Condition for 6-bit binary string
             temp_simd = SIMD[DType.bool, 8](
                  False, False, False, False, 
                  self.try_convert_int(tmp[2]), self.try_convert_int(tmp[3]),
                  self.try_convert_int(tmp[4]), self.try_convert_int(tmp[5])
          )
        elif len(tmp) == 7:  # Condition for 7-bit binary string
             temp_simd = SIMD[DType.bool, 8](
                  False, False, False, self.try_convert_int(tmp[2]),
                  self.try_convert_int(tmp[3]), self.try_convert_int(tmp[4]),
                  self.try_convert_int(tmp[5]), self.try_convert_int(tmp[6])
         )
        elif len(tmp) == 4:  # Condition for 3-bit binary string
            temp_simd = SIMD[DType.bool, 8](
                 False, False, False, 
                 False, False, False, 
                 self.try_convert_int(tmp[2]),
                 self.try_convert_int(tmp[3])
        )
        else:
            print("Decimal to Binary Representation", tmp)#," ", len(tmp))
        return temp_simd

    # Method to try converting a string to an integer
    fn try_convert_int(self, input_str: String) -> Bool:
        try:
            var value = input_str.__int__()
            return Bool(value)
        except:
            print("An error occurred")
            return False

