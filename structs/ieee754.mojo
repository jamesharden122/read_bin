from collections import InlineArray

struct IEEE754Processor[N: Int]:
    var simd_dec_list: InlineArray[SIMD[DType.uint8,8], N]
    var n: Int
    fn __init__(out self, simd_dec_list: InlineArray[SIMD[DType.uint8,8], N]):
        self.simd_dec_list = simd_dec_list
        self.n = N
        

    # Convert N 64-bit patterns into N Float64 values (compile-time N)
    fn float_64_conversion(self) -> InlineArray[Float64, N]:
        var simd_processed_bits = self.process_simd_list(self.simd_dec_list)
        var merged_simd = InlineArray[Float64, N](uninitialized=True)
        for i in range(N):
            var temp_float = self.ieee_754_double_prec(simd_processed_bits[i])
            merged_simd[i] = temp_float[0]
        return merged_simd

    # Method to convert the binary exponent to decimal
    fn binary_to_decimal(self, exponent_simd: SIMD[DType.bool, 16]) -> SIMD[DType.float64, 1]:
        var powers_of_2 = SIMD[DType.int64, 16](1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0)
        var exponent_int_simd = exponent_simd.cast[DType.int64]()
        var decimal_value = exponent_int_simd.__mul__(powers_of_2)
        var dt_sum = decimal_value.reduce_add().__sub__(1023)
        var num2 = SIMD[DType.float64, 1](2.0)
        var out = num2 ** dt_sum.cast[DType.float64]()
        return out

    # Method to create the mantissa powers of 2 SIMD
    fn create_mantissa_powers_of_2_simd(self) -> SIMD[DType.float64, 64]:
        var powers_of_2_mantissa = SIMD[DType.float64, 64](
            2.0**-1,  2.0**-2,  2.0**-3,  2.0**-4,  2.0**-5,  2.0**-6,  2.0**-7,  2.0**-8,
            2.0**-9,  2.0**-10, 2.0**-11, 2.0**-12, 2.0**-13, 2.0**-14, 2.0**-15, 2.0**-16,
            2.0**-17, 2.0**-18, 2.0**-19, 2.0**-20, 2.0**-21, 2.0**-22, 2.0**-23, 2.0**-24,
            2.0**-25, 2.0**-26, 2.0**-27, 2.0**-28, 2.0**-29, 2.0**-30, 2.0**-31, 2.0**-32,
            2.0**-33, 2.0**-34, 2.0**-35, 2.0**-36, 2.0**-37, 2.0**-38, 2.0**-39, 2.0**-40,
            2.0**-41, 2.0**-42, 2.0**-43, 2.0**-44, 2.0**-45, 2.0**-46, 2.0**-47, 2.0**-48,
            2.0**-49, 2.0**-50, 2.0**-51, 2.0**-52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        #print("Powers of2: ", powers_of_2_mantissa)
        return powers_of_2_mantissa

    # Method to convert the binary mantissa to decimal
    fn mantissa_to_decimal(self, mantissa_simd: SIMD[DType.bool, 64]) -> SIMD[DType.float64, 1]:
        var mantissa_float_simd = mantissa_simd.cast[DType.float64]()
        var powers_of_2 = self.create_mantissa_powers_of_2_simd()
        #print(mantissa_float_simd[0]*powers_of_2_mantissa[0])
        var fractional_value_simd = mantissa_float_simd.__mul__(powers_of_2)
        var decimal_value = fractional_value_simd.reduce_add()
        return 1 + decimal_value

    # Method to convert a 64-bit binary representation to IEEE 754 double precision
    fn ieee_754_double_prec(self, simd_bool: SIMD[DType.bool, 64]) -> SIMD[DType.float64,1]:
        var extended_simd = SIMD[DType.bool, 128]()
        extended_simd = extended_simd.insert[offset=0](simd_bool)
        var sign_bit: SIMD[DType.float64, 1]
        if extended_simd[0] == False:
            sign_bit = 1.0
        else:
            sign_bit = -1.0

        var exponent_simd = extended_simd.slice[16, offset=1]()
        var mantissa_simd = extended_simd.slice[64, offset=12]()
        var temp1 = self.binary_to_decimal(exponent_simd)
        var temp2 = self.mantissa_to_decimal(mantissa_simd)
        var dt = sign_bit * temp1 * temp2
        #print("Full Bit: ", simd_bool.cast[DType.uint8]())
        #print("Sign Bit: ", extended_simd[0].cast[DType.uint8]())
        #print("Exponent: ", temp1, " ", exponent_simd.cast[DType.uint8]())
        #print("Mantissa: ",temp2," ", mantissa_simd.cast[DType.uint8]( ))
        return(dt)

    # Method to try converting a string to an integer
    fn try_convert_int(self, input_str: String) -> Bool:
        try:
            var value = input_str.__int__()
            return Bool(value)
        except:
            print("An error occurred")
            return False

    # Method to convert 8 uint8 elements into a 64-bit binary representation
    fn uint8_to_bin(self, simd_value: SIMD[DType.uint8, 8]) -> SIMD[DType.bool, 64]:
        var main_array = SIMD[DType.bool, 64]()
        for i in range(8):
            var tmp = bin(simd_value[i])
            print("Decimal to Binary Representation", tmp)#," ", len(tmp))
            var temp_simd = SIMD[DType.bool, 8]()
            # Handle various lengths of binary strings and convert them into SIMD vectors
            if len(tmp) == 8:
                temp_simd = SIMD[DType.bool, 8](
                    False, False,
                    self.try_convert_int(tmp[2].__getitem__(0)), self.try_convert_int(tmp[3].__getitem__(0)),
                    self.try_convert_int(tmp[4].__getitem__(0)), self.try_convert_int(tmp[5].__getitem__(0)),
                    self.try_convert_int(tmp[6].__getitem__(0)), self.try_convert_int(tmp[7].__getitem__(0))
                )
            elif len(tmp) == 9:
                temp_simd = SIMD[DType.bool, 8](
                    False, self.try_convert_int(tmp[2].__getitem__(0)),
                    self.try_convert_int(tmp[3].__getitem__(0)), self.try_convert_int(tmp[4].__getitem__(0)),
                    self.try_convert_int(tmp[5].__getitem__(0)), self.try_convert_int(tmp[6].__getitem__(0)),
                    self.try_convert_int(tmp[7].__getitem__(0)), self.try_convert_int(tmp[8].__getitem__(0))
                )
            elif len(tmp) == 10:
                temp_simd = SIMD[DType.bool, 8](
                    self.try_convert_int(tmp[2].__getitem__(0)), self.try_convert_int(tmp[3].__getitem__(0)),
                    self.try_convert_int(tmp[4].__getitem__(0)), self.try_convert_int(tmp[5].__getitem__(0)),
                    self.try_convert_int(tmp[6].__getitem__(0)), self.try_convert_int(tmp[7].__getitem__(0)),
                    self.try_convert_int(tmp[8].__getitem__(0)), self.try_convert_int(tmp[9].__getitem__(0))
                )
            elif len(tmp) == 3:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    False, False, self.try_convert_int(tmp[2].__getitem__(0))
                )
            elif len(tmp) == 5:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, False,
                    self.try_convert_int(tmp[2].__getitem__(0)),
                    self.try_convert_int(tmp[3].__getitem__(0)),
                    self.try_convert_int(tmp[4].__getitem__(0)),
            )
            elif len(tmp) == 4:  # Condition for 3-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, 
                    False, False, False, 
                    self.try_convert_int(tmp[2].__getitem__(0)),
                    self.try_convert_int(tmp[3].__getitem__(0))
                    )
            elif len(tmp) == 6:  # Condition for 6-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, False, 
                    self.try_convert_int(tmp[2].__getitem__(0)), self.try_convert_int(tmp[3].__getitem__(0)),
                    self.try_convert_int(tmp[4].__getitem__(0)), self.try_convert_int(tmp[5].__getitem__(0))
                )
            elif len(tmp) == 7:  # Condition for 7-bit binary string
                temp_simd = SIMD[DType.bool, 8](
                    False, False, False, self.try_convert_int(tmp[2].__getitem__(0)),
                    self.try_convert_int(tmp[3].__getitem__(0)), self.try_convert_int(tmp[4].__getitem__(0)),
                    self.try_convert_int(tmp[5].__getitem__(0)), self.try_convert_int(tmp[6].__getitem__(0))
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

    # Method to convert an int64 to a float32
    fn int64_to_float32(self, value: Int) -> Float32:
        return Float32(value)

    # Method to convert an int64 to a float64
    fn int64_to_float64(self, value: Int) -> Float64:
        return Float64(value)

    # Method to process InlineArray of SIMD[uint8, 8] values (compile-time N)
    fn process_simd_list(self, simd_list: InlineArray[SIMD[DType.uint8, 8], N]) -> InlineArray[SIMD[DType.bool, 64], N]:
        var temp_dts = InlineArray[SIMD[DType.bool, 64], N](uninitialized=True)
        for i in range(N):
            var temp_val = self.uint8_to_bin(simd_list[i])
            temp_dts[i] = temp_val
        return temp_dts
