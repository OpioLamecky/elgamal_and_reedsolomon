import numpy as np
from typing import List, Tuple, Union
import random

class HammingCode:
    """Hamming(7,4) Code Implementation"""
    
    def __init__(self):
        # Generator matrix for Hamming(7,4)
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], dtype=int)
        
        # Parity check matrix
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ], dtype=int)
        
        # Syndrome to error position mapping
        self.syndrome_table = {
            (1, 1, 0): 0,  # p1 error
            (1, 0, 1): 1,  # p2 error
            (0, 1, 1): 2,  # p3 error
            (1, 1, 1): 3,  # d1 error
            (1, 0, 0): 4,  # d2 error
            (0, 1, 0): 5,  # d3 error
            (0, 0, 1): 6   # d4 error
        }
    
    def encode(self, data: List[int]) -> List[int]:
        """Encode 4-bit data into 7-bit codeword"""
        if len(data) != 4:
            raise ValueError("Data must be 4 bits for Hamming(7,4)")
        
        data_vector = np.array(data, dtype=int)
        codeword = np.dot(data_vector, self.G) % 2
        return codeword.tolist()
    
    def decode(self, received: List[int]) -> Tuple[List[int], bool, int]:
        """Decode 7-bit received word, correct single errors"""
        if len(received) != 7:
            raise ValueError("Received word must be 7 bits for Hamming(7,4)")
        
        received_vector = np.array(received, dtype=int)
        
        # Calculate syndrome
        syndrome = np.dot(self.H, received_vector) % 2
        syndrome_tuple = tuple(syndrome)
        
        # Check for errors
        if np.array_equal(syndrome, [0, 0, 0]):
            # No errors
            decoded_data = [received[2], received[4], received[5], received[6]]
            return decoded_data, False, -1
        
        # Correct single error
        if syndrome_tuple in self.syndrome_table:
            error_pos = self.syndrome_table[syndrome_tuple]
            corrected = received_vector.copy()
            corrected[error_pos] = 1 - corrected[error_pos]  # Flip the bit
            
            decoded_data = [corrected[2], corrected[4], corrected[5], corrected[6]]
            return decoded_data, True, error_pos
        
        # Multiple errors detected but cannot correct
        decoded_data = [received[2], received[4], received[5], received[6]]
        return decoded_data, False, -2


class ReedSolomonCode:
    """Reed-Solomon Code Implementation (simplified version)"""
    
    def __init__(self, n: int = 7, k: int = 4, field_size: int = 8):
        """
        RS(n,k) code over GF(2^m)
        n: codeword length
        k: message length
        field_size: m for GF(2^m)
        """
        self.n = n
        self.k = k
        self.t = (n - k) // 2  # error correction capability
        self.field_size = field_size
        
        # Generator polynomial (simplified)
        self.generator = self._create_generator_polynomial()
    
    def _gf_multiply(self, a: int, b: int) -> int:
        """Multiply two numbers in GF(2^3)"""
        result = 0
        for _ in range(self.field_size):
            if b & 1:
                result ^= a
            hi_bit_set = a & 0x4  # For GF(8), check x^2 bit
            a <<= 1
            if hi_bit_set:
                a ^= 0x3  # x^3 + x + 1 = 0b1011 -> 0x3
            b >>= 1
        return result & 0x7
    
    def _create_generator_polynomial(self) -> List[int]:
        """Create generator polynomial for RS code"""
        # For simplified demo, use fixed generator
        return [1, 3, 2, 1]  # g(x) = x^3 + 3x^2 + 2x + 1
    
    def encode(self, message: List[int]) -> List[int]:
        """Encode message into Reed-Solomon codeword"""
        if len(message) != self.k:
            raise ValueError(f"Message must be {self.k} symbols")
        
        # Pad message with zeros
        message_poly = message + [0] * (self.n - self.k)
        
        # Simple polynomial division (for demo)
        codeword = message_poly.copy()
        
        # Add redundancy (simplified)
        for i in range(self.k):
            coeff = codeword[i]
            if coeff != 0:
                for j in range(1, len(self.generator)):
                    if self.generator[j] != 0:
                        codeword[i + j] ^= self._gf_multiply(self.generator[j], coeff)
        
        return codeword
    
    def decode(self, received: List[int]) -> Tuple[List[int], bool, List[int]]:
        """Decode received word and attempt error correction"""
        if len(received) != self.n:
            raise ValueError(f"Received word must be {self.n} symbols")
        
        # Calculate syndromes (simplified)
        syndromes = []
        for i in range(2 * self.t):
            syndrome = 0
            for j in range(self.n):
                syndrome ^= self._gf_multiply(received[j], (i + 1) % 8)
            syndromes.append(syndrome)
        
        # Check for errors
        has_errors = any(s != 0 for s in syndromes)
        
        if not has_errors:
            return received[:self.k], False, []
        
        # Simplified error correction (for demo)
        corrected = received.copy()
        errors_corrected = []
        
        # Try to correct single error
        for pos in range(self.n):
            test_word = received.copy()
            # Try flipping each symbol
            for new_val in range(8):
                if new_val != received[pos]:
                    test_word[pos] = new_val
                    # Check if this fixes syndromes
                    fixed = True
                    for i in range(2 * self.t):
                        test_syndrome = 0
                        for j in range(self.n):
                            test_syndrome ^= self._gf_multiply(test_word[j], (i + 1) % 8)
                        if test_syndrome != 0:
                            fixed = False
                            break
                    
                    if fixed:
                        corrected = test_word
                        errors_corrected.append(pos)
                        return corrected[:self.k], True, errors_corrected
        
        return received[:self.k], False, []


class InteractiveECCSystem:
    """Interactive Error-Correcting Code System with User Input"""
    
    def __init__(self):
        self.hamming = HammingCode()
        self.reed_solomon = ReedSolomonCode()
    
    def text_to_binary(self, text: str) -> List[List[int]]:
        """Convert text to 4-bit blocks for Hamming encoding"""
        binary_blocks = []
        for char in text:
            # Convert character to 8-bit binary
            binary_char = format(ord(char), '08b')
            # Split into two 4-bit blocks
            block1 = [int(bit) for bit in binary_char[:4]]
            block2 = [int(bit) for bit in binary_char[4:]]
            binary_blocks.extend([block1, block2])
        return binary_blocks
    
    def binary_to_text(self, binary_blocks: List[List[int]]) -> str:
        """Convert 4-bit blocks back to text"""
        text = ""
        for i in range(0, len(binary_blocks), 2):
            if i + 1 < len(binary_blocks):
                # Combine two 4-bit blocks into one 8-bit character
                block1 = binary_blocks[i]
                block2 = binary_blocks[i + 1]
                binary_char = ''.join(str(bit) for bit in block1 + block2)
                char_code = int(binary_char, 2)
                if 32 <= char_code <= 126:  # Printable ASCII range
                    text += chr(char_code)
        return text
    
    def introduce_errors(self, data: List[int], num_errors: int = 1) -> List[int]:
        """Introduce random errors into data"""
        corrupted = data.copy()
        positions = random.sample(range(len(data)), min(num_errors, len(data)))
        
        for pos in positions:
            if isinstance(corrupted[pos], int):
                # For binary data (Hamming)
                corrupted[pos] = 1 - corrupted[pos]
            else:
                # For symbol data (Reed-Solomon)
                corrupted[pos] = (corrupted[pos] + random.randint(1, 7)) % 8
        
        return corrupted
    
    def display_binary_blocks(self, binary_blocks: List[List[int]], title: str = "Binary Blocks"):
        """Display binary blocks in a formatted way"""
        print(f"\n{title}:")
        print("-" * 50)
        for i, block in enumerate(binary_blocks):
            print(f"Block {i+1}: {block}")
    
    def display_encoded_blocks(self, encoded_blocks: List[List[int]], title: str = "Encoded Blocks"):
        """Display encoded blocks in a formatted way"""
        print(f"\n{title}:")
        print("-" * 50)
        for i, block in enumerate(encoded_blocks):
            print(f"Encoded Block {i+1}: {block}")
    
    def display_corrupted_blocks(self, corrupted_blocks: List[List[int]], title: str = "Corrupted Blocks"):
        """Display corrupted blocks in a formatted way"""
        print(f"\n{title}:")
        print("-" * 50)
        for i, block in enumerate(corrupted_blocks):
            print(f"Corrupted Block {i+1}: {block}")
    
    def interactive_hamming_demo(self):
        """Interactive Hamming code demonstration with user input"""
        print("=" * 60)
        print("HAMMING CODE INTERACTIVE DEMONSTRATION")
        print("=" * 60)
        
        # Get user input
        user_message = input("\nEnter your message to encode: ")
        
        if not user_message:
            user_message = "Hello World"  # Default message
        
        print(f"\nOriginal message: '{user_message}'")
        
        # Step 1: Convert to binary blocks
        binary_blocks = self.text_to_binary(user_message)
        self.display_binary_blocks(binary_blocks, "STEP 1: Original Message as Binary Blocks")
        
        # Step 2: Encode blocks
        encoded_blocks = []
        for i, block in enumerate(binary_blocks):
            encoded = self.hamming.encode(block)
            encoded_blocks.append(encoded)
        
        self.display_encoded_blocks(encoded_blocks, "STEP 2: Encoded Blocks with Hamming(7,4)")
        
        # Step 3: Introduce errors
        error_choice = input("\nDo you want to introduce errors? (y/n): ").lower().strip()
        num_errors = 1
        
        if error_choice == 'y':
            try:
                num_errors = int(input("How many errors per block? (default 1): ") or "1")
            except ValueError:
                num_errors = 1
        
        corrupted_blocks = []
        for i, block in enumerate(encoded_blocks):
            corrupted = self.introduce_errors(block, num_errors)
            corrupted_blocks.append(corrupted)
        
        self.display_corrupted_blocks(corrupted_blocks, "STEP 3: Corrupted Blocks with Errors")
        
        # Step 4: Decode and correct errors
        print("\nSTEP 4: Error Detection and Correction")
        print("-" * 50)
        
        decoded_blocks = []
        total_errors = 0
        corrected_errors = 0
        
        for i, (original, encoded, corrupted) in enumerate(zip(binary_blocks, encoded_blocks, corrupted_blocks)):
            print(f"\n--- Block {i+1} ---")
            print(f"Original data:    {original}")
            print(f"Encoded:          {encoded}")
            print(f"Corrupted:        {corrupted}")
            
            # Detect differences
            errors_in_block = sum(1 for e, c in zip(encoded, corrupted) if e != c)
            total_errors += errors_in_block
            print(f"Errors introduced: {errors_in_block}")
            
            # Decode and correct
            decoded, corrected, error_pos = self.hamming.decode(corrupted)
            decoded_blocks.append(decoded)
            
            if corrected:
                corrected_errors += 1
                print(f"✓ ERROR CORRECTED at position {error_pos}")
                print(f"Corrected data:   {decoded}")
            elif error_pos == -1:
                print("✓ No errors detected")
            else:
                print("✗ Multiple errors detected (cannot correct)")
        
        # Step 5: Recover original message
        recovered_text = self.binary_to_text(decoded_blocks)
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Original message:  '{user_message}'")
        print(f"Recovered message: '{recovered_text}'")
        print(f"Success: {user_message == recovered_text}")
        print(f"\nError Correction Statistics:")
        print(f"Total errors introduced: {total_errors}")
        print(f"Errors successfully corrected: {corrected_errors}")
        print(f"Correction rate: {corrected_errors}/{total_errors}")
        
        return recovered_text, total_errors, corrected_errors
    
    def interactive_reed_solomon_demo(self):
        """Interactive Reed-Solomon code demonstration with user input"""
        print("\n" + "=" * 60)
        print("REED-SOLOMON CODE INTERACTIVE DEMONSTRATION")
        print("=" * 60)
        
        # Get user input
        print("\nEnter numerical data for Reed-Solomon encoding (4 numbers between 0-7):")
        user_input = input("Enter 4 numbers separated by spaces (e.g., '1 2 3 4'): ")
        
        if user_input.strip():
            try:
                message = [int(x) for x in user_input.split()[:4]]
                if len(message) < 4:
                    message.extend([0] * (4 - len(message)))
            except ValueError:
                message = [1, 2, 3, 4]
        else:
            message = [1, 2, 3, 4]
        
        print(f"\nOriginal message symbols: {message}")
        
        # Step 1: Encode
        encoded = self.reed_solomon.encode(message)
        print(f"\nSTEP 1: Encoded codeword: {encoded}")
        
        # Step 2: Introduce errors
        error_choice = input("\nDo you want to introduce errors? (y/n): ").lower().strip()
        num_errors = 1
        
        if error_choice == 'y':
            try:
                num_errors = int(input("How many symbol errors? (default 1): ") or "1")
            except ValueError:
                num_errors = 1
        
        corrupted = self.introduce_errors(encoded, num_errors)
        print(f"\nSTEP 2: Corrupted codeword: {corrupted}")
        
        # Step 3: Decode and correct
        decoded, corrected, error_positions = self.reed_solomon.decode(corrupted)
        
        print(f"\nSTEP 3: Error Correction Results:")
        print(f"Decoded message: {decoded}")
        print(f"Errors corrected: {corrected}")
        if corrected:
            print(f"Error positions: {error_positions}")
            print("✓ Errors successfully corrected!")
        else:
            if any(e != 0 for e in np.array(encoded) - np.array(corrupted)):
                print("✗ Errors detected but could not be corrected")
            else:
                print("✓ No errors detected")
        
        return decoded, corrected, error_positions
    
    def run_complete_demo(self):
        """Run complete interactive demonstration"""
        print("ERROR-CORRECTING CODE SYSTEM")
        print("Cipher Corps - Team ECC")
        print("\nThis system demonstrates:")
        print("1. Hamming(7,4) Code - Single-bit error correction")
        print("2. Reed-Solomon Code - Symbol error correction")
        
        while True:
            print("\n" + "=" * 60)
            print("MAIN MENU")
            print("=" * 60)
            print("1. Run Hamming Code Demo (Text Messages)")
            print("2. Run Reed-Solomon Demo (Numerical Data)")
            print("3. Run Both Demos")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                self.interactive_hamming_demo()
            elif choice == '2':
                self.interactive_reed_solomon_demo()
            elif choice == '3':
                self.interactive_hamming_demo()
                self.interactive_reed_solomon_demo()
            elif choice == '4':
                print("Thank you for using the ECC System!")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
            
            if choice in ['1', '2', '3']:
                input("\nPress Enter to continue...")


def main():
    """Main function to run the interactive ECC system"""
    system = InteractiveECCSystem()
    system.run_complete_demo()


if __name__ == "__main__":
    main()