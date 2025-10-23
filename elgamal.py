# Import required libraries
import random  # For generating random numbers

class ElGamal:
    def __init__(self, bit_length=256):  # Increased from 128 to 256
        """Initialize ElGamal cryptographic system with generated parameters"""
        # Set the bit length for prime number generation (increased for larger messages)
        self.bit_length = bit_length
        
        # Generate cryptographic parameters: prime p and generator g
        self.p, self.g = self.generate_params()
        
        # Generate private and public key pair
        self.private_key, self.public_key = self.generate_keys()
        
        # Store the last used nonce - THIS CREATES THE VULNERABILITY
        self.last_nonce = None
    
    def is_prime(self, n, k=10):
        """Miller-Rabin primality test to check if a number is prime"""
        # Handle small prime cases
        if n == 2 or n == 3:
            return True
        # Exclude numbers <= 1 and even numbers
        if n <= 1 or n % 2 == 0:
            return False
        
        # Factor n-1 as 2^r * d where d is odd
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1      # Increment exponent r
            d //= 2     # Divide d by 2 until it becomes odd
        
        # Test the number k times with different random bases
        for _ in range(k):
            # Choose a random base between 2 and n-2
            a = random.randint(2, n - 2)
            # Compute x = a^d mod n
            x = pow(a, d, n)
            
            # If x is 1 or n-1, this test passes
            if x == 1 or x == n - 1:
                continue
            
            # Repeat the squaring test r-1 times
            for _ in range(r - 1):
                # Square x modulo n
                x = pow(x, 2, n)
                # If we get n-1, this test passes
                if x == n - 1:
                    break
            else:
                # If we never got n-1, the number is composite
                return False
        # All tests passed - number is probably prime
        return True
    
    def generate_params(self):
        """Generate a large prime p and a generator g for the cryptographic system"""
        while True:
            # Generate a random number with specified bit length
            p = random.getrandbits(self.bit_length)
            # Ensure the number is large by setting the highest bit and making it odd
            p |= (1 << (self.bit_length - 1)) | 1
            
            # Check if the number is prime using our primality test
            if self.is_prime(p):
                break  # Exit loop when we find a prime
        
        # For demonstration, use 2 as generator (simplified approach)
        g = 2
        return p, g
    
    def generate_keys(self):
        """Generate private and public keys for ElGamal encryption"""
        # Generate a random private key between 2 and p-2
        private_key = random.randint(2, self.p - 2)
        # Compute public key: g^private_key mod p
        public_key = pow(self.g, private_key, self.p)
        return private_key, public_key
    
    def text_to_number(self, text):
        """Convert text string to numerical representation for encryption"""
        # Encode text as UTF-8 bytes, then convert to a large integer
        return int.from_bytes(text.encode('utf-8'), 'big')
    
    def number_to_text(self, number):
        """Convert numerical representation back to text string after decryption"""
        try:
            # Calculate how many bytes are needed to represent the number
            if number == 0:
                return ""
            byte_length = (number.bit_length() + 7) // 8
            # Convert number to bytes and then decode back to text
            bytes_data = number.to_bytes(byte_length, 'big')
            return bytes_data.decode('utf-8')
        except:
            # If decoding fails, return empty string
            return f"[Number: {number}]"
    
    def encrypt(self, plaintext, reuse_nonce=False):
        """Encrypt plaintext using ElGamal encryption"""
        # Convert the text message to a numerical representation
        m = self.text_to_number(plaintext)
        
        # Check if the message is too large for our prime
        if m >= self.p:
            # Use a shorter message or split into chunks (for demo, we'll use modulo)
            m = m % self.p
            print(f"Warning: Message was truncated to fit prime size")
        
        # CHOOSE NONCE: Either reuse old one (vulnerability) or generate new
        if reuse_nonce and self.last_nonce:
            k = self.last_nonce  # REUSING NONCE - THIS IS THE VULNERABILITY
        else:
            k = random.randint(2, self.p - 2)  # Generate new random nonce
            self.last_nonce = k  # Store for potential reuse
        
        # Compute first part of ciphertext: c1 = g^k mod p
        c1 = pow(self.g, k, self.p)
        # Compute second part of ciphertext: c2 = m * (public_key^k) mod p
        c2 = (m * pow(self.public_key, k, self.p)) % self.p
        
        # Return the complete ciphertext pair
        return (c1, c2)
    
    def decrypt(self, ciphertext):
        """Decrypt ciphertext using ElGamal decryption"""
        # Unpack the ciphertext into its two components
        c1, c2 = ciphertext
        
        # Compute the shared secret: s = c1^private_key mod p
        s = pow(c1, self.private_key, self.p)
        # Compute the modular inverse of s using Fermat's Little Theorem
        s_inv = pow(s, self.p - 2, self.p)
        # Recover the message: m = c2 * s_inv mod p
        m = (c2 * s_inv) % self.p
        
        # Convert the numerical message back to text
        return self.number_to_text(m)
    
    def attack_reused_nonce(self, ciphertext1, ciphertext2):
        """Exploit the reused nonce vulnerability to find relationship between messages"""
        # Unpack both ciphertexts
        c1_1, c2_1 = ciphertext1
        c1_2, c2_2 = ciphertext2
        
        # Check if nonce was actually reused (c1 values should be identical)
        if c1_1 != c1_2:
            return "No reused nonce detected - different c1 values"
        
        # Compute the ratio: (c2_1 / c2_2) mod p = (m1 / m2) mod p
        # This works because: c2_1 = m1 * h^k, c2_2 = m2 * h^k
        # So: c2_1 / c2_2 = (m1 * h^k) / (m2 * h^k) = m1 / m2
        ratio = (c2_1 * pow(c2_2, self.p - 2, self.p)) % self.p
        
        return f"Ratio m1/m2 mod p = {ratio}"

def main():
    """Main function to run the ElGamal encryption demo"""
    print("ElGamal Encryption System with Reused Nonce Vulnerability")
    print("=" * 60)
    
    # Initialize ElGamal cryptographic system with larger prime for demonstration
    elgamal = ElGamal(bit_length=256)  # Increased to 256 bits
    
    # Display generated cryptographic parameters
    print("Generated Parameters:")
    print(f"Prime p: {elgamal.p}")
    print(f"Generator g: {elgamal.g}")
    print(f"Public Key: {elgamal.public_key}")
    print(f"Private Key: {elgamal.private_key}")
    print(f"Prime bit length: {elgamal.p.bit_length()} bits")
    print()
    
    # Main program loop
    while True:
        print("\nOptions:")
        print("1. Encrypt message")
        print("2. Decrypt message")
        print("3. Encrypt with reused nonce (vulnerability)")
        print("4. Demonstrate reused nonce attack")
        print("5. Use custom public key")
        print("6. Exit")
        
        # Get user choice
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            # Normal encryption without nonce reuse
            plaintext = input("Enter plaintext to encrypt: ").strip()
            try:
                # Encrypt with new random nonce each time
                ciphertext = elgamal.encrypt(plaintext, reuse_nonce=False)
                print(f"Ciphertext (c1, c2): {ciphertext}")
            except Exception as e:
                print(f"Encryption error: {e}")
        
        elif choice == '2':
            # Decryption of ciphertext
            try:
                c1 = int(input("Enter c1: ").strip())
                c2 = int(input("Enter c2: ").strip())
                # Decrypt the ciphertext
                plaintext = elgamal.decrypt((c1, c2))
                print(f"Decrypted: {plaintext}")
            except Exception as e:
                print(f"Decryption error: {e}")
        
        elif choice == '3':
            # Demonstrate the vulnerability by reusing nonce
            plaintext1 = input("Enter first plaintext: ").strip()
            plaintext2 = input("Enter second plaintext: ").strip()
            
            # Encrypt first message with new nonce
            ciphertext1 = elgamal.encrypt(plaintext1, reuse_nonce=False)
            # Encrypt second message REUSING the same nonce - VULNERABILITY!
            ciphertext2 = elgamal.encrypt(plaintext2, reuse_nonce=True)
            
            print(f"First ciphertext (c1, c2): {ciphertext1}")
            print(f"Second ciphertext (c1, c2): {ciphertext2}")
            print("Nonce reused! Vulnerability created.")
            print("Notice that c1 values are identical - this reveals nonce reuse!")
        
        elif choice == '4':
            # Automatically demonstrate the attack with SHORTER messages
            print("Demonstrating reused nonce attack...")
            
            # Use shorter test messages to ensure they fit
            msg1 = "Secret1"
            msg2 = "Secret2"
            
            print(f"Using shorter messages to ensure they fit in prime size:")
            print(f"Message 1: {msg1}")
            print(f"Message 2: {msg2}")
            
            # Encrypt with reused nonce
            cipher1 = elgamal.encrypt(msg1, reuse_nonce=False)
            cipher2 = elgamal.encrypt(msg2, reuse_nonce=True)  # Reuses nonce!
            
            print(f"Ciphertext 1: {cipher1}")
            print(f"Ciphertext 2: {cipher2}")
            
            # Perform the attack to detect nonce reuse
            result = elgamal.attack_reused_nonce(cipher1, cipher2)
            print(f"Attack result: {result}")
            
            # If we know one message, we can recover the other
            known_msg = msg2  # Assume we know the second message
            known_num = elgamal.text_to_number(known_msg)  # Convert to number
            
            # Calculate the ratio between the two ciphertexts
            ratio = (cipher1[1] * pow(cipher2[1], elgamal.p - 2, elgamal.p)) % elgamal.p
            # Recover the first message: m1 = ratio * m2
            recovered_num = (ratio * known_num) % elgamal.p
            recovered_msg = elgamal.number_to_text(recovered_num)
            
            print(f"Recovered message 1 using known message 2: {recovered_msg}")
            print(f"Original message 1 was: {msg1}")
            print(f"Attack successful: {recovered_msg == msg1}")
        
        elif choice == '5':
            # Allow user to input custom cryptographic parameters
            try:
                p = int(input("Enter prime p: ").strip())
                g = int(input("Enter generator g: ").strip())
                public_key = int(input("Enter public key: ").strip())
                
                # Update the ElGamal instance with custom parameters
                elgamal.p = p
                elgamal.g = g
                elgamal.public_key = public_key
                print("Custom parameters set successfully!")
                print(f"New prime bit length: {elgamal.p.bit_length()} bits")
            except Exception as e:
                print(f"Error setting parameters: {e}")
        
        elif choice == '6':
            # Exit the program
            print("Goodbye!")
            break
        
        else:
            # Handle invalid input
            print("Invalid choice. Please try again.")

# Entry point of the program
if __name__ == "__main__":
    main()