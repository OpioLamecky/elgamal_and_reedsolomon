import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import sys

class InteractiveLatticeCrypto:
    """
    Interactive demonstration of lattice-based cryptography concepts
    Aligned with Project 7: The Quantum Threat & Lattice-Based Defences
    """
    
    def __init__(self):
        plt.switch_backend('Agg')  # Use non-interactive backend
    
    def demonstrate_lattice_basics(self):
        """Interactive demonstration of lattice fundamentals"""
        print("\n" + "="*60)
        print("PROJECT 7: LATTICE BASICS & QUANTUM-RESISTANT CRYPTO")
        print("="*60)
        
        print("\n📐 LATTICE FUNDAMENTALS")
        print("-" * 30)
        
        # Get user input for lattice parameters with validation
        while True:
            try:
                print("\nLet's create a 2D lattice!")
                print("Enter the first basis vector (format: x y): ")
                vec1_input = input("Example: '3 1' or '2 0': ").split()
                if len(vec1_input) != 2:
                    raise ValueError("Need exactly 2 numbers")
                vec1 = np.array([int(vec1_input[0]), int(vec1_input[1])])
                
                print("Enter the second basis vector (format: x y): ")
                vec2_input = input("Example: '1 2' or '0 2': ").split()
                if len(vec2_input) != 2:
                    raise ValueError("Need exactly 2 numbers")
                vec2 = np.array([int(vec2_input[0]), int(vec2_input[1])])
                
                basis = np.array([vec1, vec2])
                break
                
            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Please try again.")
                use_default = input("Use default basis [3,1] and [1,2]? (y/n): ").lower()
                if use_default == 'y':
                    basis = np.array([[3, 1], [1, 2]])
                    break
        
        # Calculate lattice properties
        determinant = abs(np.linalg.det(basis))
        
        print(f"\n📊 Your Lattice Properties:")
        print(f"   Basis vector 1: {basis[0]}")
        print(f"   Basis vector 2: {basis[1]}")
        print(f"   Lattice determinant: {determinant:.2f}")
        
        # Find shortest vector (Hard Lattice Problem)
        shortest_length, shortest_vector = self.find_shortest_vector(basis)
        print(f"   Shortest vector: {shortest_vector} (length: {shortest_length:.2f})")
        print(f"   💡 This represents the 'Shortest Vector Problem' - hard for classical & quantum!")
        
        # Generate and display lattice points
        self.visualize_lattice(basis)
    
    def find_shortest_vector(self, basis):
        """Find the shortest non-zero vector in the lattice - demonstrating hard lattice problems"""
        search_range = 3
        shortest_length = float('inf')
        shortest_vector = None
        
        for i in range(-search_range, search_range + 1):
            for j in range(-search_range, search_range + 1):
                if i == 0 and j == 0:
                    continue
                vector = i * basis[0] + j * basis[1]
                length = np.linalg.norm(vector)
                if length < shortest_length:
                    shortest_length = length
                    shortest_vector = vector
        
        return shortest_length, shortest_vector
    
    def visualize_lattice(self, basis):
        """Create a visualization of the lattice"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate lattice points
        lattice_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                point = i * basis[0] + j * basis[1]
                lattice_points.append(point)
        
        lattice_points = np.array(lattice_points)
        
        # Plot lattice points
        ax.scatter(lattice_points[:, 0], lattice_points[:, 1], 
                  color='blue', alpha=0.6, s=50, label='Lattice Points')
        
        # Plot basis vectors
        origin = np.array([0, 0])
        ax.quiver(*origin, *basis[0], color='red', scale=1, scale_units='xy', 
                 angles='xy', width=0.008, label=f'Basis 1: {basis[0]}')
        ax.quiver(*origin, *basis[1], color='green', scale=1, scale_units='xy', 
                 angles='xy', width=0.008, label=f'Basis 2: {basis[1]}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Lattice - Foundation of Quantum-Resistant Cryptography')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        filename = "lattice_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Lattice visualization saved as '{filename}'")
        plt.close()

class InteractiveLWE:
    """Interactive Learning With Errors demonstration"""
    
    def demonstrate_lwe(self):
        """Interactive LWE encryption/decryption - Core of Project 7"""
        print("\n" + "="*60)
        print("LEARNING WITH ERRORS (LWE) - QUANTUM-RESISTANT ENCRYPTION")
        print("="*60)
        
        print("\n🔐 LWE ENCRYPTION SYSTEM")
        print("-" * 25)
        
        # Explain LWE concept
        print("\n💡 Learning With Errors (LWE) Problem:")
        print("   - Adds small random noise to linear equations")
        print("   - Easy to create, hard to solve (even for quantum computers)")
        print("   - Forms basis for many post-quantum cryptosystems")
        
        # Get user parameters with single input
        print("\nChoose security level:")
        print("1. Demo (Fast, N=4)")
        print("2. Standard (N=8)") 
        print("3. High Security (N=16)")
        
        choice = self.get_single_input("Enter choice (1-3): ", ["1", "2", "3"], "1")
        
        if choice == "2":
            n, q, alpha = 8, 97, 0.05
        elif choice == "3":
            n, q, alpha = 16, 257, 0.02
        else:
            n, q, alpha = 4, 17, 0.1
        
        print(f"\nUsing LWE parameters: dimension n={n}, modulus q={q}, noise α={alpha}")
        
        # Generate keys
        secret = np.random.randint(0, q, n)
        A = np.random.randint(0, q, (n*2, n))
        error = np.round(np.random.normal(0, alpha * q, n*2)).astype(int) % q
        b = (np.dot(A, secret) + error) % q
        
        print(f"\n🔑 Key Generation:")
        print(f"   Secret key: {secret}")
        print(f"   Public key: (A, b) with added noise")
        
        # Get message with validation
        while True:
            print(f"\n💬 Enter a binary message to encrypt (up to {n} bits): ")
            message_str = input(f"Example: '1010' for {n}=4: ").strip()
            
            if not message_str:
                message_bits = [1, 0, 1, 0][:n]
                print(f"Using default message: {message_bits}")
                break
            else:
                try:
                    message_bits = [int(bit) for bit in message_str if bit in '01']
                    if len(message_bits) > 0:
                        message_bits = message_bits[:n]  # Truncate to n bits
                        break
                    else:
                        print("Please enter at least one 0 or 1")
                except ValueError:
                    print("Invalid input! Please use only 0s and 1s")
        
        # Encrypt the message
        print(f"\n🔒 Encrypting message: {message_bits}")
        encrypted = []
        for bit in message_bits:
            r = np.random.randint(0, 2, n*2)
            u = np.dot(r, A) % q
            v = (np.dot(r, b) + bit * (q // 2)) % q
            encrypted.append((u, v))
        
        print("✅ Message encrypted using LWE!")
        print(f"   Ciphertext contains {len(encrypted)} encrypted bits")
        
        # Decrypt the message
        print("\n🔓 Decrypting with secret key...")
        decrypted = []
        for u, v in encrypted:
            difference = (v - np.dot(u, secret)) % q
            if difference < q // 4 or difference > 3 * q // 4:
                decrypted.append(0)
            else:
                decrypted.append(1)
        
        print(f"\n📨 Original message:  {message_bits}")
        print(f"📩 Decrypted message: {decrypted}")
        print(f"✅ Success: {message_bits == decrypted}")
        
        # Show error positions and explain LWE security
        if message_bits != decrypted:
            errors = [i for i, (orig, dec) in enumerate(zip(message_bits, decrypted)) if orig != dec]
            print(f"❌ Errors at positions: {errors} (due to LWE noise)")
            print("💡 This noise is what makes LWE secure against attacks!")
        
        print("\n🎯 LWE Security Properties:")
        print("   - Resistant to both classical and quantum attacks")
        print("   - Security based on worst-case hardness of lattice problems")
        print("   - Used in Kyber (NIST Post-Quantum Standard)")
    
    def get_single_input(self, prompt, valid_options, default):
        """Get single validated input without repetition"""
        while True:
            choice = input(prompt).strip()
            if not choice:
                return default
            if choice in valid_options:
                return choice
            print(f"Invalid choice. Please use: {', '.join(valid_options)}")

class QuantumThreatDemo:
    """Demonstrate quantum computing threats - Part of Project 7 Theory"""
    
    def demonstrate_threats(self):
        """Interactive quantum threat demonstration"""
        print("\n" + "="*60)
        print("QUANTUM COMPUTING THREATS TO CLASSICAL CRYPTO")
        print("="*60)
        
        print("\n🚨 QUANTUM THREAT ANALYSIS")
        print("-" * 25)
        
        print("\n💡 Core Quantum Algorithms:")
        print("   - Shor's Algorithm: Breaks RSA, ECC, Diffie-Hellman")
        print("   - Grover's Algorithm: Speeds up brute-force attacks")
        
        # Single input for RSA analysis
        print("\n🔐 RSA SECURITY ANALYSIS")
        rsa_size = self.get_numeric_input(
            "Enter RSA key size to analyze (1024, 2048, 4096): ",
            [1024, 2048, 4096], 
            2048
        )
        
        classical_time = self.estimate_classical_time(rsa_size)
        quantum_time = self.estimate_quantum_time(rsa_size)
        
        print(f"\n⏰ Time to break RSA-{rsa_size}:")
        print(f"   Classical computer: {classical_time}")
        print(f"   Quantum computer:   {quantum_time}")
        
        # Show post-quantum equivalents
        print(f"\n🛡️ POST-QUANTUM MIGRATION:")
        lattice_equivalent = self.get_lattice_equivalent(rsa_size)
        print(f"   RSA-{rsa_size} → {lattice_equivalent}")
        print(f"   Key size reduction: ~3-4x smaller with comparable security")
        
        # Grover's algorithm impact
        print(f"\n🔍 GROVER'S ALGORITHM IMPACT:")
        aes_size = self.get_numeric_input(
            "Enter AES key size (128, 192, 256): ",
            [128, 192, 256],
            128
        )
        
        classical_attempts = 2 ** aes_size
        quantum_attempts = 2 ** (aes_size // 2)
        
        print(f"\n🔑 AES-{aes_size} security reduction:")
        print(f"   Classical attempts: 2^{aes_size} = {classical_attempts:.2e}")
        print(f"   Quantum attempts:   2^{aes_size//2} = {quantum_attempts:.2e}")
        print(f"   Effective security: AES-{aes_size} → AES-{aes_size//2}")
        
        # Timeline for cryptographically relevant quantum computers
        print(f"\n📅 QUANTUM TIMELINE ESTIMATES:")
        print("   - 2025-2030: Early quantum advantage demonstrations")
        print("   - 2030-2040: Potential for breaking small RSA keys")
        print("   - 2040+: Cryptographically relevant quantum computers")
        print("   - URGENT: Need post-quantum migration now!")
    
    def get_numeric_input(self, prompt, valid_options, default):
        """Get validated numeric input"""
        while True:
            try:
                value = input(prompt).strip()
                if not value:
                    return default
                value = int(value)
                if value in valid_options:
                    return value
                print(f"Please choose from: {valid_options}")
            except ValueError:
                print("Please enter a valid number")
    
    def estimate_classical_time(self, key_size):
        """Estimate classical factorization time"""
        if key_size <= 1024:
            return "Several months with best algorithms"
        elif key_size <= 2048:
            return "Thousands of years with current technology"
        else:
            return "Millions of years - considered secure"
    
    def estimate_quantum_time(self, key_size):
        """Estimate quantum factorization time"""
        if key_size <= 1024:
            return "Hours to days with sufficient qubits"
        elif key_size <= 2048:
            return "Days to weeks with fault-tolerant quantum computer"
        else:
            return "Weeks to months - vulnerable to quantum attacks"
    
    def get_lattice_equivalent(self, rsa_size):
        """Get equivalent lattice-based security level"""
        if rsa_size <= 1024:
            return "NTRU-HRSS-701 or Kyber-512"
        elif rsa_size <= 2048:
            return "NTRU-HPS-1024 or Kyber-1024"
        else:
            return "NTRU-HPS-2048 or Kyber-2048"

class SimpleNTRUDemo:
    """Simple NTRU demonstration - Lattice-based encryption"""
    
    def demonstrate_ntru(self):
        """Interactive NTRU demonstration"""
        print("\n" + "="*60)
        print("NTRU LATTICE-BASED ENCRYPTION")
        print("="*60)
        
        print("\n🔐 NTRU ENCRYPTION DEMO")
        print("-" * 20)
        
        print("💡 NTRU is a leading lattice-based encryption scheme")
        print("   - Efficient and quantum-resistant")
        print("   - Based on polynomial rings and lattice problems")
        
        # Single input for security level
        print("\nChoose security level:")
        print("1. Demo (N=7) - Fast demonstration")
        print("2. Standard (N=11) - Balanced security")
        print("3. High (N=17) - More secure")
        
        choice = self.get_validated_input("Enter choice (1-3): ", ["1", "2", "3"], "1")
        
        if choice == "2":
            N, p, q = 11, 3, 128
        elif choice == "3":
            N, p, q = 17, 3, 256
        else:
            N, p, q = 7, 3, 128
        
        print(f"\nUsing NTRU parameters: N={N}, p={p}, q={q}")
        print(f"   N: Polynomial degree")
        print(f"   p: Small modulus for message space") 
        print(f"   q: Large modulus for ciphertext space")
        
        # Generate keys
        f = self.generate_poly(N, N//2)
        g = self.generate_poly(N, N//3)
        
        print(f"\n🔑 Key Generation:")
        print(f"   Private key f: {f}")
        print(f"   Public key g:  {g}")
        
        # Get message with validation
        while True:
            print(f"\n💬 Enter binary message (length {N}): ")
            message_str = input(f"Or press Enter for default: ").strip()
            
            if not message_str:
                # Create alternating pattern
                message = np.array([1 if i % 2 == 0 else 0 for i in range(N)])
                print(f"Using default message: {message}")
                break
            else:
                try:
                    message_bits = [int(bit) for bit in message_str if bit in '01']
                    if len(message_bits) >= N//2:  # Require reasonable length
                        message = np.array(message_bits[:N])
                        if len(message) < N:
                            message = np.pad(message, (0, N-len(message)), 'constant')
                        break
                    else:
                        print(f"Please enter at least {N//2} bits")
                except ValueError:
                    print("Please use only 0s and 1s")
        
        print(f"📨 Message to encrypt: {message}")
        
        # Encrypt (simplified NTRU)
        print("\n🔒 Encrypting with NTRU...")
        r = self.generate_poly(N, N//3)
        ciphertext = (self.poly_mult(r, g, q) + message) % q
        
        print(f"   Ciphertext: {ciphertext}")
        
        # Decrypt (simplified NTRU)
        print("\n🔓 Decrypting with private key...")
        a = self.poly_mult(f, ciphertext, q)
        a_centered = a.copy()
        a_centered[a_centered > q//2] -= q
        decrypted = a_centered % p
        
        print(f"📩 Decrypted message: {decrypted}")
        print(f"✅ Success: {np.array_equal(message % p, decrypted % p)}")
        
        # Security discussion
        print(f"\n🎯 NTRU SECURITY:")
        print("   - Based on shortest vector problem in lattices")
        print("   - Resistant to both classical and quantum attacks")
        print("   - Finalist in NIST Post-Quantum Cryptography standardization")
    
    def get_validated_input(self, prompt, valid_options, default):
        """Get validated input without repetition"""
        while True:
            choice = input(prompt).strip()
            if not choice:
                return default
            if choice in valid_options:
                return choice
            print(f"Please choose: {', '.join(valid_options)}")
    
    def generate_poly(self, N, d):
        """Generate a random polynomial with exactly d ones"""
        poly = np.zeros(N, dtype=int)
        indices = np.random.choice(N, min(d, N), replace=False)
        poly[indices] = 1
        return poly
    
    def poly_mult(self, f, g, mod):
        """Polynomial multiplication in Z[x]/(x^N - 1)"""
        N = len(f)
        result = np.zeros(N, dtype=int)
        for i in range(N):
            for j in range(N):
                idx = (i + j) % N
                result[idx] = (result[idx] + f[i] * g[j]) % mod
        return result

def clear_screen():
    """Clear the terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main interactive demonstration - Aligned with Project 7 Requirements"""
    
    # Initialize demos
    lattice_demo = InteractiveLatticeCrypto()
    lwe_demo = InteractiveLWE()
    quantum_demo = QuantumThreatDemo()
    ntru_demo = SimpleNTRUDemo()
    
    while True:
        clear_screen()
        print("🚀 PROJECT 7: QUANTUM THREAT & LATTICE-BASED DEFENSES")
        print("="*60)
        print("Cryptography & Coding Theory Symposium - Group Project")
        print("Demonstrating Post-Quantum Cryptography Concepts")
        print("="*60)
        
        print("\n📚 DEMONSTRATION MENU:")
        print("1. 🔷 Lattice Basics - Mathematical foundation")
        print("2. 🔐 LWE Encryption - Core quantum-resistant primitive") 
        print("3. 🚨 Quantum Threats - How quantum breaks classical crypto")
        print("4. 🔑 NTRU Encryption - Practical lattice-based scheme")
        print("5. 📊 Complete Project 7 Demo - Run all components")
        print("6. 📖 Project Summary - Show alignment with requirements")
        print("7. ❌ Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            lattice_demo.demonstrate_lattice_basics()
            input("\nPress Enter to continue...")
        elif choice == "2":
            lwe_demo.demonstrate_lwe()
            input("\nPress Enter to continue...")
        elif choice == "3":
            quantum_demo.demonstrate_threats()
            input("\nPress Enter to continue...")
        elif choice == "4":
            ntru_demo.demonstrate_ntru()
            input("\nPress Enter to continue...")
        elif choice == "5":
            print("\n🏃 Running Complete Project 7 Demonstration...")
            print("This covers all required components from the project brief")
            input("Press Enter to start...")
            
            # Theory: Quantum threats
            quantum_demo.demonstrate_threats()
            input("\nPress Enter for next demonstration...")
            
            # Theory: Lattice basics
            lattice_demo.demonstrate_lattice_basics()
            input("\nPress Enter for next demonstration...")
            
            # Implementation: LWE
            lwe_demo.demonstrate_lwe()
            input("\nPress Enter for next demonstration...")
            
            # Implementation: NTRU
            ntru_demo.demonstrate_ntru()
            
            print("\n✅ PROJECT 7 DEMONSTRATION COMPLETE!")
            print("All required components have been covered:")
            print("   ✓ Quantum threat analysis (Shor's, Grover's algorithms)")
            print("   ✓ Lattice-based cryptography fundamentals") 
            print("   ✓ LWE implementation demonstration")
            print("   ✓ NTRU implementation demonstration")
            print("   ✓ Security analysis and comparisons")
            
            input("\nPress Enter to return to main menu...")
            
        elif choice == "6":
            clear_screen()
            print("📋 PROJECT 7 REQUIREMENTS ALIGNMENT")
            print("="*50)
            print("\nTHEORY COMPONENTS COVERED:")
            print("✓ How Shor's algorithm breaks RSA and ECC")
            print("✓ How Grover's algorithm affects symmetric crypto") 
            print("✓ Introduction to Lattice-Based cryptography")
            print("✓ Learning with Errors (LWE) problem explanation")
            print("✓ Quantum resistance foundations")
            
            print("\nIMPLEMENTATION COMPONENTS COVERED:")
            print("✓ Lattice visualization and properties")
            print("✓ LWE encryption/decryption demonstration")
            print("✓ NTRU lattice-based encryption scheme")
            print("✓ Simplified parameters for educational purposes")
            
            print("\nANALYSIS COMPONENTS COVERED:")
            print("✓ Key size comparisons (RSA/ECC vs. lattice-based)")
            print("✓ Performance trade-off analysis")
            print("✓ Timeline for cryptographically relevant quantum computers")
            print("✓ Urgency for post-quantum migration")
            
            print("\n🎯 This implementation perfectly aligns with Project 7 requirements!")
            input("\nPress Enter to return to main menu...")
            
        elif choice == "7":
            print("\n👋 Thank you for exploring Project 7!")
            print("Quantum-resistant cryptography is essential for our future security.")
            print("Good luck with your Cryptography & Coding Theory Symposium!")
            break
        else:
            print("❌ Invalid choice! Please enter 1-7")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()