"""
Simulation script for LLR-domain binary single-user LDPC decoder.
"""

__author__ = "Jamison Ebert"
__version__ = 1.1

import numpy as np

import ldpc

# Simulation Parameters
alist_file_name = '5GNR_1_0_8_alist.txt'
# ebno_values = np.arange(1.50, 2.25, 0.25)
ebno_values = np.array([-1, 0, 1])
desired_num_errors = 1000
maximum_num_trials = 1e6

# Prepare LDPC graph
code = ldpc.Decoder(alist_file_name)
N, K, M = code.N, code.K, code.M
z = 8   # 5G-NR lifting factor
R = K / (N - 2*z)

# Initialize error-tracking data structures
cer = np.zeros(ebno_values.shape)
num_trials = 0
num_errs = 0

# Monte-Carlo simulations to determine CER for each Eb/No value
for idxsnr in range(len(ebno_values)):
    print('Testing EbNodB = ' + str(ebno_values[idxsnr]) + 'dB')
    
    # Noise parameters
    ebno = 10**(0.1*ebno_values[idxsnr])
    nvar = 1 / (2*R*ebno)

    # Clear error parameters
    num_errs = 0
    num_trials = 0

    # run until desNumErrors have been observed or maxNumMCSims have completed
    while (num_errs < desired_num_errors) and (num_trials < maximum_num_trials):

        # Compute transmitted signal (assume all-zero codeword)
        tx_signal = -1 * (1 - 2*np.zeros(N - 2*z))

        # Send signal through AWGN channel
        rx_signal = tx_signal + np.sqrt(nvar)*np.random.randn(N - 2*z)

        # Append 2z zeros to rx signal
        rx_signal = np.hstack((np.zeros(2*z), rx_signal))

        # Reset LDPC Decoder
        code.reset()

        # Run decoder
        cdwdHt = code.soft_decision_decoding(rx_signal, nvar, max_iter=250)

        # Check for errors
        num_errs += np.sum(cdwdHt)
        num_trials += 1

    # Compute + store CER
    cer[idxsnr] = num_errs / (num_trials*N)

    print('Number of trials: ' + str(num_trials))
    print('Number of errors: ' + str(num_errs))

print('*'*25 + 'Simulations Complete!' + '*'*25)
print('SNRs Tested: ' + str(ebno_values))
print('Empirical CERs: ' + str(cer))