    signal_power = np.mean(calibration_matrix @ y_train[42] ** 2)
    noise_level = signal_power / (10 ** (snr_db / 10))
    print(f"Noise std at 30dB: {np.sqrt(noise_level):.4f}")
    print(f"Smallest kept singular value: {S[11]:.4f}")
    print(f"Amplific