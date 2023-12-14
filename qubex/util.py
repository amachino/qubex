"""Utility functions for qubex."""


def find_nearest_frequency_combinations(
    target_frequency,
    lo_range=(8000, 12000),
    nco_range=(0, 3000),
    lo_step: int = 500,
    nco_step: int = 375,
) -> tuple[int, list[tuple[int, int]]]:
    """
    Find the nearest LO and NCO frequencies to a target frequency.

    Parameters
    ----------
    target_frequency : int
        The target frequency.
    lo_range : tuple[int, int], optional
        The range of LO frequencies to search.
    nco_range : tuple[int, int], optional
        The range of NCO frequencies to search.
    lo_step : int, optional
        The step size for LO frequencies.
    nco_step : int, optional
        The step size for NCO frequencies.
    """
    # Adjust the start of the range to the nearest multiple of the step using integer division
    lo_start = ((lo_range[0] + lo_step - 1) // lo_step) * lo_step
    nco_start = ((nco_range[0] + nco_step - 1) // nco_step) * nco_step

    # Generate the possible LO and NCO frequencies based on the adjusted ranges and steps
    lo_frequencies = [freq for freq in range(lo_start, lo_range[1] + lo_step, lo_step)]
    nco_frequencies = [
        freq for freq in range(nco_start, nco_range[1] + nco_step, nco_step)
    ]

    # Initialize variables to store the best combinations and minimum difference
    best_combinations = []
    best_frequency = 0
    min_difference = float("inf")

    # Loop through each LO frequency and find the NCO frequency that makes LO - NCO closest to the target
    for lo in lo_frequencies:
        for nco in nco_frequencies:
            lo_minus_nco = lo - nco
            difference = abs(target_frequency - lo_minus_nco)

            if difference < min_difference:
                # Clear the best_combinations list, update the best_frequency, and update the minimum difference
                best_combinations = [(lo, nco)]
                best_frequency = lo_minus_nco
                min_difference = difference
            elif difference == min_difference:
                # Add the new combination to the list
                best_combinations.append((lo, nco))

    print(f"Target frequency: {target_frequency}")
    print(f"Nearest frequency: {best_frequency}")
    print(f"Combinations: {best_combinations}")

    return best_frequency, best_combinations
