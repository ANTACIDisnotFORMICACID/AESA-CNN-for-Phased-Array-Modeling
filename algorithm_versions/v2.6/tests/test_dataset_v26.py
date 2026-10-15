import numpy as np

from dataset_v26 import SPLIT_SPEC, generate_state_profiles, target_angles_for_state, target_block_owner


def test_release_profile_counts_ranges_and_determinism():
    expected = {"train": 96, "validation": 24, "test": 48}
    for split, count in expected.items():
        a = generate_state_profiles(split)
        b = generate_state_profiles(split)
        assert len(a) == count
        assert a == b
        assert all(25.0 <= item.measurement_snr_db <= 40.0 for item in a)
        assert all(0.0 <= item.amplitude_rms_fraction <= 0.13 for item in a)
        assert all(0.0 <= item.phase_rms_deg <= 22.0 for item in a)


def test_target_blocks_do_not_overlap_and_each_state_has_64_targets():
    owners = {(ti, pi): target_block_owner(ti, pi) for ti in range(8) for pi in range(8)}
    assert set(owners.values()) == {"train", "validation", "test"}
    for split in SPLIT_SPEC:
        profile = generate_state_profiles(split)[0]
        targets = target_angles_for_state(profile)
        assert targets.shape == (64, 2)
        assert np.all((targets[:, 0] >= 60.0) & (targets[:, 0] <= 120.0))
        assert np.all((targets[:, 1] >= -30.0) & (targets[:, 1] <= 30.0))
        theta_block = np.minimum(((targets[:, 0] - 60.0) // 7.5).astype(int), 7)
        phi_block = np.minimum(((targets[:, 1] + 30.0) // 7.5).astype(int), 7)
        assert all(target_block_owner(int(ti), int(pi)) == split for ti, pi in zip(theta_block, phi_block))
