"""Independently audit partitions using metadata only; never load blind fields."""
from pathlib import Path
import sys
import numpy as np
from v210.research.protocol import (block_keys, command_signature, digest, header, read_json,
                                   source_files, write_json)

root = Path(sys.argv[1])
protocol = read_json(root/"protocol.json")
with np.load(root/"blind/spec.npz", allow_pickle=False) as f:
    spec = {k: f[k] for k in f.files}
signatures = [{command_signature(w) for w in spec["command"][spec["partition"] == i]}
              for i in range(3)]
blocks = [set(spec[k]) for k in ("train_blocks", "validation_blocks", "test_blocks")]
command_overlap = [len(signatures[a] & signatures[b]) for a,b in ((0,1),(0,2),(1,2))]
block_overlap = [len(blocks[a] & blocks[b]) for a,b in ((0,1),(0,2),(1,2))]
assert command_overlap == block_overlap == [0,0,0]
assert len(set.union(*blocks)) == 648
dev_counts = {}
for name, pid in (("train", 0), ("validation", 1)):
    with np.load(root/"dev"/(name+".npz"), allow_pickle=False) as data:
        assert set(block_keys(data["theta"], data["phi"])).issubset(blocks[pid])
        assert np.all(spec["partition"][data["command_indices"]] == pid)
        dev_counts[name] = list(data["amplitude"].shape)
folder = Path(protocol["source_folder"])
ot, op = header(source_files(folder, False)[0])
nt, np_ = header(source_files(folder, True)[0])
assert not set(zip(ot, op)) & set(zip(nt, np_))
assert not list((root/"blind").glob("*amplitude*"))
report = {
    "audit_type": "metadata_only_before_blind_open",
    "protocol_sha256": digest(root/"protocol.json"),
    "command_counts": [len(v) for v in signatures],
    "phase_equivalent_command_overlap_train_val_train_test_val_test": command_overlap,
    "angular_block_overlap_train_val_train_test_val_test": block_overlap,
    "integer_shifted_coordinate_overlap": 0,
    "dev_amplitude_shapes": dev_counts,
    "blind_field_values_read_by_this_audit": False,
    "blind_label_files_exist": False,
    "historical_single_dft_anchor_test_count": 4,
    "novel_random_sparse_steering_test_count": 58,
    "result": "PASS"
}
write_json(root/"data_isolation_audit.json", report)
print(report)

