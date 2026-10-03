import glob, os
import numpy as np
import h5py

subjects = [1, 2, 3, 4, 9]
rows = {}      # muscle -> {subject: [values]}
counts = {}
skipped = 0

for s in subjects:
    path = f"Sub_{s}/Sub{s}_segEnergetics.mat"
    if not os.path.exists(path):
        print("missing", path)
        continue
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            global skipped
            if "/EMG/SNR/" in name and isinstance(obj, h5py.Dataset):
                if obj.dtype == object:
                    skipped += 1
                    return
                v = np.array(obj).astype(float).ravel()
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return
                muscle = name.split("/")[-1]
                rows.setdefault(muscle, {}).setdefault(s, []).append(float(np.mean(v)))
                counts[s] = counts.get(s, 0) + 1
        f.visititems(visit)

print("SNR entries found per subject:", counts, "| skipped (references):", skipped)
print()
print("Median SNR per muscle (all tests and sessions)")
print("%-18s" % "muscle" + "".join("%9s" % f"Sub{s}" for s in subjects))
for m in sorted(rows):
    line = "%-18s" % m
    for s in subjects:
        vals = rows[m].get(s)
        line += "%9.1f" % np.median(vals) if vals else "%9s" % "-"
    print(line)

print()
print("Overall median per subject:")
for s in subjects:
    allv = [x for m in rows for x in rows[m].get(s, [])]
    if allv:
        print(f"Sub{s}: median {np.median(allv):.1f}, min {np.min(allv):.1f}, n={len(allv)}")