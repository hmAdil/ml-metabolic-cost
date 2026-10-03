import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

SUBJECTS = [1, 2, 3, 4, 9]     # subjects to process (missing folders are skipped)
OUT_DIR = "features"           # CSVs are saved here


def last20(n):
    """Slice covering the last 20% of cycles/breaths (~last minute of a test)."""
    return slice(int(round(0.8 * n)), n)


def build_row(test, energetics, mechanics):
    row = {}

    # EMG: stored as (1001, cycles) -> mean over last 20% of cycles
    emg = energetics["segEnergetics"][test]["EMG"]["Activity"]
    for muscle in emg.keys():
        a = np.array(emg[muscle])
        row["emg_" + muscle] = a[:, last20(a.shape[1])].mean()

    # Peak vertical ground reaction force (left/right)
    for side in ("Left", "Right"):
        g = mechanics[test]["GRF"]["V_" + side]          # (cycles, 101)
        row["grf_peak_" + side] = g[last20(g.shape[0])].mean(axis=0).max()

    # Joint range of motion from the average gait cycle
    for joint in ("Hip", "Knee", "Ankle"):
        for side in ("Left", "Right"):
            a = mechanics[test]["Angles"][f"{joint}_{side}"]
            cycle = a[last20(a.shape[0])].mean(axis=0)
            row[f"rom_{joint}_{side}"] = cycle.max() - cycle.min()

    # Target: net metabolic cost (W/kg), mean of last 20% of breaths
    mc = np.array(energetics["segEnergetics"][test]["Metabolic"]["MetCost"]).ravel()
    row["target"] = mc[last20(len(mc))].mean()
    return row


def process_subject(s):
    folder = f"Sub_{s}"
    e_path = os.path.join(folder, f"Sub{s}_segEnergetics.mat")
    m_path = os.path.join(folder, f"Sub{s}_segMechanics.mat")
    if not (os.path.exists(e_path) and os.path.exists(m_path)):
        print(f"Sub{s}: files not found in {folder}/, skipping")
        return None

    try:
        mechanics = sio.loadmat(m_path, simplify_cells=True)["segMechanics"]
    except NotImplementedError:
        print(f"Sub{s}: mechanics file is HDF5 (v7.3), needs h5py loader, skipping")
        return None

    rows = []
    with h5py.File(e_path, "r") as energetics:
        for i in range(1, 31):
            test = f"T{i}"
            try:
                r = build_row(test, energetics, mechanics)
                r["subject"] = s
                r["test"] = test
                rows.append(r)
            except Exception as err:
                print(f"Sub{s} {test} skipped: {err}")

    df = pd.DataFrame(rows)
    df = df[["subject", "test"] + [c for c in df.columns if c not in ("subject", "test")]]
    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    all_dfs = []

    for s in SUBJECTS:
        df = process_subject(s)
        if df is None:
            continue
        df.to_csv(os.path.join(OUT_DIR, f"sub{s}_features.csv"), index=False)
        all_dfs.append(df)
        print(f"Sub{s}: {df.shape[0]} rows, {int(df['target'].isna().sum())} NaN targets")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(OUT_DIR, "all_subjects.csv"), index=False)
        print("\nCombined shape:", combined.shape)
        print(combined["target"].describe())