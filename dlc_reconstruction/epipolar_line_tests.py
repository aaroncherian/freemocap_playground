# epipolar_viewer.py
# Visualize epipolar lines on a chosen camera using DLC CSVs + calibration TOML.
#
# Example:
#   python epipolar_viewer.py \
#       --calib sesh_2023-06-07_11_10_50_treadmill_calibration_01_camera_calibration.toml \
#       --csv_dir ./csvs \
#       --target Cam6 \
#       --joint right_foot_index \
#       --force_size 1280x720

from __future__ import annotations
import argparse
from pathlib import Path
import re
import tomllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------- IO Helpers ----------------
def load_calibration(calib_path: Path):
    with open(calib_path, "rb") as f:
        calib = tomllib.load(f)
    cams = {}
    for key, entry in calib.items():
        if key.startswith("cam_"):
            K = np.array(entry["matrix"], dtype=float)
            R_c2w = np.array(entry["world_orientation"], dtype=float)
            C_w   = np.array(entry["world_position"], dtype=float).reshape(3)
            R = R_c2w.T
            t = - R @ C_w
            # TOML stored as [H, W]
            H, W = entry["size"]
            name = entry.get("name", "")
            cams[key] = dict(K=K, R=R, t=t, size=(H, W), name=name)
    if not cams:
        raise ValueError("No [cam_*] sections found in calibration TOML")
    return cams

def cam_label_from_name(name: str):
    m = re.search(r"(Cam\d+)", name, re.IGNORECASE)
    return m.group(1).capitalize() if m else None

def load_dlc_multidx(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    header_bp = df.iloc[1].tolist()
    header_coord = df.iloc[2].tolist()
    cols = pd.MultiIndex.from_arrays([header_bp, header_coord])
    dfv = df.iloc[3:].copy()
    dfv.columns = cols
    dfv.reset_index(drop=True, inplace=True)
    return dfv

def extract_joint(dfv: pd.DataFrame, joint: str) -> np.ndarray:
    x = pd.to_numeric(dfv[(joint,"x")], errors="coerce").to_numpy()
    y = pd.to_numeric(dfv[(joint,"y")], errors="coerce").to_numpy()
    p = pd.to_numeric(dfv[(joint,"likelihood")], errors="coerce").to_numpy() if (joint,"likelihood") in dfv.columns else np.ones_like(x)
    return np.stack([x,y,p], axis=1)

def extract_all_points_xy(dfv: pd.DataFrame) -> dict[str,np.ndarray]:
    joints = sorted(set(dfv.columns.get_level_values(0)) - {"coords","scorer","bodyparts"})
    out = {}
    for j in joints:
        if (j,"x") in dfv.columns and (j,"y") in dfv.columns:
            x = pd.to_numeric(dfv[(j,"x")], errors="coerce").to_numpy()
            y = pd.to_numeric(dfv[(j,"y")], errors="coerce").to_numpy()
            out[j] = np.stack([x,y], axis=1)
    return out

# ---------------- Geometry ----------------
def fundamental_from_extrinsics(Ki,Kj,Ri,ti,Rj,tj):
    R_ji = Rj @ Ri.T
    t_ji = tj - R_ji @ ti
    tx = np.array([[0, -t_ji[2], t_ji[1]],
                   [t_ji[2], 0, -t_ji[0]],
                   [-t_ji[1], t_ji[0], 0]])
    E = tx @ R_ji
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    return F

def epipolar_line(F,x):
    xh = np.array([x[0],x[1],1.0])
    l = F @ xh
    return l/(np.linalg.norm(l[:2])+1e-12)

def clip_line_to_image(l,W,H):
    a,b,c=l; pts=[]
    if abs(b)>1e-12:
        for u in [0.0, float(W)]:
            v=-(a*u+c)/b; pts.append((u,v))
    if abs(a)>1e-12:
        for v in [0.0, float(H)]:
            u=-(b*v+c)/a; pts.append((u,v))
    inside=[(u,v) for (u,v) in pts if 0<=u<=W and 0<=v<=H]
    uniq=[]
    for p in inside:
        if all(np.hypot(p[0]-q[0],p[1]-q[1])>1e-6 for q in uniq):
            uniq.append(p)
    return uniq[:2] if len(uniq)>=2 else None

# ---------------- Diagnostics ----------------
def estimate_csv_frame_size(dfv: pd.DataFrame) -> tuple[float, float]:
    xs, ys = [], []
    for j in sorted(set(dfv.columns.get_level_values(0)) - {"coords","scorer","bodyparts"}):
        if (j,"x") in dfv.columns and (j,"y") in dfv.columns:
            x = pd.to_numeric(dfv[(j,"x")], errors="coerce").to_numpy()
            y = pd.to_numeric(dfv[(j,"y")], errors="coerce").to_numpy()
            if np.any(np.isfinite(x)): xs.append(np.nanmax(x))
            if np.any(np.isfinite(y)): ys.append(np.nanmax(y))
    W = float(np.nanmax(xs)) if xs else 0.0
    H = float(np.nanmax(ys)) if ys else 0.0
    return W, H
def sampson_distance(F, x_src, x_tgt):
    # x are (u,v); return scalar Sampson distance in pixels
    x1 = np.array([x_src[0], x_src[1], 1.0], dtype=float)
    x2 = np.array([x_tgt[0], x_tgt[1], 1.0], dtype=float)
    Fx1 = F @ x1
    Ftx2 = F.T @ x2
    num = (x2.T @ F @ x1)**2
    den = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-12
    return float(np.sqrt(num / den))

def build_projection_matrix(K, R, t):
    P = np.zeros((3,4), dtype=float)
    P[:,:3] = R
    P[:, 3] = t
    return K @ P

def triangulate_linear(xi, Pi, xj, Pj):
    # xi, xj are (u,v); Pi, Pj are 3x4
    x1, y1 = xi
    x2, y2 = xj
    A = np.vstack([
        x1*Pi[2,:] - Pi[0,:],
        y1*Pi[2,:] - Pi[1,:],
        x2*Pj[2,:] - Pj[0,:],
        y2*Pj[2,:] - Pj[1,:],
    ])
    # Solve A X = 0
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    X = Xh[:3] / (Xh[3] + 1e-12)
    return X  # 3D point in target/world convention of extrinsics
# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser(description="Plot epipolar lines from all other cameras onto a target camera with a frame slider.")
    ap.add_argument("--calib",type=Path,required=True)
    ap.add_argument("--csv_dir",type=Path,required=True)
    ap.add_argument("--target",type=str,required=True,help="e.g., Cam6")
    ap.add_argument("--joint",type=str,required=True,help="e.g., right_foot_index")
    ap.add_argument("--min_conf",type=float,default=0.0)
    ap.add_argument("--force_size",type=str,default=None,
                    help="Override calibration image size as 'H x W', e.g. 1280x720")
    ap.add_argument("--auto_scale",action="store_true",
                    help="If set, scale CSV coords to match the (H,W) image size.")
    args=ap.parse_args()

    cams=load_calibration(args.calib)
    label_to_camkey={cam_label_from_name(v["name"]) or k:k for k,v in cams.items()}

    target_label=args.target.capitalize()
    if target_label not in label_to_camkey:
        raise SystemExit(f"Target '{target_label}' not found in TOML names: {sorted(label_to_camkey.keys())}")
    target_key=label_to_camkey[target_label]

    # Image size from TOML (H,W), but allow override
    Ht,Wt=cams[target_key]["size"]
    if args.force_size:
        try:
            H_forced, W_forced = map(int, re.split(r"[xX]", args.force_size.strip()))
            Ht, Wt = H_forced, W_forced
        except Exception:
            raise SystemExit("Could not parse --force_size. Use e.g. 1280x720")

    # Load CSVs and map to labels via filename (expects 'CamX' in filename)
    csv_files=list(args.csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in: {args.csv_dir}")
    df_by_label={}
    for p in csv_files:
        m=re.search(r"(Cam\d+)",p.name, re.IGNORECASE)
        if not m: 
            continue
        lab=m.group(1).capitalize()
        df_by_label[lab]=load_dlc_multidx(p)

    if target_label not in df_by_label:
        raise SystemExit(f"Target '{target_label}' not found among loaded CSVs. Found: {sorted(df_by_label.keys())}")

    # Target data
    target_df=df_by_label[target_label]
    target_track=extract_joint(target_df,args.joint)      # (x,y,p) in DLC pixel coords
    target_all_xy=extract_all_points_xy(target_df)        # all joints (x,y)

    # Optional auto-scale to (Wt,Ht)
    if args.auto_scale:
        Wd, Hd = estimate_csv_frame_size(target_df)       # DLC width/height
        sx = Wt / Wd if Wd > 0 else 1.0
        sy = Ht / Hd if Hd > 0 else 1.0
        target_track[:, :2] *= [sx, sy]
        for j in target_all_xy:
            target_all_xy[j] *= [sx, sy]

    # Trim length
    T=min(len(next(iter(target_all_xy.values()))),len(target_track))
    target_track=target_track[:T]
    for j in target_all_xy:
        target_all_xy[j]=target_all_xy[j][:T]

    # Sources
    source_labels=[lab for lab in df_by_label if lab!=target_label and lab in label_to_camkey]
    if len(source_labels)==0:
        raise SystemExit("No other camera CSVs found to act as sources.")

    # Fundamentals F_{target <- source}
    F={}
    for lab in source_labels:
        i_key=label_to_camkey[lab]
        Ki,Ri,ti=cams[i_key]["K"],cams[i_key]["R"],cams[i_key]["t"]
        Kj,Rj,tj=cams[target_key]["K"],cams[target_key]["R"],cams[target_key]["t"]
        F[(lab,target_label)]=fundamental_from_extrinsics(Ki,Kj,Ri,ti,Rj,tj)

    # Source joint tracks (optionally scale too)
    source_tracks={}
    for lab in source_labels:
        tr=extract_joint(df_by_label[lab],args.joint)
        if args.auto_scale:
            tr[:, :2] *= [sx, sy]
        source_tracks[lab]=tr[:T]

        # Build projection matrices for all cameras once
    P = {}
    for lab, key in label_to_camkey.items():
        # Only for cams we actually have in TOML
        if key in cams:
            P[lab] = build_projection_matrix(cams[key]["K"], cams[key]["R"], cams[key]["t"])
    # Also store target
    P[target_label] = build_projection_matrix(cams[target_key]["K"], cams[target_key]["R"], cams[target_key]["t"])


    # --------- Plot with slider ---------
    fig,ax=plt.subplots(figsize=(8,6)); plt.subplots_adjust(bottom=0.15)
    ax.set_xlim(0,Wt); ax.set_ylim(Ht,0)
    ax.set_title(f"{target_label}: epipolar lines for joint '{args.joint}'")
    # draw the image rectangle
    ax.plot([0,Wt,Wt,0,0],[0,0,Ht,Ht,0],lw=0.5,color='k',alpha=0.4)

    # All joints (faint)
    all_scats=[(j,arr,ax.scatter([],[],s=10,alpha=0.2)) for j,arr in target_all_xy.items()]
    # Highlight chosen joint
    joint_point=ax.scatter([],[],s=50,marker='x')
    # Lines per source
    line_artists = {
        lab: ax.plot([], [], linestyle="--", label=f"{lab}->{target_label}")[0]
        for lab in source_labels
    }

        # Reprojected-from-pair marker (star)
    reproj_point = ax.scatter([], [], s=70, marker='*')

    # Text box for numeric errors
    err_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    ax.legend(loc='lower left',fontsize=8)

    # Slider
    axframe=plt.axes([0.12,0.04,0.75,0.04])
    sframe=Slider(axframe,"frame",0,T-1,valinit=0,valstep=1)

    def update(val):
        f = int(sframe.val)

        # all joints
        for _, arr, scat in all_scats:
            xy = arr[f]
            scat.set_offsets([] if np.any(np.isnan(xy)) else xy.reshape(1, 2))

        # chosen joint in target
        xyj = target_track[f, :2]
        joint_point.set_offsets([] if np.any(np.isnan(xyj)) else xyj.reshape(1, 2))

        # epipolar lines from sources
        for lab in source_labels:
            xs = source_tracks[lab][f, :2]
            ps = source_tracks[lab][f, 2]
            ln = line_artists[lab]
            if np.any(np.isnan(xs)) or ps < args.min_conf:
                ln.set_data([], [])
                continue
            l = epipolar_line(F[(lab, target_label)], xs)
            seg = clip_line_to_image(l, Wt, Ht)
            if seg:
                (u1, v1), (u2, v2) = seg
                ln.set_data([u1, u2], [v1, v2])
            else:
                ln.set_data([], [])

        # ---------- NUMERIC DIAGNOSTICS ----------
        # 1) Sampson distances from each source
        samps = []
        for lab in source_labels:
            xs = source_tracks[lab][f, :2]
            ps = source_tracks[lab][f, 2]
            if np.any(np.isnan(xs)) or ps < args.min_conf:
                samps.append((lab, np.nan))
                continue
            d = sampson_distance(F[(lab, target_label)], xs, xyj)
            samps.append((lab, d))

        # 2) Reprojection from the two best-confidence sources
        good = [(lab, source_tracks[lab][f, :2], source_tracks[lab][f, 2])
                for lab in source_labels
                if np.all(np.isfinite(source_tracks[lab][f, :2])) and source_tracks[lab][f, 2] >= args.min_conf]

        reproj_uv = None
        if len(good) >= 2:
            good_sorted = sorted(good, key=lambda t: -t[2])[:2]
            (labA, xA, _), (labB, xB, _) = good_sorted
            X = triangulate_linear(xA, P[labA], xB, P[labB])
            uvw = P[target_label] @ np.hstack([X, 1.0])
            reproj_uv = (uvw[0] / (uvw[2] + 1e-12), uvw[1] / (uvw[2] + 1e-12))

        # plot/update reprojection and assemble text
        if reproj_uv is not None and np.all(np.isfinite(reproj_uv)):
            reproj_point.set_offsets(np.array(reproj_uv).reshape(1, 2))
            reproj_err = float(np.linalg.norm(np.array(reproj_uv) - xyj))
        else:
            reproj_point.set_offsets(np.empty((0, 2)))
            reproj_err = np.nan

        lines = ["Sampson px: " + ", ".join(
            f"{lab}:{d:.1f}" if np.isfinite(d) else f"{lab}:–" for lab, d in samps)]
        if np.isfinite(reproj_err) and len(good) >= 2:
            pair_names = f"{good_sorted[0][0]}+{good_sorted[1][0]}"
            lines.append(f"Reproj ({pair_names}→{target_label}) err: {reproj_err:.1f} px")
        err_text.set_text("\n".join(lines))

        fig.canvas.draw_idle()

    sframe.on_changed(update)
    # arrow keys
    def on_key(event):
        if event.key in ['left','right']:
            v=int(sframe.val)
            v=max(0,v-1) if event.key=='left' else min(T-1,v+1)
            sframe.set_val(v)
    fig.canvas.mpl_connect('key_press_event',on_key)

    # Quick print so you can verify sizes once:
    print(f"Image size used (H,W): {(Ht,Wt)}")
    if not args.auto_scale:
        Wd, Hd = estimate_csv_frame_size(target_df)
        print(f"DLC coords approx (W,H): {(Wd,Hd)}  (no scaling applied)")

    update(0); plt.show()

if __name__=='__main__':
    main()
