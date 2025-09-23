import open3d as o3d
import numpy as np
import os

# ----------------------------
# 1. 포인트 클라우드 로드
# ----------------------------
def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd

# ----------------------------
# 2. 최근접 거리 계산
# ----------------------------
def compute_distances(src_pcd, tgt_pcd):
    tgt_kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)
    dists = []
    for p in src_pcd.points:
        [_, _, dist] = tgt_kdtree.search_knn_vector_3d(p, 1)
        dists.append(np.sqrt(dist[0]))
    return np.array(dists)

# ----------------------------
# 3. 메트릭 계산
# ----------------------------
def eval_metrics(pred_path, gt_path, thresh_list=[1.0, 2.0]):
    pred = load_pcd(pred_path)
    gt   = load_pcd(gt_path)

    # Accuracy (Pred→GT)
    acc_dists = compute_distances(pred, gt)
    # Completeness (GT→Pred)
    comp_dists = compute_distances(gt, pred)

    acc = np.mean(acc_dists)
    comp = np.mean(comp_dists)
    overall = (acc + comp) / 2

    results = {
        "acc": acc,
        "comp": comp,
        "overall": overall
    }

    for t in thresh_list:
        acc_ratio = np.mean(acc_dists < t)
        comp_ratio = np.mean(comp_dists < t)
        fscore = 2 * acc_ratio * comp_ratio / (acc_ratio + comp_ratio + 1e-8)
        results[f"acc<{t}mm"] = acc_ratio
        results[f"comp<{t}mm"] = comp_ratio
        results[f"fscore<{t}mm"] = fscore

    return results

# ----------------------------
# 4. 평가할 스캔들 (fusibile 결과)
# ----------------------------
scan_map = {
    1: "/data/dtu/dtu_eval/scan1/points_mvsnet/consistencyCheck-20250923-174401/final3d_model.ply",
    4: "/data/dtu/dtu_eval/scan4/points_mvsnet/consistencyCheck-20250923-174609/final3d_model.ply",
    9: "/data/dtu/dtu_eval/scan9/points_mvsnet/consistencyCheck-20250923-174651/final3d_model.ply",
    10: "/data/dtu/dtu_eval/scan10/points_mvsnet/consistencyCheck-20250923-174411/final3d_model.ply",
    11: "/data/dtu/dtu_eval/scan11/points_mvsnet/consistencyCheck-20250923-174419/final3d_model.ply",
    12: "/data/dtu/dtu_eval/scan12/points_mvsnet/consistencyCheck-20250923-174454/final3d_model.ply",
    13: "/data/dtu/dtu_eval/scan13/points_mvsnet/consistencyCheck-20250923-174502/final3d_model.ply",
    15: "/data/dtu/dtu_eval/scan15/points_mvsnet/consistencyCheck-20250923-174510/final3d_model.ply",
    23: "/data/dtu/dtu_eval/scan23/points_mvsnet/consistencyCheck-20250923-174520/final3d_model.ply",
    24: "/data/dtu/dtu_eval/scan24/points_mvsnet/consistencyCheck-20250923-174529/final3d_model.ply",
    29: "/data/dtu/dtu_eval/scan29/points_mvsnet/consistencyCheck-20250923-174538/final3d_model.ply",
    32: "/data/dtu/dtu_eval/scan32/points_mvsnet/consistencyCheck-20250923-174545/final3d_model.ply",
    33: "/data/dtu/dtu_eval/scan33/points_mvsnet/consistencyCheck-20250923-174553/final3d_model.ply",
    34: "/data/dtu/dtu_eval/scan34/points_mvsnet/consistencyCheck-20250923-174601/final3d_model.ply",
    49: "/data/dtu/dtu_eval/scan49/points_mvsnet/consistencyCheck-20250923-174624/final3d_model.ply",
    62: "/data/dtu/dtu_eval/scan62/points_mvsnet/consistencyCheck-20250923-174632/final3d_model.ply",
    75: "/data/dtu/dtu_eval/scan75/points_mvsnet/consistencyCheck-20250923-174639/final3d_model.ply",
    77: "/data/dtu/dtu_eval/scan77/points_mvsnet/consistencyCheck-20250923-174646/final3d_model.ply",
    110: "/data/dtu/dtu_eval/scan110/points_mvsnet/consistencyCheck-20250923-174427/final3d_model.ply",
    114: "/data/dtu/dtu_eval/scan114/points_mvsnet/consistencyCheck-20250923-174436/final3d_model.ply",
    118: "/data/dtu/dtu_eval/scan118/points_mvsnet/consistencyCheck-20250923-174445/final3d_model.ply",
}

gt_root = "/data/dtu/dtu_gt/stl"  # ground-truth 경로

# ----------------------------
# 5. 스캔별 평가 실행
# ----------------------------
all_results = []

for scan_id, pred_path in scan_map.items():
    gt_path = os.path.join(gt_root, f"stl{scan_id:03d}_total.ply")
    print(f"==== Scan {scan_id} ====")
    res = eval_metrics(pred_path, gt_path)
    print(res)
    all_results.append(res)

# ----------------------------
# 6. 평균 성능 출력 (논문 Table 1 스타일)
# ----------------------------
def avg_metric(results, key):
    return np.mean([r[key] for r in results])

print("\n==== Average over scans ====")
print(f"Accuracy (mm): {avg_metric(all_results, 'acc'):.4f}")
print(f"Completeness (mm): {avg_metric(all_results, 'comp'):.4f}")
print(f"Overall (mm): {avg_metric(all_results, 'overall'):.4f}")
print(f"Acc<1mm: {avg_metric(all_results, 'acc<1.0mm'):.3f}")
print(f"Comp<1mm: {avg_metric(all_results, 'comp<1.0mm'):.3f}")
print(f"F-score@1mm: {avg_metric(all_results, 'fscore<1.0mm'):.3f}")
print(f"Acc<2mm: {avg_metric(all_results, 'acc<2.0mm'):.3f}")
print(f"Comp<2mm: {avg_metric(all_results, 'comp<2.0mm'):.3f}")
print(f"F-score@2mm: {avg_metric(all_results, 'fscore<2.0mm'):.3f}")
