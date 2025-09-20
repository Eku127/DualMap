import argparse
import json
import os

import xlsxwriter

# ========== Argument Parser ==========
parser = argparse.ArgumentParser(description="Generate XLSX from evaluation results.")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["replica", "scannet"],
    help="Dataset name",
)
parser.add_argument(
    "--eval_path", type=str, required=True, help="Path to the evaluation output folder"
)

args = parser.parse_args()
dataset_name = args.dataset
eval_path = args.eval_path

# ========== Scene IDs ==========
if dataset_name == "replica":
    scene_ids = [
        "office_0",
        "office_1",
        "office_2",
        "office_3",
        "office_4",
        "room_0",
        "room_1",
        "room_2",
    ]
elif dataset_name == "scannet":
    scene_ids = [
        "scene0011_00",
        "scene0050_00",
        "scene0231_00",
        "scene0378_00",
        "scene0518_00",
    ]

# ========== Output File Name ==========
folder_name = os.path.basename(os.path.dirname(eval_path.rstrip("/")))
save_path = f"{folder_name.replace('-', '_')}.{dataset_name}.xlsx"

# ========== Data Aggregation ==========
auc, fmiou, macc, miou, obj_num = [], [], [], [], []

for scene_id in scene_ids:
    result_path = os.path.join(eval_path, f"{dataset_name}_{scene_id}", "results.json")
    with open(result_path) as f:
        data = json.load(f)
    auc.append(data["auc"])
    fmiou.append(data["fmiou"])
    macc.append(data["macc"])
    miou.append(data["miou"])
    obj_num.append(data["obj_num"])

# ========== Compute Averages ==========
auc.append(sum(auc) / len(auc))
fmiou.append(sum(fmiou) / len(fmiou))
macc.append(sum(macc) / len(macc))
miou.append(sum(miou) / len(miou))
obj_num.append(sum(obj_num) / len(obj_num))
scene_ids.append("Average")

# ========== Round Values ==========
auc = [round(v, 6) for v in auc]
fmiou = [round(v, 6) for v in fmiou]
macc = [round(v, 6) for v in macc]
miou = [round(v, 6) for v in miou]
obj_num = [round(v, 0) for v in obj_num]

# ========== Write to Excel ==========
workbook = xlsxwriter.Workbook(save_path)
worksheet = workbook.add_worksheet()

headers = ["Scene ID", "AUC", "FmIoU", "mAcc", "mIoU", "obj"]
worksheet.write_row("A1", headers)

for row_num, scene_id in enumerate(scene_ids, start=1):
    worksheet.write(row_num, 0, scene_id)
    worksheet.write(row_num, 1, auc[row_num - 1])
    worksheet.write(row_num, 2, fmiou[row_num - 1])
    worksheet.write(row_num, 3, macc[row_num - 1])
    worksheet.write(row_num, 4, miou[row_num - 1])
    worksheet.write(row_num, 5, obj_num[row_num - 1])

workbook.close()
print(f"Saved result to {save_path}")
