import argparse
import json
import os

import pandas as pd


def calculate_averages_and_save(output_path, scene_ids):
    total_miou = 0.0
    total_fmiou = 0.0
    total_macc = 0.0
    total_auc = 0.0
    num_scenes = 0

    scene_results = {}

    for scene_id in scene_ids:
        results_file = os.path.join(output_path, f"replica_{scene_id}", "results.json")

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
                miou = data.get("miou", 0.0)
                fmiou = data.get("fmiou", 0.0)
                macc = data.get("macc", 0.0)
                auc = data.get("auc", 0.0)

                total_miou += miou
                total_fmiou += fmiou
                total_macc += macc
                total_auc += auc
                num_scenes += 1

                scene_results[scene_id] = {
                    "miou": miou,
                    "fmiou": fmiou,
                    "macc": macc,
                    "auc": auc,
                }
        else:
            print(f"Warning: Results file not found for scene_id: {scene_id}")

    avg_miou = total_miou / num_scenes if num_scenes > 0 else 0.0
    avg_fmiou = total_fmiou / num_scenes if num_scenes > 0 else 0.0
    avg_macc = total_macc / num_scenes if num_scenes > 0 else 0.0
    avg_auc = total_auc / num_scenes if num_scenes > 0 else 0.0

    output_data = {
        "scene_results": scene_results,
        "averages": {
            "avg_miou": avg_miou,
            "avg_fmiou": avg_fmiou,
            "avg_macc": avg_macc,
            "avg_auc": avg_auc,
        },
    }

    output_file = os.path.join(output_path, "aggregated_results.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Results aggregated and saved to {output_file}")

    convert_json_to_excel(output_file)


def convert_json_to_excel(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    scene_results = data["scene_results"]
    averages = data["averages"]

    rows = []
    for scene, metrics in scene_results.items():
        rows.append(
            {
                "Seq": scene.replace("_", "").capitalize(),
                "AUC": metrics["auc"],
                "FmIoU": metrics["fmiou"],
                "mAcc": metrics["macc"],
                "mIoU": metrics["miou"],
            }
        )

    rows.append(
        {
            "Seq": "All",
            "AUC": averages["avg_auc"],
            "FmIoU": averages["avg_fmiou"],
            "mAcc": averages["avg_macc"],
            "mIoU": averages["avg_miou"],
        }
    )

    df = pd.DataFrame(rows)
    output_file = os.path.splitext(json_file_path)[0] + "_results.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Excel file saved as {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and aggregate performance metrics, then export to Excel."
    )
    parser.add_argument(
        "output_path", type=str, help="Path to the directory containing the results."
    )
    args = parser.parse_args()

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
    calculate_averages_and_save(args.output_path, scene_ids)


if __name__ == "__main__":
    main()
