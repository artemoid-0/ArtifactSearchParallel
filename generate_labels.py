import os
import csv

def generate_labels_csv_from_filenames(image_dir, filename_to_label, label_id_to_name, output_csv_path):
    """
    Generates a CSV file with image labels based on file names.

    Parameters:
        image_dir (str): path to the folder containing the images (.png)
        filename_to_label (dict): dictionary {filename: label (int)}
        label_id_to_name (dict): dictionary {label (int): human-readable class name (str)}
        output_csv_path (str): path to the output CSV file
    """
    rows = []

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".png"):
            continue

        # Get the label: if the file is not in the dictionary, default to 0
        label = filename_to_label.get(fname, 0)
        label_name = label_id_to_name.get(label, f"class_{label}")

        rows.append((fname, label, label_name))

    # Save to CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'label', 'label_name'])
        writer.writerows(rows)

    print(f"CSV saved to {output_csv_path} ({len(rows)} entries)")

# Paths to the image directories
train_dir = r"D:\DATASETS\artifact_dataset\train"
test_dir = r"D:\DATASETS\artifact_dataset\test"

# Labels for images in train (only those that are not "clean")
train_filename_to_label = {
    "image_00002_0.png": 1,
    "image_00005_0.png": 1,
    "image_00015_0.png": 1,
    "image_00029_0.png": 1,
    "image_00030_0.png": 1,
    "image_00069_0.png": 1,
    "image_00106_0.png": 1,
    "image_00111_0.png": 1,
    "image_00115_0.png": 1,
    "image_00138_0.png": 1,
    "image_00142_0.png": 1,
    "image_00146_0.png": 1,
    "image_00148_0.png": 1,
    "image_00190_0.png": 1,
    "image_00192_0.png": 1,
    "image_00209_0.png": 1,
    "image_00211_0.png": 1,
    "image_00226_0.png": 1,
    "image_00249_0.png": 1,
    "image_00251_0.png": 1,
    "image_00262_0.png": 1,
    "image_00295_0.png": 1,
    "image_00299_0.png": 1,
    "image_00319_0.png": 1,
    "image_00339_0.png": 1,
    "image_00372_0.png": 1,
    "image_00379_0.png": 1,
    "image_00430_0.png": 1,
    "image_00434_0.png": 1,
    "image_00438_0.png": 1,
    "image_00441_0.png": 1,
    "image_00446_0.png": 1,
    "image_00454_0.png": 1,
    "image_00456_0.png": 1,
    "image_00457_0.png": 1,
    "image_00460_0.png": 1,
    "image_00462_0.png": 1,
    "image_00469_0.png": 1,
    "image_00470_0.png": 1,
    "image_00493_0.png": 1,
    "image_00500_0.png": 1,
    "image_00501_0.png": 1,
    "image_00504_0.png": 1,
    "image_00506_0.png": 1,
    "image_00523_0.png": 1,
    "image_00530_0.png": 1,
    "image_00537_0.png": 1,
    "image_00544_0.png": 1,
    "image_00569_0.png": 1,
    "image_00575_0.png": 1,
    "image_00590_0.png": 1,
    "image_00592_0.png": 1,
    "image_00604_0.png": 1,
    "image_00620_0.png": 1,
    "image_00625_0.png": 1,
    "image_00632_0.png": 1,
    "image_00646_0.png": 1,
    "image_00669_0.png": 1,
    "image_00680_0.png": 1,
    "image_00687_0.png": 1,
    "image_00714_0.png": 1,
    "image_00718_0.png": 1,
    "image_00727_0.png": 1,
    "image_00733_0.png": 1,
    "image_00744_0.png": 1,
    "image_00756_0.png": 1,
    "image_00782_0.png": 1,
    "image_00786_0.png": 1,
    "image_00787_0.png": 1,
    "image_00788_0.png": 1,
    "image_00789_0.png": 1,
    "image_00790_0.png": 1,
    "image_00795_0.png": 1,
    "image_00796_0.png": 1,
    "image_00804_0.png": 1,
    "image_00805_0.png": 1,
    "image_00811_0.png": 1,
    "image_00812_0.png": 1,
    "image_00814_0.png": 1,
    "image_00815_0.png": 1,
    "image_00817_0.png": 1,
    "image_00830_0.png": 1,
    "image_00835_0.png": 1,
    "image_00838_0.png": 1,
    "image_00853_0.png": 1,
    "image_00872_0.png": 1,
    "image_00878_0.png": 1,
    "image_00887_0.png": 1,
    "image_00897_0.png": 1,
    "image_00931_0.png": 1,
    "image_00932_0.png": 1,
    "image_00939_0.png": 1,
    "image_00946_0.png": 1,
    "image_00949_0.png": 1,
    "image_00951_0.png": 1,
    "image_00956_0.png": 1,
    "image_00959_0.png": 1,
    "image_00971_0.png": 1,
    "image_00979_0.png": 1,
    "image_01004_0.png": 1,
    "image_01009_0.png": 1,
    "image_01016_0.png": 1,
    "image_01037_0.png": 1,
    "image_01101_0.png": 1,
    "image_01113_0.png": 1,
    "image_01114_0.png": 1,
    "image_01116_0.png": 1,
    "image_01120_0.png": 1,
    "image_01124_0.png": 1,
    "image_01140_0.png": 1,
    "image_01143_0.png": 1,
    "image_01189_0.png": 1,
    "image_01210_0.png": 1,
    "image_01237_0.png": 1,
    "image_01241_0.png": 1,
    "image_01276_0.png": 1,
    "image_01286_0.png": 1,
    "image_01292_0.png": 1,
    "image_01304_0.png": 1,
    "image_01307_0.png": 1,
    "image_01315_0.png": 1,
    "image_01323_0.png": 1,
    "image_01334_0.png": 1,
    "image_01336_0.png": 1,
    "image_01341_0.png": 1,
    "image_01338_0.png": 1,
    "image_01412_0.png": 1,
    "image_01413_0.png": 1,
    "image_01429_0.png": 1,
    "image_01457_0.png": 1,
    "image_01499_0.png": 1,
    "image_01510_0.png": 1,
    "image_01519_0.png": 1,
    "image_01530_0.png": 1,
    "image_01559_0.png": 1,
    "image_01572_0.png": 1,
    "image_01577_0.png": 1,
    "image_01589_0.png": 1,
    "image_01608_0.png": 1,
    "image_01609_0.png": 1,
    "image_01646_0.png": 1,
    "image_01647_0.png": 1,
    "image_01648_0.png": 1,
    "image_01655_0.png": 1,
    "image_01657_0.png": 1,
    "image_01668_0.png": 1,
    "image_01683_0.png": 1,
    "image_01695_0.png": 1,
    "image_01700_0.png": 1,
    "image_01720_0.png": 1,
    "image_01725_0.png": 1,
    "image_01740_0.png": 1,
    "image_01757_0.png": 1,
    "image_01764_0.png": 1,
    "image_01765_0.png": 1,
    "image_01770_0.png": 1,
    "image_01799_0.png": 1,
}

# Labels for images in test
test_filename_to_label = {
    "image_00003_0.png": 1,
    "image_00011_0.png": 1,
    "image_00018_0.png": 1,
    "image_00025_0.png": 1,
    "image_00030_0.png": 1,
    "image_00045_0.png": 1,
    "image_00050_0.png": 1,
    "image_00060_0.png": 1,
    "image_00061_0.png": 1,
    "image_00062_0.png": 1,
    "image_00066_0.png": 1,
    "image_00074_0.png": 1,
    "image_00102_0.png": 1,
    "image_00111_0.png": 1,
    "image_00133_0.png": 1,
    "image_00151_0.png": 1,
    "image_00167_0.png": 1,
}

# Mapping label IDs to human-readable names
label_id_to_name = {
    0: "clean",
    1: "object",
    # 2: "shadow"
}

# Generate CSV labels
generate_labels_csv_from_filenames(train_dir, train_filename_to_label, label_id_to_name, "train_labels.csv")
generate_labels_csv_from_filenames(test_dir, test_filename_to_label, label_id_to_name, "test_labels.csv")
