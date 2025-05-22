## Dataset Preparation

This guide walks you through preparing the datasets required for the project. We cover three datasets: **MARS**, **MM-Fi**, and **mRI**. Each section explains how to download, organize, and generate the data.

---

### 1. MARS

The MARS dataset contains multi-view action sequences from the **Mars** repository.

1. **Clone the repository**

   ```bash
   git clone https://github.com/SizheAn/MARS.git
   ```

2. **Generate the dataset**

   From the project root, run:

   ```bash
   python tools/create_data.py mars \
       --root-path ${MARS_REPO_ROOT} \
       --out-dir data/mars
   ```

   * **Option**: At the prompt, type `wo` to exclude outliers and build the dataset without them.

---

### 2. MM-Fi

The MM-Fi dataset combines millimeter-wave radar with video-based action labels.

1. **Download and extract**

   Follow instructions in the [MM-Fi dataset repo](https://github.com/ybhbingo/MMFi_dataset). Organize the files as:

   ```plain
   mmfi/
   ├── filtered_mmwave/           # Extracted from filtered_mmwave.zip
   ├── MMFi_Dataset/              # Main dataset folder
   ├── MMFi_actions_segments/     # Segmented action data
   └── MMFi_action_segments.csv   # Annotations CSV
   ```

2. **Align filtered radar data**

   From the project root, run:

   ```bash
   python tools/dataset_converters/mmfi_align_filtered.py \
       -d ${MMFI_ROOT}/MMFi_Dataset \
       -f ${MMFI_ROOT}/filtered_mmwave
   ```

3. **Generate the dataset**

   ```bash
   python tools/create_data mmfi \
       --root-path ${MMFI_ROOT}/MMFi_Dataset \
       --out-dir data/mmfi
   ```

   * **Options**:

     * Type `both` to build both filtered and unfiltered versions.
     * Type `y` to unify the coordinate system between radar points and skeleton features.

---

### 3. mRI

The mRI dataset provides motion capture and radar imaging data.

1. **Clone the repository**

   ```bash
   git clone https://github.com/SizheAn/mRI.git
   ```

2. **Download & extract**

   Ensure the following structure under `${MRI_ROOT}/dataset_release`:

   ```plain
   mRI/
   └── dataset_release/
       ├── aligned_data/
       ├── features/
       ├── model/
       └── raw_data/
   ```

3. **Generate the dataset**

   From the project root, run:

   ```bash
   python tools/create_data.py mri \
       --root-path ${MRI_ROOT}/dataset_release \
       --out-dir data/mri
   ```

---

### Output Structure

After running the above commands, your `data/` directory will contain:

```plain
data/
├── mars/      # Processed MARS dataset
├── mmfi/      # Processed MM-Fi dataset
└── mri/       # Processed mRI dataset
```

All datasets will be ready for use in training and evaluation scripts.
