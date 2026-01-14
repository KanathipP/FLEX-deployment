#!/usr/bin/env python3
"""
FLEX Dataset Setup Script

Downloads and organizes EEG datasets for Federated Learning experiments.

Datasets:
1. Motor Imagery (BCI Competition IV Dataset 2a)
   - Source: https://www.bbci.de/competition/iv/
   - 9 subjects, 22 EEG channels, 4 classes
   - Assigned to: Hospital A (partition 0), Hospital B (partition 1)

2. Mental Arithmetic (PhysioNet)
   - Source: https://physionet.org/content/eegmat/1.0.0/
   - 36 subjects, 21 EEG channels, 2 classes
   - Assigned to: Hospital C (partition 2), Hospital D (partition 3)

Usage:
    python scripts/setup_data.py                    # Download and setup all data
    python scripts/setup_data.py --task motor       # Only Motor Imagery
    python scripts/setup_data.py --task mental      # Only Mental Arithmetic
    python scripts/setup_data.py --output ./data    # Custom output directory
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import argparse
import urllib.request
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

# Default output directory (relative to project root)
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "client_data"

# Hospital to partition mapping
HOSPITAL_MAPPING = {
    "hospital-a": 0,  # Motor Imagery - Subject A01-A04
    "hospital-b": 1,  # Motor Imagery - Subject A05-A09
    "hospital-c": 2,  # Mental Arithmetic - Subject 00-17
    "hospital-d": 3,  # Mental Arithmetic - Subject 18-35
}

# Dataset URLs
PHYSIONET_MENTAL_ARITHMETIC_URL = "https://physionet.org/files/eegmat/1.0.0/"
BCI_COMPETITION_MOTOR_IMAGERY_URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"

# ============================================================================
# Utility Functions
# ============================================================================

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step: int, text: str):
    """Print a step indicator."""
    print(f"[Step {step}] {text}")


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"  📥 Downloading: {desc or url}")
        
        # Create parent directory if needed
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  ❌ Error downloading: {e}")
        return False


def create_metadata(partition_dir: Path, task: str, hospital: str):
    """Create metadata.json file for a partition."""
    # Extract partition number from directory name (e.g., "client0" -> 0, "0" -> 0)
    import re
    match = re.search(r'\d+', partition_dir.name)
    partition_num = int(match.group()) if match else 0
    
    metadata = {
        "task": task,
        "hospital": hospital,
        "partition": partition_num
    }
    
    metadata_path = partition_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  📝 Created: {metadata_path}")


# ============================================================================
# Motor Imagery Dataset (BCI Competition IV - Dataset 2a)
# ============================================================================

def download_motor_imagery(output_dir: Path) -> bool:
    """Download and organize Motor Imagery dataset from BCI Competition IV."""
    print_header("Motor Imagery Dataset (BCI Competition IV)")
    
    # Check if data already exists
    partition_0 = output_dir / "0"
    partition_1 = output_dir / "1"
    
    if partition_0.exists() and list(partition_0.glob("*.gdf")):
        print("  ℹ️  Motor Imagery data already exists in partition 0")
        if partition_1.exists() and list(partition_1.glob("*.gdf")):
            print("  ℹ️  Motor Imagery data already exists in partition 1")
            print("  ⏭️  Skipping download (data exists)")
            return True
    
    print_step(1, "Downloading BCI Competition IV Dataset 2a...")
    
    # Download zip to temp location
    temp_zip = output_dir / "_temp_BCICIV_2a_gdf.zip"
    
    if not download_file(BCI_COMPETITION_MOTOR_IMAGERY_URL, temp_zip, desc="BCICIV_2a_gdf.zip (~650MB)"):
        print("  ❌ Failed to download Motor Imagery dataset")
        return False
    
    # Extract and organize
    success = setup_motor_imagery_from_zip(temp_zip, output_dir)
    
    # Cleanup temp zip
    if temp_zip.exists():
        temp_zip.unlink()
    
    return success


def setup_motor_imagery_from_zip(zip_path: Path, output_dir: Path):
    """Extract and organize Motor Imagery data from downloaded zip."""
    print_step(2, "Extracting Motor Imagery dataset...")
    
    if not zip_path.exists():
        print(f"  ❌ File not found: {zip_path}")
        print("  Please download the dataset first")
        return False
    
    # Create temp extraction directory
    temp_dir = output_dir / "_temp_motor"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract zip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)
    
    print_step(3, "Organizing files into hospital partitions (train/test)...")
    
    # Hospital A (partition 0): Subjects A01-A04 (4 subjects)
    # Hospital B (partition 1): Subjects A05-A08 (4 subjects)
    hospital_a_subjects = ["A01", "A02", "A03", "A04"]
    hospital_b_subjects = ["A05", "A06", "A07", "A08"]
    
    # Create partition directories with train/test subfolders
    partition_0 = output_dir / "client0"
    partition_1 = output_dir / "client1"
    
    partition_0_train = partition_0 / "train"
    partition_0_test = partition_0 / "test"
    partition_1_train = partition_1 / "train"
    partition_1_test = partition_1 / "test"
    
    for d in [partition_0_train, partition_0_test, partition_1_train, partition_1_test]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find and copy GDF files
    # *T.gdf = Training data -> train/
    # *E.gdf = Evaluation data -> test/
    gdf_files = list(temp_dir.rglob("*.gdf"))
    
    for gdf_file in gdf_files:
        subject_id = gdf_file.stem[:3]  # e.g., "A01" from "A01T.gdf"
        is_training = gdf_file.stem.endswith("T")
        
        if subject_id in hospital_a_subjects:
            dest_dir = partition_0_train if is_training else partition_0_test
            shutil.copy2(gdf_file, dest_dir / gdf_file.name)
            split_type = "train" if is_training else "test"
            print(f"  📁 {gdf_file.name} → partition 0/{split_type} (Hospital A)")
        elif subject_id in hospital_b_subjects:
            dest_dir = partition_1_train if is_training else partition_1_test
            shutil.copy2(gdf_file, dest_dir / gdf_file.name)
            split_type = "train" if is_training else "test"
            print(f"  📁 {gdf_file.name} → partition 1/{split_type} (Hospital B)")
    
    # Create metadata files
    create_metadata(partition_0, "motor_imaginary", "hospital-a")
    create_metadata(partition_1, "motor_imaginary", "hospital-b")
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    
    print("\n  ✅ Motor Imagery dataset setup complete!")
    return True


# ============================================================================
# Mental Arithmetic Dataset (PhysioNet)
# ============================================================================

def download_mental_arithmetic(output_dir: Path) -> bool:
    """Download and organize Mental Arithmetic dataset from PhysioNet."""
    print_header("Mental Arithmetic Dataset (PhysioNet)")
    
    print_step(1, "Downloading from PhysioNet...")
    
    # PhysioNet dataset: Subject{XX}_1.edf (baseline) and Subject{XX}_2.edf (task)
    # _1.edf -> train/ (baseline recordings)
    # _2.edf -> test/ (task recordings)
    
    base_url = "https://physionet.org/files/eegmat/1.0.0/"
    
    # Create partition directories with train/test subfolders
    partition_2 = output_dir / "client2"  # Hospital C
    partition_3 = output_dir / "client3"  # Hospital D
    
    partition_2_train = partition_2 / "train"
    partition_2_test = partition_2 / "test"
    partition_3_train = partition_3 / "train"
    partition_3_test = partition_3 / "test"
    
    for d in [partition_2_train, partition_2_test, partition_3_train, partition_3_test]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Download subjects (4 per hospital for balance)
    # Hospital C: Subjects 00-03 (4 subjects)
    # Hospital D: Subjects 04-07 (4 subjects)
    subjects_hospital_c = list(range(0, 4))   # 0-3
    subjects_hospital_d = list(range(4, 8))   # 4-7
    
    def download_subject(subject_num: int, train_dir: Path, test_dir: Path) -> bool:
        """Download EDF files for a single subject into train/test folders."""
        subject_str = f"Subject{subject_num:02d}"
        
        # _1.edf -> train (baseline)
        train_filename = f"{subject_str}_1.edf"
        train_dest = train_dir / train_filename
        if not train_dest.exists():
            url = f"{base_url}{train_filename}"
            if not download_file(url, train_dest, desc=f"{train_filename} (train)"):
                return False
        else:
            print(f"  ⏭️  Skipping (exists): {train_filename}")
        
        # _2.edf -> test (task)
        test_filename = f"{subject_str}_2.edf"
        test_dest = test_dir / test_filename
        if not test_dest.exists():
            url = f"{base_url}{test_filename}"
            if not download_file(url, test_dest, desc=f"{test_filename} (test)"):
                return False
        else:
            print(f"  ⏭️  Skipping (exists): {test_filename}")
        
        return True
    
    print_step(2, "Downloading Hospital C subjects (00-09)...")
    for subj in subjects_hospital_c:
        if not download_subject(subj, partition_2_train, partition_2_test):
            print(f"  ⚠️  Failed to download Subject{subj:02d}")
    
    print_step(3, "Downloading Hospital D subjects (10-19)...")
    for subj in subjects_hospital_d:
        if not download_subject(subj, partition_3_train, partition_3_test):
            print(f"  ⚠️  Failed to download Subject{subj:02d}")
    
    # Create metadata files
    print_step(4, "Creating metadata files...")
    create_metadata(partition_2, "mental_arithmetic", "hospital-c")
    create_metadata(partition_3, "mental_arithmetic", "hospital-d")
    
    print("\n  ✅ Mental Arithmetic dataset setup complete!")
    return True


# ============================================================================
# Main Setup Function
# ============================================================================

def setup_all_data(output_dir: Path, motor_zip: Optional[Path] = None):
    """Setup all datasets."""
    print_header("FLEX Dataset Setup")
    
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Motor Imagery
    if motor_zip and motor_zip.exists():
        print_header("Motor Imagery Dataset (BCI Competition IV)")
        setup_motor_imagery_from_zip(motor_zip, output_dir)
    else:
        download_motor_imagery(output_dir)
    
    # Setup Mental Arithmetic
    download_mental_arithmetic(output_dir)
    
    # Print summary
    print_header("Setup Summary")
    
    for partition_id in range(4):
        partition_dir = output_dir / str(partition_id)
        if partition_dir.exists():
            files = list(partition_dir.iterdir())
            metadata_file = partition_dir / "metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                task = metadata.get("task", "unknown")
                hospital = metadata.get("hospital", "unknown")
            else:
                task = "unknown"
                hospital = "unknown"
            
            data_files = [f for f in files if f.suffix in ['.gdf', '.edf']]
            print(f"  Partition {partition_id} ({hospital}): {len(data_files)} data files [{task}]")
        else:
            print(f"  Partition {partition_id}: Not setup")
    
    print("\n✅ Dataset setup complete!")
    print(f"\nData location: {output_dir.absolute()}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FLEX Dataset Setup - Download and organize EEG datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_data.py                          # Setup all datasets
  python setup_data.py --task mental            # Only Mental Arithmetic
  python setup_data.py --motor-zip ./data.zip   # Use downloaded BCI zip
  python setup_data.py --output ./my_data       # Custom output directory
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for datasets (default: data_multitask/)"
    )
    
    parser.add_argument(
        "--task", "-t",
        choices=["all", "motor", "mental"],
        default="all",
        help="Which dataset to setup (default: all)"
    )
    
    parser.add_argument(
        "--motor-zip",
        type=Path,
        help="Path to downloaded BCI Competition IV zip file"
    )
    
    args = parser.parse_args()
    
    # Resolve output path
    output_dir = args.output.resolve()
    
    if args.task == "mental":
        download_mental_arithmetic(output_dir)
    elif args.task == "motor":
        if args.motor_zip:
            setup_motor_imagery_from_zip(args.motor_zip, output_dir)
        else:
            download_motor_imagery(output_dir)
    else:
        setup_all_data(output_dir, args.motor_zip)


if __name__ == "__main__":
    main()
