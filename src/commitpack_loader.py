"""
Load and process CommitPackFT dataset (Python only)
"""
from datasets import load_dataset
from typing import List, Dict
import os
import json
from tqdm import tqdm


"""
Load and process CommitPackFT dataset (Python only)

Note: CommitPackFT uses deprecated dataset scripts. 
We download the raw data from the bigcode/commitpack repository instead.
"""
from datasets import load_dataset
from typing import List, Dict
import os
import json
from tqdm import tqdm
import subprocess


def download_raw_commitpack(output_dir: str = "data/commitpack_raw", num_files: int = 2):
    """
    Download raw CommitPack data using git clone with LFS.
    
    Args:
        output_dir: Directory to clone the repository
        num_files: Number of Python files to download (each is ~500MB with thousands of examples)
    
    Returns:
        Path to the cloned repository
    """
    if os.path.exists(output_dir):
        print(f"✓ CommitPack already cloned at: {output_dir}")
        
        # Check if LFS files are downloaded (not just pointers)
        python_dir = os.path.join(output_dir, "data", "python")
        if os.path.exists(python_dir):
            sample_file = os.path.join(python_dir, "python-0001.jsonl")
            if os.path.exists(sample_file):
                # Check file size - LFS pointers are ~134 bytes, real files are 500MB+
                file_size = os.path.getsize(sample_file)
                if file_size < 1000:  # Less than 1KB = pointer file
                    print(f"  Files are LFS pointers, downloading {num_files} file(s) (~{num_files * 500}MB)...")
                    try:
                        # Pull only first N Python files to save bandwidth
                        # Each file contains thousands of examples
                        file_patterns = [f"data/python/python-{i:04d}.jsonl" for i in range(1, num_files + 1)]
                        
                        for pattern in file_patterns:
                            print(f"    Downloading {pattern}...")
                            subprocess.run([
                                "git", "lfs", "pull",
                                "--include", pattern
                            ], cwd=output_dir, check=True)
                        
                        print(f"  ✓ Downloaded {num_files} Python data file(s)")
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Failed to pull LFS files: {e}")
                        return None
        
        return output_dir
    
    print("Cloning bigcode/commitpack repository...")
    print(f"Note: Will download {num_files} Python file(s) (~{num_files * 500}MB)")
    
    try:
        # Clone without LFS initially (faster)
        print("Step 1/3: Cloning repository structure...")
        subprocess.run([
            "git", "clone",
            "--no-checkout",
            "https://huggingface.co/datasets/bigcode/commitpack",
            output_dir
        ], check=True)
        
        # Sparse checkout for Python only
        print("Step 2/3: Setting up sparse checkout for Python...")
        subprocess.run(["git", "sparse-checkout", "init"], cwd=output_dir, check=True)
        subprocess.run([
            "git", "sparse-checkout", "set", "data/python"
        ], cwd=output_dir, check=True)
        
        # Checkout the files
        subprocess.run(["git", "checkout"], cwd=output_dir, check=True)
        
        print(f"✓ Successfully cloned to: {output_dir}")
        
        # Now pull LFS files for only first N Python files
        print(f"Step 3/3: Downloading {num_files} Python data file(s)...")
        try:
            # Download only the first N files (each has thousands of examples)
            file_patterns = [f"data/python/python-{i:04d}.jsonl" for i in range(1, num_files + 1)]
            
            for i, pattern in enumerate(file_patterns, 1):
                print(f"  [{i}/{num_files}] Downloading {pattern}...")
                subprocess.run([
                    "git", "lfs", "pull",
                    "--include", pattern
                ], cwd=output_dir, check=True)
            
            print(f"  ✓ Downloaded {num_files} Python data file(s)")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to pull LFS files: {e}")
            return None
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        return None


def download_and_filter_commitpack(
    output_file: str = "data/commitpack_python.jsonl",
    max_samples: int = 1000,
    force_download: bool = False
):
    """
    Download CommitPack and filter for Python-only examples.
    
    This uses the raw data from bigcode/commitpack repository.
    The data is organized by language, so we can directly access Python files.
    
    Args:
        output_file: Where to save filtered Python examples
        max_samples: Maximum number of Python samples to save
        force_download: Re-download even if file exists
    
    Returns:
        Path to output file, or None if download fails
    """
    
    # Check if already downloaded
    if os.path.exists(output_file) and not force_download:
        print(f"✓ Filtered dataset already exists at: {output_file}")
        with open(output_file, 'r') as f:
            count = sum(1 for _ in f)
        print(f"  Contains {count} Python examples")
        return output_file
    
    print("Downloading CommitPackFT dataset...")
    print("Note: This will download raw Python data from bigcode/commitpack")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    try:
        # First, download the raw repository (only 1-2 files)
        repo_dir = download_raw_commitpack(num_files=1)  # Download only 1 file (~500MB with ~10k examples)
        if not repo_dir:
            raise Exception("Failed to clone repository")
        
        # Look for Python data files
        python_dir = os.path.join(repo_dir, "data", "python")
        
        if not os.path.exists(python_dir):
            # Try alternative locations
            python_dir = os.path.join(repo_dir, "python")
            
        if not os.path.exists(python_dir):
            print(f"✗ Python directory not found in repository")
            print(f"  Checked: {python_dir}")
            print(f"  Available directories:")
            data_dir = os.path.join(repo_dir, "data")
            if os.path.exists(data_dir):
                print(f"    {os.listdir(data_dir)}")
            else:
                print(f"    {os.listdir(repo_dir)}")
            raise Exception("Python data not found")
        
        # Load Python JSONL files
        import glob
        python_files = glob.glob(os.path.join(python_dir, "*.jsonl"))
        
        if not python_files:
            print(f"✗ No JSONL files found in {python_dir}")
            raise Exception("No Python data files found")
        
        print(f"\nFound {len(python_files)} Python data files")
        print(f"Loading and filtering to {max_samples} examples...")
        
        python_examples = []
        
        for file_path in tqdm(python_files, desc="Loading files"):
            if len(python_examples) >= max_samples:
                break
                
            with open(file_path, 'r') as f:
                for line in f:
                    if len(python_examples) >= max_samples:
                        break
                    
                    try:
                        example = json.loads(line)
                        
                        # Keep only relevant fields (same as CommitPackFT format)
                        filtered_example = {
                            'commit': example.get('commit', ''),
                            'old_file': example.get('old_file', ''),
                            'new_file': example.get('new_file', ''),
                            'old_contents': example.get('old_contents', ''),
                            'new_contents': example.get('new_contents', ''),
                            'subject': example.get('subject', ''),
                            'message': example.get('message', ''),
                        }
                        
                        # Basic filtering (following commitpackft_filters.py)
                        subject = filtered_example['subject'].strip().lower()
                        
                        # Skip if empty or too short
                        if len(subject) < 10 or len(subject.split()) < 4:
                            continue
                        
                        # Skip if old and new contents are the same
                        if filtered_example['old_contents'] == filtered_example['new_contents']:
                            continue
                        
                        # Skip if contents are too long
                        if len(filtered_example['old_contents']) > 50000:
                            continue
                        
                        python_examples.append(filtered_example)
                        
                    except json.JSONDecodeError:
                        continue
        
        # Save to file
        print(f"\nSaving {len(python_examples)} Python examples to {output_file}...")
        with open(output_file, 'w') as f:
            for example in python_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"✓ Saved {len(python_examples)} Python examples")
        return output_file
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to sample data...")
        return None


def load_commitpack_python(
    file_path: str = "data/commitpack_python.jsonl",
    max_samples: int = None
) -> List[Dict]:
    """
    Load filtered Python examples from CommitPackFT.
    
    Args:
        file_path: Path to filtered dataset
        max_samples: Maximum number of samples to load
    
    Returns:
        List of commit examples
    """
    
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}")
        print("Run download first or use sample data")
        return []
    
    examples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            example = json.loads(line)
            examples.append(example)
    
    print(f"Loaded {len(examples)} Python examples from CommitPackFT")
    return examples


def convert_to_bugfix_format(commit_example: Dict) -> Dict:
    """
    Convert CommitPackFT example to bug-fix format.
    
    The old_contents is treated as "buggy" code and new_contents as "fixed".
    """
    return {
        'task_id': f"CommitPack/{commit_example['commit'][:8]}",
        'prompt': commit_example.get('subject', 'Fix code issues'),
        'buggy_code': commit_example['old_contents'],
        'fixed_code': commit_example['new_contents'],
        'commit_message': commit_example.get('message', ''),
        'file_name': commit_example.get('new_file', 'unknown.py')
    }
