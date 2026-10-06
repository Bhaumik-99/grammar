#!/usr/bin/env python3
"""Automated Kaggle GPU Pipeline Orchestrator for SHL Grammar Scoring 2026.

This script manages sequential and parallel Kaggle kernel execution respecting:
1. Kaggle's maximum quota of 2 concurrent GPU sessions.
2. The pipeline dependency DAG:
   - Group A (Independent): 01, 02, 03, 04, 05
   - Group B (Depends on A): 06 (needs 01, 03), 07 (needs 01-03), 08 (needs 01-03)
   - Final Submission: notebook/ (needs all 01-08)

Usage:
    python run_pipeline_kaggle.py status        # Show status of all stages
    python run_pipeline_kaggle.py run           # Auto-queue and push stages as GPU slots open up
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Pipeline stages definition
STAGES = [
    {
        "id": "01",
        "name": "shl-2026-eda-asr",
        "dir": "pipeline/01_eda_asr",
        "deps": [],
        "gpu": True,
        "description": "Audio EDA, silence/pause stats, faster-whisper transcripts",
    },
    {
        "id": "02",
        "name": "shl-2026-asr-verbatim-views",
        "dir": "pipeline/02_asr_verbatim_views",
        "deps": [],
        "gpu": True,
        "description": "Disfluent Whisper prompt & Parakeet CTC error-preserving views",
    },
    {
        "id": "03",
        "name": "shl-2026-audio-embeddings",
        "dir": "pipeline/03_audio_embeddings",
        "deps": [],
        "gpu": True,
        "description": "Whisper-v3 encoder, WavLM-large, w2v-BERT layer embeddings",
    },
    {
        "id": "04",
        "name": "shl-2026-audio-llm-embeddings",
        "dir": "pipeline/04_audio_llm_embeddings",
        "deps": [],
        "gpu": True,
        "description": "Voxtral-Mini-3B audio-token embeddings, HuBERT-large",
    },
    {
        "id": "05",
        "name": "shl-2026-qwen2-audio-embeddings",
        "dir": "pipeline/05_qwen2_audio_embeddings",
        "deps": [],
        "gpu": True,
        "description": "Qwen2-Audio-7B frozen state audio features",
    },
    {
        "id": "06",
        "name": "shl-2026-text-llm-features",
        "dir": "pipeline/06_text_llm_features",
        "deps": ["01", "03"],
        "gpu": True,
        "description": "Qwen3-8B hidden states, LLM judge, perplexity, GEC, Qwen3-Embed",
    },
    {
        "id": "07",
        "name": "shl-2026-text-llm-features-views",
        "dir": "pipeline/07_text_llm_features_views",
        "deps": ["01", "02", "03"],
        "gpu": True,
        "description": "LLM text features on verbatim/error-preserving ASR views",
    },
    {
        "id": "08",
        "name": "shl-2026-text-deberta",
        "dir": "pipeline/08_text_deberta",
        "deps": ["01", "02", "03"],
        "gpu": True,
        "description": "DeBERTa-v3-large cross-validated regressor on clean + CTC",
    },
    {
        "id": "final",
        "name": "shl-2026-grammar-scoring-engine",
        "dir": "notebook",
        "deps": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "gpu": False,
        "description": "Full tabular + multi-modal feature stacking & submission.csv",
    },
]


def get_username() -> str:
    meta_path = Path("pipeline/01_eda_asr/kernel-metadata.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            kid = d.get("id", "")
            if "/" in kid:
                return kid.split("/")[0]
    import kaggle
    return kaggle.api.get_config_value("username")


def get_kernel_status(username: str, slug: str):
    import kaggle
    kernel_id = f"{username}/{slug}"
    try:
        res = kaggle.api.kernels_status(kernel_id)
        status_str = str(res.status)
        if "RUNNING" in status_str:
            return "RUNNING", None
        elif "COMPLETE" in status_str:
            return "COMPLETE", None
        elif "ERROR" in status_str:
            return "ERROR", res.failure_message
        elif "QUEUED" in status_str:
            return "QUEUED", None
        else:
            return status_str, None
    except Exception as e:
        err = str(e)
        if "404" in err or "Permission 'kernels.get' was denied" in err or "not found" in err.lower():
            return "NOT_STARTED", None
        return "UNKNOWN", err


def push_kernel(dir_path: str):
    print(f"\n[>>>] Pushing kernel from {dir_path} ...", flush=True)
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", dir_path],
        capture_output=True,
        text=True,
    )
    print(res.stdout.strip(), flush=True)
    if res.stderr.strip():
        print(f"[STDERR] {res.stderr.strip()}", flush=True)
    return res.returncode == 0 and "Kernel version" in res.stdout


def show_status(username: str):
    header_sep = "=" * 68
    print(f"\n{header_sep}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] KAGGLE PIPELINE STATUS (User: {username})", flush=True)
    print(header_sep, flush=True)
    print(f"{'ID':<4} {'Stage Name':<28} {'GPU':<4} {'Status':<13} {'Deps'}", flush=True)
    print("-" * 68, flush=True)
    
    status_map = {}
    running_gpu = 0
    for s in STAGES:
        st, msg = get_kernel_status(username, s["name"])
        status_map[s["id"]] = st
        if st in ("RUNNING", "QUEUED") and s["gpu"]:
            running_gpu += 1
        
        status_disp = st
        if st == "RUNNING":
            status_disp = "[RUNNING]"
        elif st == "COMPLETE":
            status_disp = "COMPLETE"
        elif st == "ERROR":
            status_disp = "FAILED"
        elif st == "NOT_STARTED":
            status_disp = "NOT_STARTED"
            
        deps_str = ",".join(s["deps"]) if s["deps"] else "-"
        # Shorten stage name for compact display if needed
        short_name = s["name"].replace("shl-2026-", "")
        gpu_str = "Yes" if s["gpu"] else "No"
        print(f"{s['id']:<4} {short_name:<28} {gpu_str:<4} {status_disp:<13} {deps_str}", flush=True)
        if msg:
            print(f"     -> {msg[:60]}...", flush=True)
            
    print("-" * 68, flush=True)
    print(f"Active GPU sessions: {running_gpu} / 2 (Kaggle limit)", flush=True)
    print(f"{header_sep}\n", flush=True)
    return status_map, running_gpu


def run_orchestrator(username: str, poll_interval: int = 45):
    print(f"\n[*] Starting Automated Pipeline Orchestrator for {username}")
    print(f"[*] Polling interval: {poll_interval}s")
    
    while True:
        status_map, running_gpu = show_status(username)
        
        # Check if all completed or final notebook is completed
        final_complete = status_map.get("final") == "COMPLETE"
        all_complete = all(status_map.get(s["id"]) == "COMPLETE" for s in STAGES)
        if all_complete or final_complete:
            print("\n[SUCCESS] FINAL SUBMISSION NOTEBOOK COMPLETED ON KAGGLE!", flush=True)
            print("[*] Downloading submission.csv and outputs to local directory...", flush=True)
            download_submission(username)
            break
            
        # Check if any failed
        any_failed = [s["name"] for s in STAGES if status_map.get(s["id"]) == "ERROR"]
        if any_failed:
            print(f"\n[ALERT] The following kernels encountered an error: {any_failed}", flush=True)
            print("[ALERT] Please inspect logs using: kaggle kernels output <username>/<kernel-slug>", flush=True)
            # We don't abort immediately because other independent kernels may still be progressing
            
        # Check if we can schedule a new kernel
        available_gpu_slots = 2 - running_gpu
        pushed_any = False
        
        for s in STAGES:
            st = status_map.get(s["id"])
            if st != "NOT_STARTED":
                continue  # already started or finished
                
            # Check dependencies
            deps_met = all(status_map.get(dep_id) == "COMPLETE" for dep_id in s["deps"])
            if not deps_met:
                continue  # wait for dependencies
                
            # Check GPU availability
            if s["gpu"] and available_gpu_slots <= 0:
                continue  # no GPU slot free
                
            # Can run this stage!
            print(f"\n[+] Dependencies met for Stage {s['id']} ({s['name']}). Triggering now...")
            success = push_kernel(s["dir"])
            if success:
                pushed_any = True
                if s["gpu"]:
                    available_gpu_slots -= 1
            else:
                print(f"[!] Push failed for {s['name']}. Will retry in next cycle.")
                
            if available_gpu_slots <= 0:
                break
                
        if not pushed_any:
            print(f"[*] In progress. Next check in {poll_interval} seconds... (Ctrl+C to stop monitor)")
            time.sleep(poll_interval)
        else:
            time.sleep(5)


def download_submission(username: str, output_dir: str = "."):
    """Download submission.csv and kernel artifacts locally."""
    kernel_id = f"{username}/shl-2026-grammar-scoring-engine"
    print(f"\n[*] Fetching outputs from Kaggle kernel: {kernel_id} ...", flush=True)
    res = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", output_dir],
        capture_output=True,
        text=True,
    )
    print(res.stdout.strip(), flush=True)
    if res.stderr.strip():
        print(f"[STDERR] {res.stderr.strip()}", flush=True)
        
    sub_path = Path(output_dir) / "submission.csv"
    if sub_path.exists():
        print(f"\n[SUCCESS] submission.csv successfully saved locally to: {sub_path.resolve()}", flush=True)
        try:
            import pandas as pd
            df = pd.read_csv(sub_path)
            print(f"Shape: {df.shape}")
            print("Preview:")
            print(df.head(5).to_string())
        except Exception:
            pass
    else:
        print(f"[!] Warning: submission.csv not found in downloaded output folder yet.")


def main():
    parser = argparse.ArgumentParser(description="Kaggle SHL Pipeline Runner")
    parser.add_argument("action", choices=["status", "run", "push", "download"], nargs="?", default="status")
    parser.add_argument("--stage", help="Stage ID or directory to push (for 'push' action)")
    parser.add_argument("--interval", type=int, default=45, help="Poll interval in seconds")
    args = parser.parse_args()

    username = get_username()
    
    if args.action == "status":
        show_status(username)
    elif args.action == "push":
        if not args.stage:
            print("Please specify --stage <id> (e.g. 01, 02, etc.)")
            sys.exit(1)
        matched = [s for s in STAGES if s["id"] == args.stage or s["dir"] == args.stage]
        if not matched:
            print(f"Stage '{args.stage}' not found.")
            sys.exit(1)
        push_kernel(matched[0]["dir"])
    elif args.action == "run":
        run_orchestrator(username, poll_interval=args.interval)
    elif args.action == "download":
        download_submission(username)


if __name__ == "__main__":
    main()
