"""
Merge benchmark results from multiple GPU folders.

This script loads benchmark_results.json from each gpu_X folder
and merges them into a single benchmark_results_merged.json file.

Usage:
    python engine/merge_benchmark_results.py outputs/benchmark/start_0_end_125
    python engine/merge_benchmark_results.py --output /path/to/output.json benchmark_results/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def merge_results(input_dir: str, output_file: str = None):
    """
    Merge benchmark results from all benchmark_results.json files (searches recursively).

    Args:
        input_dir: Directory to search for benchmark_results.json files
        output_file: Optional path for merged output file (default: input_dir/benchmark_results_merged.json)
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find all benchmark_results.json files recursively (exclude merged files)
    results_files = sorted([f for f in input_path.rglob("benchmark_results.json") if "merged" not in f.name])

    if not results_files:
        print(f"Error: No benchmark_results.json files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(results_files)} benchmark_results.json files:")
    for f in results_files:
        print(f"  - {f.relative_to(input_path)}")

    # Load and merge results from each file (deduplicate by question_id)
    all_results = []
    seen_qids = {}  # qid -> file where it was first seen
    duplicates = 0
    duplicate_details = []  # List of (qid, file) tuples for duplicates

    for results_file in results_files:
        try:
            with open(results_file, "r") as f:
                file_results = json.load(f)

            # Add only unique question_ids
            new_count = 0
            file_duplicates = 0
            for r in file_results:
                qid = r.get("question_id")
                if qid not in seen_qids:
                    seen_qids[qid] = str(results_file.relative_to(input_path))
                    all_results.append(r)
                    new_count += 1
                else:
                    duplicates += 1
                    file_duplicates += 1
                    duplicate_details.append({
                        "question_id": qid,
                        "duplicate_file": str(results_file.relative_to(input_path)),
                        "original_file": seen_qids[qid]
                    })

            dup_msg = f" ({file_duplicates} duplicates)" if file_duplicates > 0 else ""
            print(f"  {results_file.relative_to(input_path)}: {new_count} new results ({len(file_results)} total){dup_msg}")
        except Exception as e:
            print(f"  Error loading {results_file}: {e}")

    if not all_results:
        print("Error: No results found to merge")
        sys.exit(1)

    print(f"\nTotal merged results: {len(all_results)}")
    print(f"Duplicate entries skipped: {duplicates}")
    
    if duplicate_details:
        print(f"\n⚠️  Duplicate question_ids found:")
        for dup in duplicate_details[:20]:  # Show first 20
            print(f"    ID {dup['question_id']}: in {dup['duplicate_file']} (already in {dup['original_file']})")
        if len(duplicate_details) > 20:
            print(f"    ... and {len(duplicate_details) - 20} more")

    # Sort by question_id if available
    try:
        all_results.sort(key=lambda x: x.get("question_id", 0))
        print("Results sorted by question_id")
    except Exception as e:
        print(f"Could not sort results: {e}")

    # Determine output file path
    if output_file:
        merged_file = Path(output_file)
        # Create parent directory if it doesn't exist
        merged_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        merged_file = input_path / "benchmark_results_merged.json"
    
    with open(merged_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nMerged results saved to: {merged_file}")

    # Generate detailed report (save alongside merged file)
    generate_detailed_report(all_results, str(merged_file.parent))


def generate_detailed_report(results: list, output_dir: str):
    """
    Generate comprehensive benchmark report with detailed statistics.
    
    Args:
        results: List of result dictionaries from all questions
        output_dir: Directory to save report
    """
    import statistics
    
    print("\n" + "="*100)
    print(" " * 30 + "DETAILED BENCHMARK REPORT")
    print("="*100)
    
    # Overall Statistics
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    overall_acc = (correct / total * 100) if total > 0 else 0
    
    print(f"\n📊 OVERALL STATISTICS")
    print("-"*100)
    print(f"  Total Questions:     {total}")
    print(f"  Correct:             {correct} ({overall_acc:.2f}%)")
    print(f"  Incorrect:           {total - correct} ({100 - overall_acc:.2f}%)")
    
    # Execution Time Statistics
    all_exec_times = []
    for r in results:
        if "execution_times" in r:
            all_exec_times.extend(r["execution_times"])
    
    if all_exec_times:
        print(f"\nEXECUTION TIME STATISTICS (All SQL Queries)")
        print("-"*100)
        print(f"  Total Queries Executed:  {len(all_exec_times)}")
        print(f"  Mean Execution Time:     {statistics.mean(all_exec_times)*1000:.2f} ms")
        print(f"  Median Execution Time:   {statistics.median(all_exec_times)*1000:.2f} ms")
        print(f"  Std Dev:                 {statistics.stdev(all_exec_times)*1000:.2f} ms" if len(all_exec_times) > 1 else "  Std Dev: N/A")
        print(f"  Min Execution Time:      {min(all_exec_times)*1000:.2f} ms")
        print(f"  Max Execution Time:      {max(all_exec_times)*1000:.2f} ms")
    
    # Generation Statistics
    total_generated = sum(r.get("metrics", {}).get("generated", 0) for r in results)
    total_valid = sum(r.get("metrics", {}).get("valid_generations", 0) for r in results)
    total_errors = sum(r.get("metrics", {}).get("execution_errors", 0) for r in results)
    unique_sqls = sum(r.get("metrics", {}).get("unique_valid_sqls", 0) for r in results)
    
    print(f"\n🔧 GENERATION STATISTICS")
    print("-"*100)
    print(f"  Total SQL Generated:     {total_generated}")
    if total_generated > 0:
        print(f"  Valid Executions:        {total_valid} ({total_valid/total_generated*100:.1f}% success rate)")
        print(f"  Execution Errors:        {total_errors} ({total_errors/total_generated*100:.1f}%)")
        print(f"  Unique SQL Variations:   {unique_sqls} (avg {unique_sqls/total:.1f} per question)")
    else:
        print(f"  No SQL generated (check for errors)")
    
    # Confidence Statistics
    confidences = [r.get("winner_confidence", 0) for r in results]
    if confidences:
        print(f"\n📈 CONFIDENCE STATISTICS (Majority Voting)")
        print("-"*100)
        print(f"  Mean Confidence:         {statistics.mean(confidences):.3f}")
        print(f"  Median Confidence:       {statistics.median(confidences):.3f}")
        print(f"  High Confidence (>0.5):  {sum(1 for c in confidences if c > 0.5)} ({sum(1 for c in confidences if c > 0.5)/len(confidences)*100:.1f}%)")
        print(f"  Low Confidence (<0.2):   {sum(1 for c in confidences if c < 0.2)} ({sum(1 for c in confidences if c < 0.2)/len(confidences)*100:.1f}%)")
    
    # Difficulty Breakdown
    print(f"\n📚 ACCURACY BY DIFFICULTY")
    print("-"*100)
    difficulty_stats = {}
    for diff in ["simple", "moderate", "challenging", "unknown"]:
        diff_results = [r for r in results if r.get("difficulty") == diff]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r.get("is_correct", False))
            diff_total = len(diff_results)
            diff_acc = (diff_correct / diff_total * 100) if diff_total > 0 else 0
            difficulty_stats[diff] = {
                "correct": diff_correct,
                "total": diff_total,
                "accuracy": diff_acc
            }
            print(f"  {diff.capitalize():<15} {diff_correct:>3}/{diff_total:<3} ({diff_acc:>6.2f}%)")
    
    # Per-Database Breakdown
    print(f"\n🗄️  ACCURACY BY DATABASE")
    print("-"*100)
    db_results = {}
    for r in results:
        db_id = r.get("db_id", "unknown")
        if db_id not in db_results:
            db_results[db_id] = []
        db_results[db_id].append(r)
    
    # Sort databases by accuracy (descending)
    db_accuracy = []
    for db_id, db_qs in db_results.items():
        db_correct = sum(1 for r in db_qs if r.get("is_correct", False))
        db_total = len(db_qs)
        db_acc = (db_correct / db_total * 100) if db_total > 0 else 0
        db_accuracy.append((db_id, db_acc, db_correct, db_total))
    
    db_accuracy.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  {'Database':<40} {'Correct':>10} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-"*70)
    for db_id, acc, correct_count, total_count in db_accuracy:
        db_name = db_id.split('/')[-1] if '/' in db_id else db_id
        print(f"  {db_name:<40} {correct_count:>10} {total_count:>8} {acc:>9.2f}%")
    
    # Selection Phase Statistics
    selection_made = sum(1 for r in results if r.get("selection", {}).get("selected_sql") is not None)
    print(f"\n🎯 SQL SELECTION PHASE STATISTICS")
    print("-"*100)
    if total > 0:
        print(f"  Selection Made:          {selection_made} ({selection_made/total*100:.1f}%)")
        print(f"  Majority Vote Used:      {total - selection_made} ({(total - selection_made)/total*100:.1f}%)")
    else:
        print(f"  No results to report")
    
    # Save detailed report to file
    report_data = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": overall_acc
        },
        "by_difficulty": difficulty_stats,
        "by_database": {},
        "execution_times": {
            "mean_ms": statistics.mean(all_exec_times)*1000 if all_exec_times else 0,
            "median_ms": statistics.median(all_exec_times)*1000 if all_exec_times else 0,
            "min_ms": min(all_exec_times)*1000 if all_exec_times else 0,
            "max_ms": max(all_exec_times)*1000 if all_exec_times else 0,
            "std_ms": statistics.stdev(all_exec_times)*1000 if len(all_exec_times) > 1 else 0
        },
        "generation": {
            "total_generated": total_generated,
            "valid_executions": total_valid,
            "errors": total_errors,
            "unique_sqls": unique_sqls
        },
        "confidence": {
            "mean": statistics.mean(confidences) if confidences else 0,
            "median": statistics.median(confidences) if confidences else 0,
            "high_confidence_ratio": sum(1 for c in confidences if c > 0.5)/len(confidences) if confidences else 0
        }
    }
    
    # Add per-database stats
    for db_id, db_qs in db_results.items():
        db_correct = sum(1 for r in db_qs if r.get("is_correct", False))
        db_total = len(db_qs)
        db_acc = (db_correct / db_total * 100) if db_total > 0 else 0
        report_data["by_database"][db_id] = {
            "total": db_total,
            "correct": db_correct,
            "accuracy": db_acc
        }
    
    report_file = os.path.join(output_dir, "detailed_report.json")
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    print("="*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge benchmark results from multiple GPU folders"
    )
    parser.add_argument(
        "input_dir",
        help="Directory to search for benchmark_results.json files"
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_file",
        default=None,
        help="Output file path for merged results (default: <input_dir>/benchmark_results_merged.json)"
    )
    
    args = parser.parse_args()
    
    merge_results(args.input_dir, args.output_file)
