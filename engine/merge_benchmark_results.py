"""
Merge benchmark results from multiple GPU folders.

This script loads benchmark_results.json from each gpu_X folder
and merges them into a single benchmark_results_merged.json file.

Usage:
    python engine/merge_benchmark_results.py outputs/benchmark/start_0_end_125
"""

import json
import os
import sys
from pathlib import Path


def merge_results(output_dir: str):
    """
    Merge benchmark results from all gpu_X folders.
    
    Args:
        output_dir: Directory containing gpu_X subdirectories
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    # Find all gpu_X directories
    gpu_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("gpu_")])
    
    if not gpu_dirs:
        print(f"Error: No gpu_X directories found in {output_dir}")
        sys.exit(1)
    
    print(f"Found {len(gpu_dirs)} GPU directories: {[d.name for d in gpu_dirs]}")
    
    # Load and merge results from each GPU
    all_results = []
    
    for gpu_dir in gpu_dirs:
        results_file = gpu_dir / "benchmark_results.json"
        
        if not results_file.exists():
            print(f"  Warning: {results_file} not found, skipping...")
            continue
        
        with open(results_file, "r") as f:
            gpu_results = json.load(f)
        
        print(f"  {gpu_dir.name}: {len(gpu_results)} results")
        all_results.extend(gpu_results)
    
    if not all_results:
        print("Error: No results found to merge")
        sys.exit(1)
    
    print(f"\nTotal merged results: {len(all_results)}")
    
    # Sort by question_id if available
    try:
        all_results.sort(key=lambda x: x.get("question_id", 0))
        print("Results sorted by question_id")
    except Exception as e:
        print(f"Could not sort results: {e}")
    
    # Save merged results
    merged_file = output_path / "benchmark_results_merged.json"
    with open(merged_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nMerged results saved to: {merged_file}")
    
    # Generate detailed report
    generate_detailed_report(all_results, str(output_path))


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
    if len(sys.argv) < 2:
        print("Usage: python merge_benchmark_results.py <output_directory>")
        print("\nExample:")
        print("  python merge_benchmark_results.py outputs/benchmark/start_0_end_125")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    merge_results(output_dir)
