"""
Analyze duplicate SQL generations in benchmark results.

This script finds all benchmark_results*.json files and analyzes:
1. How many duplicate SQLs are generated per question
2. How many unique SQLs vs total generations
3. Potential compute savings from deduplication

Usage:
    python engine/analyze_duplicates.py benchmark_results/
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_all_results(root_dir: str):
    """Load all benchmark_results*.json files recursively."""
    root_path = Path(root_dir)
    results_files = list(root_path.rglob("benchmark_results*.json"))
    
    # Exclude merged files
    results_files = [f for f in results_files if "merged" not in f.name]
    
    print(f"Found {len(results_files)} result files:")
    for f in sorted(results_files):
        print(f"  - {f.relative_to(root_path)}")
    
    all_results = []
    for results_file in sorted(results_files):
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    print(f"  Warning: {results_file} has unexpected format")
        except Exception as e:
            print(f"  Error loading {results_file}: {e}")
    
    print(f"\nTotal questions loaded: {len(all_results)}")
    return all_results


def analyze_duplicates(results: list):
    """Analyze duplicate patterns in benchmark results."""
    print("\n" + "="*80)
    print(" " * 25 + "DUPLICATE ANALYSIS REPORT")
    print("="*80)
    
    # Track metrics across all questions
    total_questions = len(results)
    questions_with_results = 0
    
    duplicate_stats = []
    unique_sql_ratios = []
    execution_duplicate_stats = []
    
    for q in results:
        # Check if this question has generation data
        metrics = q.get("metrics", {})
        generated = metrics.get("generated", 0)
        valid = metrics.get("valid_generations", 0)
        unique_sqls = metrics.get("unique_valid_sqls", 0)
        
        if generated > 0:
            questions_with_results += 1
            
            # Calculate duplicate rate
            if generated > 0:
                duplicate_rate = 1 - (unique_sqls / generated)
                duplicate_stats.append(duplicate_rate)
                unique_sql_ratios.append(unique_sqls / generated)
            
            # Track execution duplicates (from execution_times)
            exec_times = q.get("execution_times", [])
            if len(exec_times) > 0 and unique_sqls > 0:
                exec_duplicate_rate = 1 - (unique_sqls / len(exec_times))
                execution_duplicate_stats.append(exec_duplicate_rate)
    
    print(f"\n📊 GENERATION STATISTICS")
    print("-"*80)
    print(f"  Questions with generation data: {questions_with_results}/{total_questions}")
    
    if duplicate_stats:
        avg_duplicate_rate = sum(duplicate_stats) / len(duplicate_stats) * 100
        avg_unique_ratio = sum(unique_sql_ratios) / len(unique_sql_ratios) * 100
        
        print(f"  Average duplicate rate: {avg_duplicate_rate:.1f}%")
        print(f"  Average unique SQL ratio: {avg_unique_ratio:.1f}%")
        print(f"  ")
        print(f"  Interpretation:")
        print(f"    - {avg_duplicate_rate:.1f}% of generated SQLs are duplicates")
        print(f"    - Deduplication before execution could save ~{avg_duplicate_rate:.1f}% of execution time")
        
        # Breakdown by duplicate rate buckets
        low_dup = sum(1 for d in duplicate_stats if d < 0.3)
        med_dup = sum(1 for d in duplicate_stats if 0.3 <= d < 0.6)
        high_dup = sum(1 for d in duplicate_stats if d >= 0.6)
        
        print(f"\n  Duplicate Rate Distribution:")
        print(f"    Low (<30%):     {low_dup:>4} questions ({low_dup/len(duplicate_stats)*100:.1f}%)")
        print(f"    Medium (30-60%): {med_dup:>4} questions ({med_dup/len(duplicate_stats)*100:.1f}%)")
        print(f"    High (>60%):    {high_dup:>4} questions ({high_dup/len(duplicate_stats)*100:.1f}%)")
    
    print(f"\n🔧 EXECUTION STATISTICS")
    print("-"*80)
    if execution_duplicate_stats:
        avg_exec_dup = sum(execution_duplicate_stats) / len(execution_duplicate_stats) * 100
        print(f"  Average execution duplicate rate: {avg_exec_dup:.1f}%")
        print(f"  (This measures how many executions were duplicates)")
    
    # Analyze by difficulty
    print(f"\n📚 BREAKDOWN BY DIFFICULTY")
    print("-"*80)
    
    difficulty_stats = defaultdict(list)
    for q in results:
        diff = q.get("difficulty", "unknown")
        metrics = q.get("metrics", {})
        generated = metrics.get("generated", 0)
        unique_sqls = metrics.get("unique_valid_sqls", 0)
        
        if generated > 0 and unique_sqls > 0:
            unique_ratio = unique_sqls / generated
            difficulty_stats[diff].append(unique_ratio)
    
    for diff in ["simple", "moderate", "challenging", "unknown"]:
        if diff in difficulty_stats and difficulty_stats[diff]:
            ratios = difficulty_stats[diff]
            avg_ratio = sum(ratios) / len(ratios) * 100
            print(f"  {diff.capitalize():<15} {avg_ratio:>6.1f}% unique (avg)")
    
    # Analyze by database
    print(f"\n🗄️  BREAKDOWN BY DATABASE (Top 10)")
    print("-"*80)
    
    db_stats = defaultdict(list)
    for q in results:
        db_id = q.get("db_id", "unknown")
        metrics = q.get("metrics", {})
        generated = metrics.get("generated", 0)
        unique_sqls = metrics.get("unique_valid_sqls", 0)
        
        if generated > 0 and unique_sqls > 0:
            unique_ratio = unique_sqls / generated
            db_stats[db_id].append(unique_ratio)
    
    db_averages = []
    for db_id, ratios in db_stats.items():
        avg_ratio = sum(ratios) / len(ratios) * 100
        db_name = db_id.split('/')[-1] if '/' in db_id else db_id
        db_averages.append((db_name, avg_ratio, len(ratios)))
    
    # Sort by number of questions (descending)
    db_averages.sort(key=lambda x: x[2], reverse=True)
    
    print(f"  {'Database':<30} {'Unique %':>10} {'Questions':>10}")
    print("  " + "-"*52)
    for db_name, avg_ratio, count in db_averages[:10]:
        print(f"  {db_name:<30} {avg_ratio:>9.1f}% {count:>10}")
    
    # Potential savings
    print(f"\n💾 POTENTIAL COMPUTE SAVINGS")
    print("-"*80)
    
    if duplicate_stats:
        total_generated = sum(q.get("metrics", {}).get("generated", 0) for q in results)
        total_unique = sum(q.get("metrics", {}).get("unique_valid_sqls", 0) for q in results)
        
        print(f"  Total SQL generated:     {total_generated:,}")
        print(f"  Total unique SQL:        {total_unique:,}")
        print(f"  Duplicate executions:    {total_generated - total_unique:,}")
        print(f"  ")
        print(f"  If we deduplicate BEFORE execution:")
        print(f"    - Executions saved:    {total_generated - total_unique:,} ({(1 - total_unique/total_generated)*100:.1f}%)")
        print(f"    - New execution count: {total_unique:,}")
        
        # Estimate time savings (assuming execution is ~50% of total time)
        exec_savings = (1 - total_unique/total_generated) * 50  # % of total time
        print(f"  ")
        print(f"  Estimated total time savings: ~{exec_savings:.1f}%")
        print(f"  (Assuming execution is ~50% of total pipeline time)")
    
    print("\n" + "="*80)


def analyze_selection_impact(results: list):
    """Analyze how often selection phase changes the result."""
    print("\n" + "="*80)
    print(" " * 25 + "SELECTION PHASE IMPACT")
    print("="*80)
    
    selection_made = 0
    selection_changed = 0
    selection_correct_changes = 0
    
    for q in results:
        selection = q.get("selection", {})
        selected_sql = selection.get("selected_sql")
        winner_sql = q.get("winner_sql")
        is_correct = q.get("is_correct", False)
        
        if selected_sql is not None:
            selection_made += 1
            
            # Check if selection changed the result
            if winner_sql and selected_sql != winner_sql:
                selection_changed += 1
                
                # Check if the change improved correctness
                # (would need to re-execute to know for sure, so we skip this)
    
    if selection_made > 0:
        print(f"\n  Selection phase ran:     {selection_made}/{len(results)} questions ({selection_made/len(results)*100:.1f}%)")
        print(f"  Selection changed result: {selection_changed}/{selection_made} ({selection_changed/selection_made*100:.1f}%)")
        print(f"  ")
        print(f"  Interpretation:")
        if selection_changed / selection_made < 0.1:
            print(f"    - Selection rarely changes the result (<10%)")
            print(f"    - Consider skipping selection phase to save compute")
        else:
            print(f"    - Selection changes result {selection_changed/selection_made*100:.1f}% of the time")
            print(f"    - Selection phase may be valuable")
    
    print("\n" + "="*80)


def analyze_confidence_distribution(results: list):
    """Analyze confidence scores to find early stopping opportunities."""
    print("\n" + "="*80)
    print(" " * 25 + "CONFIDENCE DISTRIBUTION (Early Stopping Analysis)")
    print("="*80)
    
    # We can't analyze this directly from results, but we can check
    # final confidence scores to see if they're typically very high
    confidences = [q.get("winner_confidence", 0) for q in results]
    
    if confidences:
        print(f"\n  Final Confidence Distribution:")
        print(f"  ")
        
        very_high = sum(1 for c in confidences if c >= 0.8)
        high = sum(1 for c in confidences if 0.6 <= c < 0.8)
        medium = sum(1 for c in confidences if 0.4 <= c < 0.6)
        low = sum(1 for c in confidences if c < 0.4)
        
        total = len(confidences)
        
        print(f"    Very High (≥0.8):  {very_high:>5} ({very_high/total*100:>5.1f}%)")
        print(f"    High (0.6-0.8):    {high:>5} ({high/total*100:>5.1f}%)")
        print(f"    Medium (0.4-0.6):  {medium:>5} ({medium/total*100:>5.1f}%)")
        print(f"    Low (<0.4):        {low:>5} ({low/total*100:>5.1f}%)")
        print(f"  ")
        print(f"  Mean confidence:   {sum(confidences)/len(confidences):.3f}")
        print(f"  Median confidence: {sorted(confidences)[len(confidences)//2]:.3f}")
        print(f"  ")
        print(f"  Interpretation:")
        if very_high / total > 0.5:
            print(f"    - >50% questions have very high confidence (≥0.8)")
            print(f"    - Early stopping could work well for these")
            print(f"    - Generate in batches, stop when confidence ≥0.8")
        else:
            print(f"    - Confidence is more distributed")
            print(f"    - Early stopping may only help for {very_high/total*100:.1f}% of questions")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_duplicates.py <benchmark_results_directory>")
        print("\nExample:")
        print("  python analyze_duplicates.py benchmark_results/")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        sys.exit(1)
    
    results = load_all_results(root_dir)
    
    if not results:
        print("No results found to analyze")
        sys.exit(1)
    
    analyze_duplicates(results)
    analyze_selection_impact(results)
    analyze_confidence_distribution(results)
    
    print("\n✅ Analysis complete!")
