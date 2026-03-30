# pipeline_runner.py
# YOUR OWN CODE — Full Pipeline Runner
# Connects all modules and processes NCERT dataset
# Calculates evaluation metrics using YOUR OWN formulas

import json
import os
import sys
import time
import re
sys.path.append("modules")

from simplification import TextSimplifier
from summarization   import TextSummarizer
from preprocessing   import TextPreprocessor


# ════════════════════════════════════════════════════════════════
# YOUR OWN EVALUATION METRICS — No external library used
# ════════════════════════════════════════════════════════════════

def count_syllables(word):
    """
    YOUR OWN syllable counter.
    Counts vowel groups in a word as syllables.
    Used inside Flesch-Kincaid Grade calculation.
    """
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e'):
        count -= 1
    return max(1, count)


def flesch_kincaid_grade(text):
    """
    YOUR OWN Flesch-Kincaid Grade Level implementation.
    Formula: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
    Lower score = easier to read.
    Grade 6 = easy, Grade 12 = very hard.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    num_sentences = max(1, len(sentences))
    words = text.split()
    num_words = max(1, len(words))
    num_syllables = sum(count_syllables(w) for w in words)
    score = (0.39 * (num_words / num_sentences) +
             11.8 * (num_syllables / num_words) - 15.59)
    return round(score, 2)


def rouge1_score(reference, hypothesis):
    """
    YOUR OWN ROUGE-1 F1 Score implementation.
    Measures word overlap between original and output.
    Precision = overlap/hypothesis words
    Recall    = overlap/reference words
    F1        = harmonic mean of precision and recall
    """
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    if not ref_words or not hyp_words:
        return 0.0
    overlap   = ref_words.intersection(hyp_words)
    precision = len(overlap) / len(hyp_words)
    recall    = len(overlap) / len(ref_words)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def compression_ratio(original, compressed):
    """
    YOUR OWN compression ratio calculator.
    Measures how much shorter the output is vs input.
    """
    orig_words = max(1, len(original.split()))
    comp_words = len(compressed.split())
    return round((1 - comp_words / orig_words) * 100, 1)


def readability_label(score):
    if score <= 6:  return "Grade 6 (Easy)"
    if score <= 8:  return "Grade 8 (Moderate)"
    if score <= 10: return "Grade 10 (Difficult)"
    return "Undergraduate (Very Difficult)"


# ════════════════════════════════════════════════════════════════
# PROCESS ONE PARAGRAPH
# ════════════════════════════════════════════════════════════════

def process_paragraph(para, preprocessor, simplifier,
                       summarizer, para_num):
    result = {
        "paragraph_num" : para_num,
        "original_text" : para,
        "original_words": len(para.split()),
        "original_fk"   : flesch_kincaid_grade(para),
    }

    # Preprocess
    clean_para = preprocessor.preprocess(para)

    # Simplification
    try:
        simplified = simplifier.simplify(clean_para, level="Grade 8")
        result["simplified_text"]  = simplified
        result["simplified_words"] = len(simplified.split())
        result["simplified_fk"]    = flesch_kincaid_grade(simplified)
        result["simp_compression"] = compression_ratio(para, simplified)
        result["simp_rouge1"]      = rouge1_score(para, simplified)
        result["simp_status"]      = "success"
    except Exception as e:
        result["simplified_text"]  = para
        result["simplified_words"] = len(para.split())
        result["simplified_fk"]    = result["original_fk"]
        result["simp_compression"] = 0.0
        result["simp_rouge1"]      = 0.0
        result["simp_status"]      = f"error: {str(e)[:50]}"

    # Summarization
    try:
        summary = summarizer.summarize(clean_para)
        result["summary_text"]     = summary
        result["summary_words"]    = len(summary.split())
        result["summary_fk"]       = flesch_kincaid_grade(summary)
        result["summ_compression"] = compression_ratio(para, summary)
        result["summ_rouge1"]      = rouge1_score(para, summary)
        result["summ_status"]      = "success"
    except Exception as e:
        result["summary_text"]     = ""
        result["summary_words"]    = 0
        result["summary_fk"]       = 0.0
        result["summ_compression"] = 0.0
        result["summ_rouge1"]      = 0.0
        result["summ_status"]      = f"error: {str(e)[:50]}"

    return result


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNER
# ════════════════════════════════════════════════════════════════

def run_pipeline(
    dataset_path         = "data/ncert_dataset.json",
    results_path         = "data/pipeline_results.json",
    report_path          = "data/pipeline_report.txt",
    max_para_per_chapter = 3
):
    print("Loading dataset...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Dataset: {dataset['total_chapters']} chapters, "
          f"{dataset['total_paragraphs']} paragraphs, "
          f"{dataset['total_words']} words")

    print("\nLoading models...")
    preprocessor = TextPreprocessor()
    simplifier   = TextSimplifier()
    summarizer   = TextSummarizer()
    print("All modules ready. Starting pipeline...\n")

    results = {
        "source"   : dataset["source"],
        "chapters" : []
    }

    total_start = time.time()

    for chapter in dataset["chapters"]:
        ch_num   = chapter["chapter_number"]
        ch_title = chapter["chapter_title"]
        paras    = chapter["paragraphs"]

        if not paras:
            print(f"Chapter {ch_num}: No paragraphs — skipping")
            continue

        if max_para_per_chapter:
            paras = paras[:max_para_per_chapter]

        print(f"\n{'='*60}")
        print(f"Chapter {ch_num}: {ch_title}")
        print(f"Processing {len(paras)} paragraphs...")
        print(f"{'='*60}")

        ch_start   = time.time()
        ch_results = []

        for i, para in enumerate(paras):
            print(f"\n  Paragraph {i+1}/{len(paras)} "
                  f"({len(para.split())} words)...")
            para_result = process_paragraph(
                para, preprocessor, simplifier, summarizer, i+1)
            ch_results.append(para_result)

            print(f"    Simplification : "
                  f"{para_result['original_words']} -> "
                  f"{para_result['simplified_words']} words "
                  f"({para_result['simp_compression']}% reduction)")
            print(f"    Summarization  : "
                  f"{para_result['original_words']} -> "
                  f"{para_result['summary_words']} words "
                  f"({para_result['summ_compression']}% reduction)")
            print(f"    FK Grade       : "
                  f"{para_result['original_fk']} -> "
                  f"{para_result['simplified_fk']}")

        ch_time = round(time.time() - ch_start, 1)
        valid   = [r for r in ch_results
                   if r["simp_status"] == "success"]

        if valid:
            avg_simp_comp = round(
                sum(r["simp_compression"] for r in valid)/len(valid), 1)
            avg_summ_comp = round(
                sum(r["summ_compression"] for r in valid)/len(valid), 1)
            avg_orig_fk   = round(
                sum(r["original_fk"]   for r in valid)/len(valid), 2)
            avg_simp_fk   = round(
                sum(r["simplified_fk"] for r in valid)/len(valid), 2)
            avg_rouge1    = round(
                sum(r["simp_rouge1"]   for r in valid)/len(valid), 4)
        else:
            avg_simp_comp = avg_summ_comp = 0.0
            avg_orig_fk   = avg_simp_fk   = avg_rouge1 = 0.0

        chapter_result = {
            "chapter_number"      : ch_num,
            "chapter_title"       : ch_title,
            "paragraphs_processed": len(ch_results),
            "avg_simp_compression": avg_simp_comp,
            "avg_summ_compression": avg_summ_comp,
            "avg_original_fk"     : avg_orig_fk,
            "avg_simplified_fk"   : avg_simp_fk,
            "fk_improvement"      : round(avg_orig_fk - avg_simp_fk, 2),
            "avg_rouge1"          : avg_rouge1,
            "processing_time_s"   : ch_time,
            "paragraphs"          : ch_results
        }

        results["chapters"].append(chapter_result)
        print(f"\n  Chapter {ch_num} Average Results:")
        print(f"  Simplification : {avg_simp_comp}%")
        print(f"  Summarization  : {avg_summ_comp}%")
        print(f"  FK Improvement : {avg_orig_fk} -> "
              f"{avg_simp_fk} (-{round(avg_orig_fk-avg_simp_fk,2)})")

    total_time = round(time.time() - total_start, 1)

    chs = results["chapters"]
    if chs:
        results["overall"] = {
            "total_chapters_processed": len(chs),
            "avg_simp_compression"    : round(
                sum(c["avg_simp_compression"] for c in chs)/len(chs), 1),
            "avg_summ_compression"    : round(
                sum(c["avg_summ_compression"] for c in chs)/len(chs), 1),
            "avg_original_fk"         : round(
                sum(c["avg_original_fk"]  for c in chs)/len(chs), 2),
            "avg_simplified_fk"       : round(
                sum(c["avg_simplified_fk"] for c in chs)/len(chs), 2),
            "avg_fk_improvement"      : round(
                sum(c["fk_improvement"]   for c in chs)/len(chs), 2),
            "avg_rouge1"              : round(
                sum(c["avg_rouge1"]       for c in chs)/len(chs), 4),
            "total_processing_time_s" : total_time,
        }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    generate_report(results, report_path)
    return results


# ════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ════════════════════════════════════════════════════════════════

def generate_report(results, report_path):
    lines = []
    lines.append("=" * 65)
    lines.append("  SMART ACADEMIC TEXT PROCESSING - PIPELINE RESULTS")
    lines.append("  Dataset: NCERT Class 8 Science Textbook")
    lines.append("=" * 65)
    lines.append(
        f"\n{'Chapter':<35} {'Simp%':>6} {'Summ%':>6} "
        f"{'FK-Orig':>7} {'FK-Simp':>7} {'ROUGE1':>7}")
    lines.append("-" * 65)

    for ch in results["chapters"]:
        title = f"Ch{ch['chapter_number']}: {ch['chapter_title'][:25]}"
        lines.append(
            f"{title:<35} "
            f"{ch['avg_simp_compression']:>5}% "
            f"{ch['avg_summ_compression']:>5}% "
            f"{ch['avg_original_fk']:>7} "
            f"{ch['avg_simplified_fk']:>7} "
            f"{ch['avg_rouge1']:>7}")

    if "overall" in results:
        ov = results["overall"]
        lines.append("-" * 65)
        lines.append(
            f"{'AVERAGE':<35} "
            f"{ov['avg_simp_compression']:>5}% "
            f"{ov['avg_summ_compression']:>5}% "
            f"{ov['avg_original_fk']:>7} "
            f"{ov['avg_simplified_fk']:>7} "
            f"{ov['avg_rouge1']:>7}")
        lines.append("=" * 65)
        lines.append("\nOVERALL SUMMARY:")
        lines.append(
            f"  Chapters processed       : "
            f"{ov['total_chapters_processed']}")
        lines.append(
            f"  Avg simplification       : "
            f"{ov['avg_simp_compression']}%")
        lines.append(
            f"  Avg summarization        : "
            f"{ov['avg_summ_compression']}%")
        lines.append(
            f"  Avg FK grade (original)  : "
            f"{ov['avg_original_fk']} "
            f"({readability_label(ov['avg_original_fk'])})")
        lines.append(
            f"  Avg FK grade (simplified): "
            f"{ov['avg_simplified_fk']} "
            f"({readability_label(ov['avg_simplified_fk'])})")
        lines.append(
            f"  FK grade improvement     : "
            f"-{ov['avg_fk_improvement']} grades easier")
        lines.append(
            f"  Avg ROUGE-1 score        : "
            f"{ov['avg_rouge1']}")
        lines.append(
            f"  Total processing time    : "
            f"{ov['total_processing_time_s']}s")

    report_text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run_pipeline(
        dataset_path         = "data/ncert_dataset.json",
        results_path         = "data/pipeline_results.json",
        report_path          = "data/pipeline_report.txt",
        max_para_per_chapter = 3
    )