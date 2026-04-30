import json
import os
import re

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def parse_markdown_to_reportlab(md_text, styles):
    """A lightweight parser to convert basic markdown to ReportLab Paragraphs."""
    flowables = []
    body_style = styles["BodyText"]

    # Split by double newline to separate paragraphs and blocks
    blocks = md_text.split("\n\n")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Replace Markdown bold (**text**) with ReportLab bold (<b>text</b>)
        block = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", block)
        # Replace Markdown inline code (`text`) with ReportLab courier font
        block = re.sub(r"`(.*?)`", r'<font name="Courier">\1</font>', block)

        # Handle Headers
        if block.startswith("### "):
            flowables.append(Paragraph(block[4:], styles["Heading3"]))
        elif block.startswith("## "):
            flowables.append(Paragraph(block[3:], styles["Heading2"]))
        elif block.startswith("# "):
            flowables.append(Paragraph(block[2:], styles["Heading1"]))
        else:
            # Handle list items and normal paragraphs
            lines = block.split("\n")
            is_list = any(
                line.strip().startswith("* ") or re.match(r"^\d+\.\s", line.strip())
                for line in lines
            )

            if is_list:
                for line in lines:
                    line = line.strip()
                    if line.startswith("* "):
                        # Convert asterisk to a bullet point
                        flowables.append(Paragraph("&bull; " + line[2:], body_style))
                    else:
                        flowables.append(Paragraph(line, body_style))
            else:
                # Normal paragraph, replace single newlines with spaces to let ReportLab wrap text
                text = block.replace("\n", " ")
                flowables.append(Paragraph(text, body_style))

        flowables.append(Spacer(1, 8))

    return flowables


def generate_report(
    data,
    methodology_md="",
    article_sizes_json="",
    output_pdf_name="detailed_benchmark_report.pdf",
):
    # Extract values for plotting
    chunk_sizes = [d["chunk_size"] for d in data]
    mean_acc_lengths = [d["mean_acceptance_length"] for d in data]

    # Extract acceptance rates by position
    num_rates = len(data[0]["acceptance_rates"])
    acc_rates_by_pos = [[] for _ in range(num_rates)]

    for d in data:
        for i, rate in enumerate(d["acceptance_rates"]):
            acc_rates_by_pos[i].append(rate)

    # List to store paths of generated plot images for later cleanup
    generated_plots = []

    # --- Generate Benchmark Plots ---
    # Plot A: Chunk Size vs Mean Acceptance Length
    plt.figure(figsize=(6, 4))
    plt.plot(
        chunk_sizes, mean_acc_lengths, marker="o", color="b", linestyle="-", linewidth=2
    )
    plt.title("Chunk Size vs Mean Acceptance Length")
    plt.xlabel("Max Article Chunk Size (num tokens)")
    plt.ylabel("Mean Acceptance Length")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(chunk_sizes, rotation=45, ha="right")  # Updated here
    plot1_path = "plot_mean_acc_len.png"
    plt.savefig(plot1_path, bbox_inches="tight", dpi=150)
    plt.close()
    generated_plots.append(plot1_path)

    # Plot B: Chunk Size vs Acceptance Rates (Combined for all positions)
    plt.figure(figsize=(6, 4))
    for i in range(num_rates):
        plt.plot(
            chunk_sizes,
            acc_rates_by_pos[i],
            marker="s",
            linestyle="-",
            linewidth=2,
            label=f"Token Pos {i+1} Rate",
        )
    plt.title("Chunk Size vs Acceptance Rates (Combined)")
    plt.xlabel("Max Article Chunk Size (num tokens)")
    plt.ylabel("Acceptance Rate")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.xticks(chunk_sizes, rotation=45, ha="right")  # Updated here
    plot2_path = "plot_combined_acc_rates.png"
    plt.savefig(plot2_path, bbox_inches="tight", dpi=150)
    plt.close()
    generated_plots.append(plot2_path)

    # Plot C: Generate Individual Plots per Token Position
    individual_plot_paths = []
    for i in range(num_rates):
        plt.figure(figsize=(6, 3))
        plt.plot(
            chunk_sizes,
            acc_rates_by_pos[i],
            marker="^",
            color="green",
            linestyle="-",
            linewidth=1.5,
        )
        plt.title(f"Chunk Size vs Acceptance Rate (Token Position {i+1})")
        plt.xlabel("Max Article Chunk Size (num tokens)")
        plt.ylabel("Acceptance Rate")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xticks(chunk_sizes, rotation=45, ha="right")  # Updated here
        plot_path = f"plot_acc_rate_pos_{i+1}.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=130)
        plt.close()
        individual_plot_paths.append(plot_path)
        generated_plots.append(plot_path)

    # --- Generate Article Size Histogram (If Data Provided) ---
    has_article_sizes = bool(article_sizes_json)
    if has_article_sizes:
        article_data = json.loads(article_sizes_json)
        raw_sizes = article_data.get("article_sizes", [])

        # Filter outliers > 100,000
        filtered_sizes = [s for s in raw_sizes if s <= 100000]
        outliers_count = len(raw_sizes) - len(filtered_sizes)

        # Generate Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(filtered_sizes, bins=30, color="#8E44AD", edgecolor="black", alpha=0.8)
        plt.title("Distribution of Article Sizes")
        plt.xlabel("Article Size (num tokens)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45, ha="right")  # Updated here to ensure consistency
        hist_plot_path = "plot_article_sizes_hist.png"
        plt.savefig(hist_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        generated_plots.append(hist_plot_path)

    # --- Build Document Elements using ReportLab ---
    doc = SimpleDocTemplate(output_pdf_name, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SectionHeader",
            parent=styles["Heading2"],
            spaceBefore=12,
            spaceAfter=6,
        )
    )

    elements = []
    section_num = 1

    # Title
    elements.append(
        Paragraph("Effect of Context Size on Eagle3 Performance", styles["Title"])
    )
    elements.append(Spacer(1, 12))

    # Add Methodology Section
    if methodology_md:
        elements.append(
            Paragraph(f"<b>{section_num}. Methodology</b>", styles["Heading1"])
        )
        elements.append(Spacer(1, 6))
        methodology_flowables = parse_markdown_to_reportlab(methodology_md, styles)
        elements.extend(methodology_flowables)
        elements.append(Spacer(1, 12))
        section_num += 1

    # Add Summary Plots
    elements.append(
        Paragraph(f"<b>{section_num}. Overview Analysis</b>", styles["Heading1"])
    )
    elements.append(Spacer(1, 6))
    elements.append(
        Paragraph(f"{section_num}.1 Mean Acceptance Length", styles["SectionHeader"])
    )
    elements.append(Image(plot1_path, width=400, height=266))
    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            f"{section_num}.2 Combined Token Position Acceptance Rates",
            styles["SectionHeader"],
        )
    )
    elements.append(Image(plot2_path, width=400, height=266))
    elements.append(Spacer(1, 24))
    section_num += 1

    # Add Individual Plots to PDF
    elements.append(
        Paragraph(
            f"<b>{section_num}. Individual Token Position Analysis</b>",
            styles["Heading1"],
        )
    )
    elements.append(Spacer(1, 6))
    for i, plot_path in enumerate(individual_plot_paths):
        elements.append(
            Paragraph(
                f"{section_num}.{i+1} Token Position {i+1} Acceptance Rate",
                styles["SectionHeader"],
            )
        )
        elements.append(Image(plot_path, width=350, height=175))
        elements.append(Spacer(1, 10))
    section_num += 1

    # Add Data Summary Table
    elements.append(Spacer(1, 24))
    elements.append(
        Paragraph(f"<b>{section_num}. Data Summary Table</b>", styles["Heading1"])
    )
    elements.append(Spacer(1, 12))

    table_data = []
    headers = ["Chunk Size", "Mean Acc Length"] + [
        f"Pos {i+1} Rate" for i in range(num_rates)
    ]
    table_data.append(headers)

    for d in data:
        row = [str(d["chunk_size"]), f"{d['mean_acceptance_length']:.4f}"]
        row.extend([f"{rate:.4f}" for rate in d["acceptance_rates"]])
        table_data.append(row)

    t = Table(table_data)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EAEAEA")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    elements.append(t)
    section_num += 1

    # Add Histogram Section
    if has_article_sizes:
        elements.append(Spacer(1, 24))
        elements.append(
            Paragraph(
                f"<b>{section_num}. Article Size Distribution</b>", styles["Heading1"]
            )
        )
        elements.append(Spacer(1, 6))

        note_text = (
            f"The histogram below illustrates the distribution of article sizes within the dataset. "
            f"Articles exceeding 100,000 tokens were considered outliers and omitted from this visualization. "
            f"<b>Note:</b> {outliers_count} outlier(s) were filtered out."
        )

        elements.append(Paragraph(note_text, styles["BodyText"]))
        elements.append(Spacer(1, 12))
        elements.append(Image(hist_plot_path, width=400, height=266))

    # Build PDF
    doc.build(elements)
    print(f"Detailed report successfully generated at: {output_pdf_name}")

    # Clean up generated image files after building the PDF
    for plot_file in generated_plots:
        if os.path.exists(plot_file):
            os.remove(plot_file)


if __name__ == "__main__":
    RESULTS_DIR = "results"
    METHODOLOGY_FILE = "methodology.md"
    ARTICLE_SIZES_FILE = "article_sizes.json"

    # 1. Load main JSON data from directory
    all_results = []
    if os.path.isdir(RESULTS_DIR):
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(RESULTS_DIR, filename)
                try:
                    with open(filepath, "r") as f:
                        res = json.load(f)
                        all_results.append(res)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    else:
        print(
            f"Error: Directory '{RESULTS_DIR}' was not found. Please ensure it exists."
        )
        exit(1)

    if not all_results:
        print(f"Error: No valid JSON files found in '{RESULTS_DIR}'.")
        exit(1)

    # Sort results by chunk_size to ensure plots line up properly
    all_results.sort(key=lambda x: x.get("chunk_size", 0))

    # 2. Load Methodology Markdown
    methodology_text = ""
    if os.path.exists(METHODOLOGY_FILE):
        with open(METHODOLOGY_FILE, "r", encoding="utf-8") as f:
            methodology_text = f.read()

    # 3. Load Article Sizes JSON
    article_sizes_text = ""
    if os.path.exists(ARTICLE_SIZES_FILE):
        with open(ARTICLE_SIZES_FILE, "r", encoding="utf-8") as f:
            article_sizes_text = f.read()

    # 4. Run the report generator
    generate_report(all_results, methodology_text, article_sizes_text)
