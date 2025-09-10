import pandas as pd
import os
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load your CSV file
csv_path = "simulated_chatbot_responses.csv"

try:
    df = pd.read_csv(csv_path)
    print("📁 File loaded:", csv_path)
    print("📊 Number of rows in CSV:", len(df))
    print("📋 Columns found:", df.columns.tolist())
except Exception as e:
    print("❌ Error loading file:", e)
    exit()

# Check required columns
required_cols = {"Answer", "Chatbot_Response"}
if not required_cols.issubset(df.columns):
    print(f"❌ CSV is missing required columns: {required_cols - set(df.columns)}")
    exit()

# Initialize scorers
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_scores, bleu_scores, bert_f1_scores = [], [], []

print("🔍 Starting evaluation...\n")

# Evaluate each row
for i in range(len(df)):
    ref = str(df.loc[i, "Answer"])
    hyp = str(df.loc[i, "Chatbot_Response"])

    # ROUGE-L
    try:
        rouge_f = rouge.score(ref, hyp)['rougeL'].fmeasure
    except Exception as e:
        print(f"❌ ROUGE error on row {i+1}: {e}")
        rouge_f = 0.0

    # BLEU
    try:
        bleu = sentence_bleu(
            [ref.split()],
            hyp.split(),
            smoothing_function=SmoothingFunction().method1
        )
    except Exception as e:
        print(f"❌ BLEU error on row {i+1}: {e}")
        bleu = 0.0

    # BERTScore
    try:
        _, _, f1 = score([hyp], [ref], lang="en", verbose=False)
        bert_f1 = f1[0].item()
    except Exception as e:
        print(f"❌ BERTScore error on row {i+1}: {e}")
        bert_f1 = 0.0

    # Store results
    rouge_scores.append(round(rouge_f, 4))
    bleu_scores.append(round(bleu, 4))
    bert_f1_scores.append(round(bert_f1, 4))

    print(f"✅ Evaluated row {i+1}/{len(df)}")

# Add metrics to DataFrame
df["ROUGE-L"] = rouge_scores
df["BLEU"] = bleu_scores
df["BERTScore_F1"] = bert_f1_scores

# Save evaluated results
output_file = "evaluated_chatbot_responses.xlsx"
df.to_excel(output_file, index=False)

# Confirm save location
full_path = os.path.abspath(output_file)
if os.path.exists(output_file):
    print(f"\n✅ Evaluation complete. File saved at:\n{full_path}")
else:
    print("❌ File was not saved.")
