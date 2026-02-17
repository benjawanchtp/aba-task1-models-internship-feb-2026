# t5_autotune_binary.py
# Auto Fine-tune (Hyperparameter Search) NEG/POS with Seq2SeqTrainer + Optuna
# Excel columns: A=id, G=text, H=sentiment
# Split: train/val/test (เหมือน BERT auto) + export all trials + train final + test + save model

import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)

# ----------------------------
# 0) CONFIG
# ----------------------------
EXCEL_PATH = "/Users/bbbbben/Desktop/Project in Japan/Task1/ABA Dataset (remove off).xlsx"
MODEL_NAME = "t5-base"

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 4

SEED = 42
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAINVAL = 0.2

N_TRIALS = 5
OUTPUT_DIR = "t5_autotune_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Excel columns by index (0-index): A=id, G=text, H=sentiment
ID_COL_POS = 0
TEXT_COL_POS = 6
SENT_COL_POS = 7

LABELS = ["negative", "positive"]
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "Negative", 1: "Positive"}

# ----------------------------
# 1) Prompt + label helpers (หัวใจของ T5)
# ----------------------------

def build_prompt(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    return f"sentiment classification (negative, positive): {text}"

def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(c for c in s if c.isalpha())
    if s.startswith("neg"):
        return "negative"
    if s.startswith("pos"):
        return "positive"
    return "negative"  # fallback ป้องกันพังเวลา decode แปลกๆ


def labels_to_ids(texts):
    return np.array([LABEL2ID[normalize_label(t)] for t in texts], dtype=int)

# ----------------------------
# 2) Load + clean
# ----------------------------
df_raw = pd.read_excel(EXCEL_PATH)

if df_raw.shape[1] <= max(ID_COL_POS, TEXT_COL_POS, SENT_COL_POS):
    raise ValueError(
        f"Excel has {df_raw.shape[1]} columns; need at least {SENT_COL_POS+1} columns for A/G/H."
    )

id_col = df_raw.columns[ID_COL_POS]
text_col = df_raw.columns[TEXT_COL_POS]
sent_col = df_raw.columns[SENT_COL_POS]

df = df_raw[[id_col, text_col, sent_col]].rename(
    columns={id_col: "id", text_col: "text", sent_col: "sentiment"}
).copy()


df = df.dropna(subset=["text", "sentiment"]).copy()
df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

df = df[df["sentiment"].isin(LABELS)].copy()
df["label"] = df["sentiment"].map(LABEL2ID).astype(int)
df = df.reset_index(drop=True)

print("Rows:", len(df))
print("Label counts:\n", df["sentiment"].value_counts())

# ----------------------------
# 3) Split Train/Val/Test (เหมือน BERT auto)
# ----------------------------
# ใช้ train_test_split สองรอบ: รอบแรกแยก test ออกมา, รอบสองแยก val จาก trainval
trainval_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
)
train_df, val_df = train_test_split(
    trainval_df,
    test_size=VAL_SIZE_IN_TRAINVAL,
    random_state=SEED,
    stratify=trainval_df["label"],
)

print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

# ----------------------------
# 4) Tokenize + Dataset 
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# ฟังก์ชัน preprocess สำหรับ map ลง Dataset จะทำการสร้าง prompt จาก text และ normalize label จาก sentiment แล้ว tokenize ทั้ง input และ target 
def preprocess(batch):
    inputs = [build_prompt(t) for t in batch["text"]]
    targets = [normalize_label(s) for s in batch["sentiment"]]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    labels = tokenizer(
        text_target=targets,
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def make_ds(frame: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(frame[["text", "sentiment"]].reset_index(drop=True))
    return ds.map(preprocess, batched=True)

# สร้าง Dataset สำหรับ train/val/test โดยใช้ฟังก์ชัน make_ds ที่ทำการ map preprocess ซึ่งจะได้คอลัมน์ input_ids, attention_mask, labels 
train_ds = make_ds(train_df)
val_ds   = make_ds(val_df)
test_ds  = make_ds(test_df)

# ----------------------------
# 5) Metrics (Seq2Seq: decode generated tokens -> map to neg/pos)
# ----------------------------

# ฟังก์ชัน compute_metrics จะรับ eval_pred ซึ่งประกอบด้วย preds (token ids ที่โมเดล generate) และ labels (token ids จริงที่มี -100 สำหรับ padding) โดยจะ decode ทั้งคู่เป็นข้อความ แล้ว map เป็น 0/1 ก่อนคำนวณ accuracy, f1_macro, precision_macro, recall_macro
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # preds อาจเป็น tuple ในบางเวอร์ชัน
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # labels: replace -100 -> pad_token_id แล้วค่อย decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    y_pred = labels_to_ids(decoded_preds)
    y_true = labels_to_ids(decoded_labels)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "precision_macro": p,
        "recall_macro": r,
    }

# ----------------------------
# 6) model_init 
# ----------------------------
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ----------------------------
# 7) Optuna objective
# ----------------------------

# ฟังก์ชัน objective สำหรับ Optuna จะรับ trial แล้วทำการตั้งค่า hyperparameters จาก trial.suggest_* จากนั้นสร้าง Seq2SeqTrainingArguments และ Seq2SeqTrainer เพื่อฝึกสอนโมเดลบน train_ds และประเมินผลบน val_ds โดยจะเก็บ metrics ต่างๆ เป็น user attributes ของ trial และคืนค่า f1_macro เพื่อให้ Optuna ใช้ในการ optimize
def objective(trial):
    set_seed(SEED)

    lr = trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True)
    bs = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("num_train_epochs", [2, 3, 4])
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)

    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number}")

    args = Seq2SeqTrainingArguments(
        output_dir=trial_dir,
        eval_strategy="epoch",             
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=wd,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model_init(),
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # ✅ tune บน val เท่านั้น 
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

# Timing การฝึกสอนและประเมินผลในแต่ละ trial เพื่อเก็บเป็น user attributes ของ trial นั้นๆ
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0


    metrics = trainer.evaluate(val_ds)

    trial.set_user_attr("train_seconds", train_seconds)
    trial.set_user_attr("eval_accuracy", metrics.get("eval_accuracy"))
    trial.set_user_attr("eval_precision_macro", metrics.get("eval_precision_macro"))
    trial.set_user_attr("eval_recall_macro", metrics.get("eval_recall_macro"))
    trial.set_user_attr("eval_loss", metrics.get("eval_loss"))

    return metrics["eval_f1_macro"]

# ----------------------------
# 8) Run study + export all trials
# ----------------------------
# สร้าง Optuna study แล้ว optimize ด้วย objective function ที่เราสร้างไว้ โดยกำหนด n_trials ตามที่ต้องการ จากนั้นดึง best_params และ best_value (f1_macro) มาแสดงผล
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
best_params = study.best_params

print("\n=== BEST HYPERPARAMETERS ===")
print(study.best_params)
print("Best f1_macro:", study.best_value)

rows = []
for t in study.trials:
    if t.value is None:
        continue

    duration_hms = "00:00:00"
    if t.datetime_start and t.datetime_complete:
        diff = t.datetime_complete - t.datetime_start
        duration_hms = str(timedelta(seconds=int(diff.total_seconds())))

    rows.append({
        "trial_id": t.number,
        "f1_macro": t.value,
        "state": str(t.state),
        "duration_hms": duration_hms,
        **t.params,
        **t.user_attrs,
    })

df_history = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
HISTORY_XLSX = os.path.join(OUTPUT_DIR, "all_trials_detailed_results.xlsx")
df_history.to_excel(HISTORY_XLSX, index=False)
print(f"\n✅ Saved trials history to: {HISTORY_XLSX}")

# ----------------------------
# 9) Train final model with best hyperparameters
# ----------------------------

final_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "t5_best_model_run"),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    logging_steps=50,
    **best_params
)

final_trainer = Seq2SeqTrainer(
    model=model_init(),
    args=final_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

final_trainer.train()

# ----------------------------
# 10) Evaluate on test + report + save predictions + save model
# ----------------------------
#
test_metrics = final_trainer.evaluate(test_ds)
print("\nTEST METRICS:", test_metrics)

pred_out = final_trainer.predict(test_ds)

preds = pred_out.predictions
if isinstance(preds, tuple):
    preds = preds[0]

decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

label_ids = pred_out.label_ids
label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

y_pred = labels_to_ids(decoded_preds)
y_true = labels_to_ids(decoded_labels)

print("\n=== TEST classification_report ===")
print(classification_report(y_true, y_pred, target_names=[ID2LABEL[0], ID2LABEL[1]], digits=4, zero_division=0))

print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(confusion_matrix(y_true, y_pred))

# Save final test metrics (เหมือน BERT auto)
final_summary = {
    "best_learning_rate": study.best_params.get("learning_rate"),
    "best_batch_size": study.best_params.get("per_device_train_batch_size"),
    "best_num_train_epochs": study.best_params.get("num_train_epochs"),
    "best_weight_decay": study.best_params.get("weight_decay"),
    "test_loss": test_metrics.get("eval_loss"),
    "test_accuracy": test_metrics.get("eval_accuracy"),
    "test_f1_macro": test_metrics.get("eval_f1_macro"),
    "test_precision_macro": test_metrics.get("eval_precision_macro"),
    "test_recall_macro": test_metrics.get("eval_recall_macro"),
}
FINAL_XLSX = os.path.join(OUTPUT_DIR, "final_test_metrics.xlsx")
pd.DataFrame([final_summary]).to_excel(FINAL_XLSX, index=False)
print("\n✅ Saved final test metrics to:", FINAL_XLSX)

# Save predictions to Excel
out_df = test_df.copy().reset_index(drop=True)
out_df["pred_text_raw"] = decoded_preds
out_df["pred_sentiment"] = [ID2LABEL[i] for i in y_pred]
out_df["correct"] = (y_pred == y_true)

PRED_XLSX = os.path.join(OUTPUT_DIR, "t5_bestmodel_test_predictions.xlsx")
out_df.to_excel(PRED_XLSX, index=False)
print("✅ Saved test predictions to:", PRED_XLSX)

# Save final best model
SAVE_DIR = os.path.join(OUTPUT_DIR, "t5_best_model_saved")
final_trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("\n✅ Saved best model to:", SAVE_DIR)
