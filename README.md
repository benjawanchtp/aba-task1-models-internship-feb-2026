# aba-task1-models-internship-feb-2026
This is a repository that contains the experiments for ABA Task 1, jointly conducted with internship students in February 2026. 

# Project Overview
**ABA-Task1-Models** ในโปรเจคนี้เป็นการนำชุดข้อมูล **Hotel Review** มาพัฒนาโมเดลสำหรับทำ **Sentiment Classification** ภายใต้แนวคิด **Aspect-Based Analysis (ABA)** เพื่อประเมินประสิทธิภาพของโมเดล **Transformers** ในการวิเคราะห์ความคิดเห็นเชิงลึกจากรีวิวลูกค้า

# Objective
1. เพื่อทดสอบประสิทธิภาพของโมเดล Trasformers ในการจำแนกความรู้สึกจากรีวิว (Sentiment Classification from Hoel Review)
2. เปรียบเทียบประสิทธิภาพของโมเดลที่มีสถาปัตยกรรมต่างกัน

# Model Architecture 
ในการทดสอบประสิทธิภาพโมเดล Transformers ในงานนี้ได้เลือกใช้โมเดลหลายสถาปัตยกรรมเพื่อเปรียบเทียบผลลัพธ์ ได้แก่ 
1. BERT-base-uncased และ RoBERTa เป็นโมเดลแบบ encoder-based เหมาะสำหรับงานจำแนกประเภทข้อความ (classification)
2. BART เป็นโมเดลแบบ encoder–decoder ที่สามารถทำได้ทั้งงานจำแนก (classification) และงานสร้างข้อความ (generation)
3. T5 เป็นโมเดล text-to-text ที่แปลงทุกงานให้อยู่ในรูปแบบการสร้างข้อความ และสามารถทำงานแบบ multitask ได้ภายใต้สถาปัตยกรรมเดียว

# Training Config
ในการตั้งค่า config ของแต่ละโมเดล ได้นำการตั้งค่าอ้างอิงมาจาก **[Hugging Face](https://huggingface.co/docs/transformers/trainer)** ซึ่งค่าที่นำมาเป็นค่ามาตราฐานที่ที่กันทั่วไป
โดยการตั้งค่า parameters :

  -	learning_rate = 2e-5
  -	batch_size (train/eval) = 16
  -	num_train_epochs = 3
  -	weight_decay = 0.01
  -	max_length = 256
 
# Project Structure
ในโฟลเดอร์เก็บข้อมูลจะแบ่งออกเป็นทั้งหมด 3 โฟลเดอร์ และ 2 ไฟล์งาน โดยรายละเอียดแต่ละโฟลเดอร์และงานมีดังนี้ :

## 1. dataset
เป็นชุดข้อมมูลที่ใช้สำหรับการทดลอง (Experiments) โดยจะมีการเลือกเฉพาะ Column ที่ใช้งานจริงคือ `Column A : ID`, `Column G : Selected Content`, `Column H : Pos/Neg`
โดยชุดข้อมูลจะแบ่งออกเป็นทั้งหมด 2 ชุดข้อมูลคือ
- **Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx** : เป็นชุดข้อมูลที่ยังมี noise (have topic, sentiment : off) อยู่ และยังเป็นชุดที่ได้นำข้อมูลไอดีที่ไม่เอาออกทั้งหมด 151 ID เพื่อทดสอบการจำแนกอารมณ์ของโมเดล
- **ABA Dataset (remove off).xlsx** : ชุดข้อมูลนี้ได้มีการจัดการ noise (delete topic, sentiment : off) ออกแล้ว โดยการจัดการจะเป็นการจัดการด้วยมือ (manual) เพื่อนำมาทดสอบประสิทธิภาพการจำแนกอารมณ์ของโมเดลในขณะที่ใช้ข้อมูลที่ไม่มี noise

> สำหรับ version dataset original ก่อนที่นำมา preprocess data ใช้เป็นชุดข้อมูลที่ชื่อ [Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT](https://docs.google.com/spreadsheets/d/1hf5YqZMAMbDOSIH9OwhQvOTIIBXpdaPV_54rxZbVRdU/edit?gid=850627401#gid=850627401)


## 2. model_code
เป็นโฟลเดอร์ที่รวม Python Script (.py) สำหรับการรันการทดลองทั้งหมดไว้
- **model_code_ABA_T5** : โค้ดสำหรับรันการทดลอง Multi-task Learning โดยนำหลักการของโมเดล T5 มาใช้ โดยแยก prompt ออกเป็นทั้งหมด 2 format
  - `ABA_T5_multi_prefix_format.py` : prefix format
  - `ABA_T5_multi_prompt_format.py` : prompt format 
- **model_code_auto_finetune** : โค้ดสำหรับรันการทดลอง Auto Finetuning โดยใช้ Optuna Hyperparameters แบ่งออกเป็นทั้งหมด 4 โค้ดตามโมเดลที่ใช้เทรน
  - `bart_autofine.py`
  - `bert_autofine.py`
  - `roberta_autofine.py`
  - `t5_autofine.py`
- **model_code_kfold** : โค้ดสำหรับรันการทดลอง K-fold จะแบ่งออกเป็นทั้งหมด 4 โค้ดตามโมเดลที่ใช้เทรน โดยในโค้ดจะมีการเทรนด้วย [K = 1,3]
  - `bart_kfold.py`
  - `bert_kfold.py`
  - `roberta_kfold.py`
  - `t5_kfold.py`
- **model_code_romove_off** : โค้ดสำหรับรันการทดลองกับชุดข้อมูลที่ไม่มี noise  (delete topic, sentiment : off) จะแบ่งออกเป็นทั้งหมด 4 โค้ดตามโมเดลที่ใช้เทรน
  - `bart_remove_off.py`
  - `bert_remove_off.py`
  - `roberta_remove_off.py`
  - `t5_remove_off.py`
- **model_code_with_off** : โค้ดสำหรับรันการทดลองกับชุดข้อมูลที่ยังมี noise  (have topic, sentiment : off) จะแบ่งออกเป็นทั้งหมด 4 โค้ดตามโมเดลที่ใช้เทรน
  - `bart_base.py`
  - `bert_base.py`
  - `roberta_base.py`
  - `t5_base.py`
 
## 3. ตารางผลการทดลอง
- **with "Off" dataset > BERT-base-uncase, RoBERTa-base, BART-base, T5-base**

  
| Model              | Accuracy | Precision | Recall  | F1-score |
|--------------------|---------:|----------:|--------:|---------:|
| BERT-base-uncase   | 88.08%   | 77.37%    | 68.66%  | 71.94%   |
| BART-base          | 86.53%   | 71.74%    | 67.98%  | 69.49%   |
| RoBERTa-base       | 88.60%   | 75.90%    | 75.85%  | 75.86%   |
| **T5-base**      | **88.60%** | **88.60%** | **77.59%** | **77.97%** |

- **without "Off" dataset > BERT-base-uncase, RoBERTa-base, BART-base, T5-base**


| Model              | Accuracy | Precision | Recall  | F1-score |
|--------------------|---------:|----------:|--------:|---------:|
| BERT-base-uncase   | 98.16%   | 98.15%    | 98.16%  | 98.15%   |
| BART-base          | 97.55%   | 97.62%    | 97.55%  | 97.47%   |
| RoBERTa-base       | 95.71%   | 95.92%    | 95.71%  | 95.47%   |
| **T5-base**     | **98.77%** | **98.77%** | **98.77%** | **98.77%** |

- **K-Fold = 1 > BERT-base-uncase, RoBERTa-base, BART-base, T5-base**

| Model              | K | Accuracy | F1-macro | Precision | Recall |
|--------------------|--:|----------------:|----------------:|-----------------:|--------------:|
| BERT-base-uncase   | 1 | 0.9723 | 0.9531 | 0.9443 | 0.9625 |
| BART-base          | 1 | 0.9815 | 0.9676 | 0.9744 | 0.9612 |
| RoBERTa-base       | 1 | 0.9784 | 0.9625 | 0.9657 | 0.9593 |
| **T5-base**     | 1 | **0.9846** | **0.9735** | **0.9703** | **0.9768** |

- **K-Fold = 3 > BERT-base-uncase, RoBERTa-base, BART-base, T5-base**

| Model              | K |Accuracy | Precision | Recall  | F1-macro |
|--------------------|--:|---------:|----------:|--------:|---------:|
|**BART-base**       | 3 |**0.9846** | **0.9809** | **0.9653** | **0.9729** |
| T5-base            | 3 |0.9825 | 0.9735 | 0.9664 | 0.9697 |
| BERT-base-uncase   | 3 |0.9774 | 0.9506 | 0.9749 | 0.9621 |
| Roberta-base       | 3 |0.9733 | 0.9642 | 0.9424 | 0.9526 |
 


















#### install & setup ####
1. create virtual environment

`python -m venv benjawan_nu`

**MacOS/Linux**

`source benjawan_nu/bin/activate`

**Window**

`benjawan_nu\Scripts\activate`

2. install dependencies

(ติดตั้งแพกเกจทั้งหมดจาก lock file)

`pip install -r requirements.lock.txt`

(แนะนำให้อัปเดต pip ก่อนติดตั้ง)

`python -m pip install --upgrade pip`

3. verifly installation  (ตรวจสอบว่า environment ถูกต้อง)

`pip list`

**requiremrnt**
ใน lock file ตัวอย่างแพกเกจที่ใช้ใน project นี้ :
- transformers
- datasets
- torch
- scikit-learn
- optuna
- pandas
- numpy
- evaluate

