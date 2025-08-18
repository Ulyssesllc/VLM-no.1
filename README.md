# QWENxGAN Workflow (Việt)

## 1. Môi trường
```bash
conda create -n qwen-gan python=3.10 -y
conda activate qwen-gan
pip install -r requirements.txt
```
Nếu dùng GPU NVIDIA: chọn đúng bản torch có CUDA (ví dụ `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

## 2. Cấu trúc dữ liệu
```
<root>/
  datasets Adidas Samba/
    real/  # ảnh thật
    fake/  # ảnh giả
    labels.csv  # image_path,text,label (0=fake,1=real)
```
Tạo `labels.csv` ví dụ:
```csv
image_path,text,label
real/img001.jpg,Logo sắc nét,1
fake/fake12.jpg,Chữ mờ,0
```

## 3. Fine-tune Qwen2-VL phân lớp
Chuẩn bị file `labels.csv` rồi chạy:
```bash
python -m src.train_classifier --data_root "datasets Adidas Samba" --csv labels.csv --epochs 3 --batch_size 2
```
Checkpoint sẽ lưu ở `outputs/classifier`.

## 4. Tạo ảnh giả bằng DCGAN (augment trong vòng lặp adversarial)
Script `train_adversarial.py` hiện tại: giả lập GAN đã huấn luyện (không update). Bạn có thể mở rộng thêm bước train DCGAN trước.
Chạy:
```bash
python -m src.train_adversarial --data_root "datasets Adidas Samba" --csv labels.csv --epochs 3 --gan_ratio 0.3
```

## 5. Giải thích kết quả dạng văn bản
Sau khi có checkpoint, có thể tải `AutoModelForSequenceClassification` và tự sinh câu giải thích bằng prompt template + rule-based hoặc fine-tune thêm phiên bản Instruct của Qwen2-VL (cần chuẩn bị cặp (ảnh+text, giải thích)).

## 6. Export mô hình
ONNX:
```python
import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('outputs/classifier')
dummy_inputs = {"input_ids": torch.zeros(1,16,dtype=torch.long), "attention_mask": torch.ones(1,16,dtype=torch.long)}
torch.onnx.export(model, (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]), "classifier.onnx", input_names=["input_ids","attention_mask"], output_names=["logits"], opset_version=17)
```
(Đối với mô hình đa modal cần xem forward signature cụ thể của Qwen2-VL sau khi thêm head.)

Quantization động (Linear):
```python
import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('outputs/classifier')
quant = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
quant.save_pretrained('outputs/classifier-quant')
```

CoreML: dùng `coremltools.convert` trên bản ONNX hoặc traced model.

## 7. Ghi chú mở rộng
- Cần bổ sung script train DCGAN riêng nếu muốn cải thiện chất lượng ảnh synthetic.
- Cân bằng lớp nếu dataset lệch (WeightedRandomSampler hoặc class weights).
- Thêm evaluation chuẩn (precision/recall, ROC-AUC) trong production.
- Kiểm tra license model Qwen2-VL cho mục đích sử dụng.

## 8. TODO tương lai
- Training vòng lặp GAN + cập nhật generator.
- Module giải thích (explanation head) sinh câu tự động.
- Pipeline export TFLite.

## 9. Multimodal Chatbot (Instruction Tuning) 🔄
### a. Tạo dataset instruction
1) Tạo `labels.csv` (đã mô tả ở trên).
2) Sinh dataset chat JSONL:
```bash
python -m src.build_chat_dataset --data_root "datasets Adidas Samba" --labels_csv labels.csv --out_jsonl chat_dataset.jsonl
```
### b. (Tuỳ chọn) Augment bằng GAN
Sinh ảnh giả -> thêm vào `labels.csv` (label=0) rồi chạy lại bước build JSONL.
### c. Fine-tune LoRA (Qwen2.5-VL-Instruct)
```bash
python -m src.train_chat_lora --model_profile vl_small --data_root "datasets Adidas Samba" --jsonl chat_dataset.jsonl --epochs 1 --batch_size 1 --use_4bit
```
### d. Inference dạng chatbot
```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
model_name = 'outputs/chat_lora'  # sau fine-tune
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
image = Image.open('test.jpg').convert('RGB')
query = 'Sản phẩm này có phải hàng thật không?'
inputs = processor(images=image, text=query, return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```
### e. Deploy
- Chỉ deploy backbone + LoRA weights (merge nếu cần).
- GAN không cần deploy.

## 10. Tuỳ chỉnh loại Qwen (model_profile)
Scripts `train_classifier.py` và `train_chat_lora.py` hỗ trợ 2 cách:
- `--model_profile <key>`: dùng mã tắt (dễ đổi) 
- hoặc `--model_name <huggingface_id>` để chỉ định trực tiếp (ưu tiên nếu set).

Bảng profile hiện tại:
| Key | HF Model |
|-----|----------|
| vl_base | Qwen/Qwen2-VL |
| vl_small | Qwen/Qwen2.5-VL-0.5B-Instruct |
| vl_medium | Qwen/Qwen2.5-VL-1.5B-Instruct |
| vl_large | Qwen/Qwen2.5-VL-7B-Instruct |
| text_small | Qwen/Qwen2.5-1.5B-Instruct |
| text_large | Qwen/Qwen2.5-7B-Instruct |

Ví dụ phân loại nhanh với model nhỏ:
```bash
python -m src.train_classifier --model_profile vl_small --data_root "datasets Adidas Samba" --csv labels.csv
```

Ví dụ chat LoRA với model medium (1.5B):
```bash
python -m src.train_chat_lora --model_profile vl_medium --data_root "datasets Adidas Samba" --jsonl chat_dataset.jsonl --epochs 1 --use_4bit
```

Ghi chú: Nếu model không có sẵn head SequenceClassification, code sẽ tự tạo `CustomQwenClassifier` dựa trên hidden states cuối.


---
Happy hacking!

## License

Phần mã nguồn trong repo này phát hành dưới giấy phép MIT (xem file `LICENSE`).

Lưu ý: Các mô hình Qwen/Qwen2.* tải từ Hugging Face tuân theo giấy phép riêng của nhà phát hành (Alibaba / cộng tác). Bạn cần kiểm tra và tuân thủ license gốc của model trước khi sử dụng trong sản phẩm thương mại. MIT ở đây chỉ áp dụng cho code hỗ trợ (scripts, util) do bạn/nhóm bổ sung.
