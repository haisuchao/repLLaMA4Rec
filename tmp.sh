python << 'EOF'
import re

path = "/media/administrator/Data1/Projects/Python/repLLaMA/tevatron/src/tevatron/retriever/trainer.py"

with open(path) as f:
    content = f.read()

# Fix 1: save_safetensors
content = content.replace(
    'safe_serialization=self.args.save_safetensors',
    'safe_serialization=getattr(self.args, "save_safetensors", True)'
)

# Fix 2: self.tokenizer → dùng processing_class nếu không có tokenizer
content = content.replace(
    'if self.tokenizer is not None:',
    'if getattr(self, "tokenizer", None) is not None or getattr(self, "processing_class", None) is not None:'
)
content = content.replace(
    'self.tokenizer.save_pretrained(',
    '(self.tokenizer if getattr(self, "tokenizer", None) else self.processing_class).save_pretrained('
)

with open(path, 'w') as f:
    f.write(content)

print("Đã patch xong trainer.py")

# Xác nhận không còn self.tokenizer raw nào
remaining = [i+1 for i, l in enumerate(content.splitlines()) if 'self.tokenizer' in l]
if remaining:
    print(f"Còn self.tokenizer ở dòng: {remaining} — cần kiểm tra thêm")
else:
    print("Không còn self.tokenizer raw nào")
EOF