import os

def remove_duplicates_trec(input_file, output_file, is_qrels=False):
    seen = set()
    cleaned_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            
            # Bỏ qua dòng trống
            if len(parts) < 3:
                continue
                
            query_id = parts[0]
            # Trong qrels: <query> <0> <docid> <score> -> docid ở cột 2 (index 2)
            # Trong run: <query> <Q0> <docid> <rank> <score> <run> -> docid ở cột 2 (index 2)
            doc_id = parts[2]
            
            # Tạo khóa nhận diện duy nhất
            unique_pair = f"{query_id}_{doc_id}"
            
            if unique_pair not in seen:
                seen.add(unique_pair)
                cleaned_lines.append(line)
            else:
                print(f"Đã xóa dòng trùng lặp: query={query_id}, doc={doc_id} trong file {input_file}")

    # Ghi ra file mới
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    print(f"✅ Đã dọn dẹp xong file: {output_file}\n")

# --- Thực thi ---
qrels_in = "embeddings/results/qrels.txt"
qrels_out = "embeddings/results/qrels_clean.txt"

run_in = "embeddings/results/rank_result.trec"
run_out = "embeddings/results/rank_result_clean.trec"

print("Đang xử lý qrels...")
remove_duplicates_trec(qrels_in, qrels_out, is_qrels=True)

print("Đang xử lý rank_result...")
remove_duplicates_trec(run_in, run_out, is_qrels=False)