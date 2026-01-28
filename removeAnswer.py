import json
import os

def remove_answers_and_save_perfectly(input_file):
    try:
        directory, filename = os.path.split(input_file)
        
        output_dir = "AMC23_remove answers from result"
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, filename)

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        separators = [
            "**Final Answer**",
            "The answer is",
            "the answer is",
            "Therefore, the answer is",
            "Therefore, unique solution is",
            "Therefore, I can be confident",
            "Therefore, I feel confident",
            "the answer should be",
            "Hence, the answer is",
            "Thus, the answer is",
            #"i think the answer is"
            "Final Answer"
        ]

        count = 0
        if "detailed_results" in data:
            for item in data["detailed_results"]:
                if "thinking_content" in item:
                    content = item["thinking_content"]
                    
                
                    found_indices = []
                    for sep in separators:
                        index = content.find(sep)
                        if index != -1:
                            found_indices.append(index)
                    
                    if found_indices:
                        cutoff_index = min(found_indices)
                        
                        item["thinking_content"] = content[:cutoff_index].strip()
                        count += 1
                    
                    else:
                        lines = content.split('\n')
                        if lines and "\\boxed" in lines[-1]:
                            item["thinking_content"] = "\n".join(lines[:-1]).strip()
                            count += 1

        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ 완벽하게 처리되었습니다! 총 {count}개의 항목이 수정되었습니다.")
        print(f"📂 저장 폴더: {output_dir}")
        print(f"📄 저장 파일: {output_file}")

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")

# --- 실행 설정 ---
#target_file = '/home/jiyelee/master_thesis/evaluation_results_amc23_5/amc23_results_think_Qwen_Qwen3-32B_20251214_180434.json'
target_file = 'evaluation_results_amc23_5/amc23_results_think_Qwen_Qwen3-14B_20251212_234333.json'
remove_answers_and_save_perfectly(target_file)