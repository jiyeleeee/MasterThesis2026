from datasets import load_dataset
import json

def prepare_unsafe_inputs(num_samples=5000):
    print("BeaverTails loading...")
    dataset = load_dataset("PKU-Alignment/BeaverTails", split='330k_test')    
    unsafe_data = dataset.filter(lambda x: x['is_safe'] == False)
    
    # 3. 실험에 사용할 샘플링 (너무 많으면 시간이 오래 걸리므로 num_samples로 조절)
    if len(unsafe_data) > num_samples:
        #unsafe_data = unsafe_data.select(range(num_samples))
        unsafe_data = unsafe_data.shuffle(seed=42).select(range(num_samples))
    
    # 질문(prompt)과 나중에 비교를 위한 카테고리 정보 저장
    unsafe_prompts = []
    for item in unsafe_data:
        unsafe_prompts.append({
            "prompt": item['prompt'],
            "category": item['category']
        })
        
    
    with open("unsafe_prompts.json", "w", encoding="utf-8") as f:
        json.dump(unsafe_prompts, f, ensure_ascii=False, indent=4)
    
    print(f"extracted: {len(unsafe_prompts)} questions.")
    return unsafe_prompts


prompts_list = prepare_unsafe_inputs(num_samples=5000)