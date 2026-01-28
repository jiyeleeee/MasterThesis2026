import json
import os
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InjectionExperiment")

class StudentInjector:
    def __init__(self, model_path, gpu_util=0.9):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 소형 모델 로드
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.9, 
            trust_remote_code=True,
            max_model_len=4096 
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=4096
        )

    def run_injection(self, traces_data, batch_size=16):
        formatted_prompts = []
        
        system_prompt = (
            "You will be given a question and reasoning from bigger model. "
            "so you must not generate your own reasoning!! very important! "
            "Your task is to follow the provided reasoning exactly and provide ONLY the final answer. "
            "Do not generate your own reasoning. Simply execute the final answer as guided by the thoughts provided."
            "Instruction: Your internal thinking is already completed. Now, provide ONLY the direct final answer to the user's question above."
        )
        
        # 1. 모든 프롬프트를 미리 포맷팅
        for item in traces_data:
            question = item['original_prompt']
            thought = item['thinking_process']
            
            # 질문 + <think>내용 구성
            user_content = (
                f"{question}\n\n"
                f"<think>\n{thought.strip()}\n</think>\n\n"
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True 
            )
            formatted_prompts.append(input_text)

        # 2. 배치 실행 로직
        results = []
        total_samples = len(formatted_prompts)
        total_batches = (total_samples + batch_size - 1) // batch_size
        
        logger.info(f"🚀 Starting batch inference: {total_samples} samples in {total_batches} batches.")

        for i in range(0, total_samples, batch_size):
            current_batch_idx = i // batch_size + 1
            batch_prompts = formatted_prompts[i : i + batch_size]
            batch_data_subset = traces_data[i : i + batch_size]
            
            logger.info(f"📦 Processing batch {current_batch_idx}/{total_batches}...")
            
            # vLLM 배치 생성
            outputs = self.llm.generate(batch_prompts, self.sampling_params)

            # 결과 정리 및 취합
            for j, output in enumerate(outputs):
                results.append({
                    "prompt": batch_data_subset[j]['original_prompt'],
                    "injected_thought": batch_data_subset[j]['thinking_process'],
                    "student_output": output.outputs[0].text.strip(),
                    "true_categories": batch_data_subset[j]['true_categories'],
                    "teacher_model": batch_data_subset[j]['teacher_model']
                })

        # 3. 파일 저장
        model_name = self.model_path.split('/')[-1]
        output_file = f"results_injected_{model_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"✅ Result saved : {output_file}")

if __name__ == "__main__":
    trace_path = "traces_Qwen3-32B.json" 
    if os.path.exists(trace_path):
        with open(trace_path, "r", encoding="utf-8") as f:
            traces_data = json.load(f)
        
        STUDENT_PATH = "Qwen/Qwen3-8B" 
        
        injector = StudentInjector(STUDENT_PATH)
        injector.run_injection(traces_data, batch_size=16)
    else:
        logger.error(f"❌ {trace_path}  you can't find a file.")