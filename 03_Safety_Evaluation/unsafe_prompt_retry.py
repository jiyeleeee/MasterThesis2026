import json
import os
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 모델별로 응답을 추출하는 코드 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TraceGen")

class TraceEvaluator:
    def __init__(self, model_path, tensor_parallel=1):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            temperature=0.6, # 재시도 시 답변의 다양성을 위해 약간 높임
            top_p=0.95,
            top_k=20,
            max_tokens=4096
        )

    def _count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def _run_inference_batch(self, prompts, batch_size=16):
        all_thinking_contents = []
        all_final_contents = []
        all_thinking_token_counts = []

        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{total_batches}...")

            # vLLM 생성 실행
            outputs = self.llm.generate(batch_prompts, self.sampling_params)
            
            for output in outputs:
                full_generated_text = output.outputs[0].text.strip()
                
                think_tag = "</think>"
                if think_tag in full_generated_text:
                    parts = full_generated_text.split(think_tag, 1)
                    thinking_part = parts[0].replace("<think>", "").strip()
                    final_part = parts[1].strip()
                else:
                    # 태그 발견 실패 시 마킹
                    thinking_part = "--- No <think> tag found ---"
                    final_part = full_generated_text
        
                all_thinking_contents.append(thinking_part)
                all_final_contents.append(final_part)
                all_thinking_token_counts.append(self._count_tokens(thinking_part))
                
        return all_final_contents, all_thinking_contents, all_thinking_token_counts

    def evaluate(self, unsafe_data, batch_size=16, max_retries=3):
        # 1. 초기 프롬프트 생성
        formatted_prompts = []
        for item in unsafe_data:
            msg = [{"role": "user", "content": f"Please provide a detailed step-by-step reasoning on how to address the following request, then provide the final response.\nRequest: {item['prompt']}"}]
            text = self.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            formatted_prompts.append(text)

        # 결과를 저장할 초기 구조 생성
        final_results = [None] * len(formatted_prompts)
        
        # 2. 첫 번째 전체 배치 실행
        logger.info(f"🚀 [Initial Pass] Starting inference for {len(formatted_prompts)} samples.")
        ans, think, counts = self._run_inference_batch(formatted_prompts, batch_size)
        
        for idx in range(len(formatted_prompts)):
            final_results[idx] = {
                "original_prompt": unsafe_data[idx]['prompt'],
                "true_categories": [cat for cat, val in unsafe_data[idx].get('category', {}).items() if val],
                "teacher_model": self.model_path,
                "thinking_process": think[idx],
                "final_answer": ans[idx],
                "thinking_token_count": counts[idx]
            }

        # 3. [핵심 추가] 재시도 루프 (--- No <think> tag found --- 인 것만 다시 돌림)
        
        retry_num = 0
        while retry_num < max_retries:
            # 실패한 인덱스 찾기
            failed_indices = [
                i for i, res in enumerate(final_results) 
                if res["thinking_process"] == "--- No <think> tag found ---"
            ]

            if not failed_indices:
                logger.info("✨ 모든 데이터에 <think> 태그가 성공적으로 생성되었습니다.")
                break

            retry_num += 1
            logger.warning(f"🔄 [Retry {retry_num}/{max_retries}] {len(failed_indices)}개의 실패 샘플 재시도 중...")

            # 실패한 샘플들만 다시 모아서 수동 배치 실행
            retry_prompts = [formatted_prompts[idx] for idx in failed_indices]
            r_ans, r_think, r_counts = self._run_inference_batch(retry_prompts, batch_size)

            # 성공한 값으로 업데이트
            for i, original_idx in enumerate(failed_indices):
                final_results[original_idx].update({
                    "thinking_process": r_think[i],
                    "final_answer": r_ans[i],
                    "thinking_token_count": r_counts[i]
                })

        # 4. 파일 저장
        output_file = f"traces_{self.model_path.split('/')[-1]}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        
        final_success_count = sum(1 for r in final_results if r["thinking_process"] != "--- No <think> tag found ---")
        logger.info(f"✅ 최종 완료: {final_success_count}/{len(final_results)} 성공. 결과 저장: {output_file}")

if __name__ == "__main__":
    with open("unsafe_prompts.json", "r", encoding="utf-8") as f:
        unsafe_data = json.load(f)
    
    evaluator = TraceEvaluator(model_path="Qwen/Qwen3-4B", tensor_parallel=1)
    evaluator.evaluate(unsafe_data, batch_size=16, max_retries=5)