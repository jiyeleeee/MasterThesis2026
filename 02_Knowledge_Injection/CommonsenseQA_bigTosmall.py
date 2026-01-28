import os
import re
import json
import time
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
from datasets import load_dataset
from datetime import datetime
from transformers import AutoTokenizer # Keep this for the chat template
import logging

# 앞서 정의한 기본 평가 클래스 임포트 (같은 파일에 있다면 생략 가능)
# from your_script import CommonsenseQAEvaluator 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommonsenseQAEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen3-32B"): 
        """
        Initializes the CommonsenseQAEvaluator with a specified LLM.
        """
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9,
            max_model_len=3072, 
            trust_remote_code=True,
            dtype="auto" 
        )
        
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_tokenizer.pad_token_id = 151643

        #self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Qwen 모델에 따라 pad_token_id 설정 (필요시 수정)
        #if self.hf_tokenizer.pad_token_id is None:
        #    self.hf_tokenizer.pad_token_id = self.hf_tokenizer.eos_token_id

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=3072, 
            stop=["<|im_end|>", "<|endoftext|>"] 
        )
        logger.info("Model and tokenizer initialization complete!")

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.hf_tokenizer.encode(text, add_special_tokens=False))

    # _run_inference_batch는 수정하지 않고 그대로 둡니다.
    # (단, TruthfulQA 코드처럼 (final, thinking, tokens) 순서로 리턴한다고 가정합니다.
    #  만약 순서가 다르다면 아래 evaluate에서 받아오는 변수 순서만 맞춰주세요.)
    def _run_inference_batch(self, prompts: List[str], batch_size: int = 16) -> Tuple[List[str], List[str], List[int]]:
        all_thinking_contents = []
        all_final_contents = []
        all_thinking_token_counts = []

        total_batches = (len(prompts) + batch_size - 1) // batch_size
        logger.info(f"Starting inference for {len(prompts)} prompts in {total_batches} batches...")

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}...")
            
            outputs = self.llm.generate(batch_prompts, self.sampling_params)

            for output in outputs:
                full_generated_text = output.outputs[0].text.strip()
                
                think_tag = "</think>"
                if think_tag in full_generated_text:
                    parts = full_generated_text.split(think_tag, 1)
                    thinking_part = parts[0].replace("<think>", "").strip()
                    final_part = parts[1].strip()
                else:
                    thinking_part = "--- No <think> tag found ---"
                    final_part = full_generated_text

                all_thinking_contents.append(thinking_part)
                all_final_contents.append(final_part)
                all_thinking_token_counts.append(self._count_tokens(thinking_part))
                
        logger.info("Inference complete.")
        # TruthfulQA 코드와 호환되도록 (final, thinking, tokens) 순서로 리턴
        return all_final_contents, all_thinking_contents, all_thinking_token_counts

    def _extract_answer_from_text(self, final_content: str) -> str:
        """
        Extracts the final answer (A-E) using robust regex logic.
        """
        cleaned_text = final_content.replace('**', '').strip()

        latex_pattern = r"\\boxed\{(?:\\text\{)?\s*([A-E])"
        latex_matches = re.findall(latex_pattern, cleaned_text, re.IGNORECASE)

        if latex_matches:
            return latex_matches[-1].strip().upper()

        # 1. [Strong Pattern] "Answer is E", "Final Answer: A" 등
        strong_pattern = r"(?:Final\s*Answer|Answer|Best\s*Answer|Correct\s*Answer)(?:\s*is)?[:.\s]*([A-E])"
        all_matches = re.findall(strong_pattern, cleaned_text, re.IGNORECASE)

        if all_matches:
            return all_matches[-1].strip().upper()

        # 2. [Weak Pattern] 괄호 (A) 또는 독립된 문자 A
        weak_matches = re.findall(r"\(?([A-E])\)?", cleaned_text)
        
        if weak_matches:
            return weak_matches[-1].strip().upper()

        return ""

    def save_results(self, results: Dict, filename: str):
        output_dir = "evaluation_results_commonsense_multipass"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save results to {filepath}: {e}")

    def evaluate(self, max_samples: int = None, batch_size: int = 16, max_passes: int = 3):
        """
        Evaluates CommonsenseQA using the robust Multi-Pass Retry logic 
        matches the structure of the TruthfulQA evaluator.
        """
        logger.info(f"Starting CommonsenseQA Evaluation ({max_passes}-Pass Retry Logic)")
        logger.info("="*40)

        try:
            dataset = load_dataset("commonsense_qa", split="validation")
        except Exception as e:
            logger.error(f"Failed to load dataset 'commonsense_qa': {e}")
            return {}

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        logger.info(f"Preparing initial data for {len(dataset)} samples...")
        
        # --- Step 1: Prepare Prompts & Metadata ---
        all_prompts_map = {}
        all_metadata_map = {}
        initial_ids_to_process = []

        # System Prompt 설정 
        system_content = (
            "Please do your step-by-step reasoning and conclude the best answer following the format like 'Final answer: A' with a single letter (A, B, C, D, or E) "

        )

        for i, example in enumerate(dataset):
            # 데이터셋 ID가 없으면 인덱스를 ID로 사용
            example_id = example.get('id', str(i))
            
            question = example['question']
            choices_data = example['choices']
            choice_letters = choices_data['label']
            choice_texts = choices_data['text']

            formatted_choices = "\n".join([f"({label}) {text}" for label, text in zip(choice_letters, choice_texts)])
            user_content = f"Question: {question}\n\nOptions:{formatted_choices}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            final_prompt = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

            all_prompts_map[example_id] = final_prompt
            
            all_metadata_map[example_id] = {
                'id': example_id,
                'question': question,
                'correct_answer_key': example['answerKey'],
                'choices': formatted_choices
            }
            initial_ids_to_process.append(example_id)

        # --- Step 2: Initialize Loop Variables ---
        final_results_map = {}
        ids_to_process = initial_ids_to_process
        current_pass = 0
        start_time = time.time()

        # --- Step 3: Multi-Pass Inference Loop ---
        while ids_to_process and current_pass < max_passes:
            current_pass += 1
            logger.info(f"--- Starting Pass {current_pass}/{max_passes} for {len(ids_to_process)} samples ---")

            # A) 이번 패스용 프롬프트 준비
            prompts_for_this_pass = [all_prompts_map[eid] for eid in ids_to_process]
            metadata_for_this_pass = [all_metadata_map[eid] for eid in ids_to_process]

            # B) 배치 추론 실행
            # (주의: _run_inference_batch의 리턴 순서가 final, thinking, tokens 인지 확인 필요)
            batch_final_contents, batch_thinking_contents, batch_thinking_token_counts = \
                self._run_inference_batch(prompts_for_this_pass, batch_size)

            ids_failing_this_pass = []

            # C) 결과 처리
            for i, final_content in enumerate(batch_final_contents):
                metadata = metadata_for_this_pass[i]
                current_id = metadata['id']
                thinking_content = batch_thinking_contents[i]

                # 실패 기준: <think> 태그 누락
                is_failure = (thinking_content == "--- No <think> tag found ---")

                if is_failure and current_pass < max_passes:
                    # 실패했고, 재시도 기회가 남았으면 다음 라운드로 넘김
                    ids_failing_this_pass.append(current_id)
                else:
                    # 성공했거나 마지막 기회인 경우 -> 결과 확정
                    # 정답 추출 및 채점
                    predicted_answer = self._extract_answer_from_text(final_content)
                    is_correct = (predicted_answer == metadata['correct_answer_key'])

                    final_results_map[current_id] = {
                        'metadata': metadata,
                        'predicted_answer': predicted_answer,
                        'final_content': final_content,
                        'thinking_content': thinking_content,
                        'thinking_tokens': batch_thinking_token_counts[i],
                        'is_correct': is_correct,
                        'pass_processed': current_pass
                    }

            # D) 다음 루프 준비
            ids_to_process = ids_failing_this_pass
            if ids_to_process:
                logger.info(f"Pass {current_pass} complete. {len(ids_to_process)} samples failed <think> check and will be retried.")
            else:
                logger.info(f"Pass {current_pass} complete. All samples processed successfully.")

        end_time = time.time()
        
        if ids_to_process:
            logger.warning(f"After {max_passes} passes, {len(ids_to_process)} samples still failed. Using last result.")

        # --- Step 4: Finalize & Sort Results ---
        # 원본 데이터셋 순서대로 정렬
        sorted_results = [final_results_map[eid] for eid in initial_ids_to_process if eid in final_results_map]
        
        total_correct = sum(1 for r in sorted_results if r['is_correct'])
        num_samples = len(sorted_results)
        accuracy = total_correct / num_samples if num_samples > 0 else 0
        avg_thinking_tokens = sum(r['thinking_tokens'] for r in sorted_results) / num_samples if num_samples > 0 else 0
        total_time = end_time - start_time

        # 상세 리포트 생성
        detailed_results = []
        for res in sorted_results:
            meta = res['metadata']
            detailed_results.append({
                'id': meta['id'],
                'question': meta['question'],
                'correct_answer_key': meta['correct_answer_key'],
                'predicted_answer': res['predicted_answer'],
                'is_correct': res['is_correct'],
                'thinking_tokens': res['thinking_tokens'],
                'pass_processed': res['pass_processed'],
                'thinking_content': res['thinking_content'],
                'final_content': res['final_content']
            })

        logger.info("CommonsenseQA Evaluation Complete.")
        logger.info(f"   Accuracy: {accuracy:.3f} ({total_correct}/{num_samples})")
        logger.info(f"   Average Thinking Tokens: {avg_thinking_tokens:.2f}")
        logger.info(f"   Total Evaluation Time: {total_time:.2f} seconds")

        results = {
            "dataset": "CommonsenseQA",
            "model": self.model_name,
            "score": accuracy,
            "avg_thinking_tokens": avg_thinking_tokens,
            "time_taken_seconds": total_time,
            "passes_run": current_pass,
            "total_questions": num_samples,
            "detailed_results": detailed_results
        }

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_model_safe = self.model_name.replace('/', '_') 
        self.save_results(results, filename=f"commonsenseqa_multipass_{filename_model_safe}_{timestamp_str}.json")

        return {"accuracy": accuracy, "details": detailed_results}

class BigToSmallCSQAEvaluator(CommonsenseQAEvaluator):
    """
    Big Model이 생성한 추론(<think>)을 입력으로 받아 
    Small Model이 정답(A-E)을 맞히는지 평가하는 클래스 (Distillation 평가)
    """

    def __init__(self, small_model_name: str, big_results_file: str):
        # 1. Small Model 로드 (부모 클래스 init 호출)
        super().__init__(model_name=small_model_name)
        
        # 2. Big Model의 결과 JSON 로드
        self.eval_data = self._load_and_merge_data(big_results_file)
    
    def _load_and_merge_data(self, json_filepath: str) -> List[Dict]:
        """
        [핵심 로직]
        1. Big Model JSON: ID, Question, Thinking Content, Correct Answer 가져오기
        2. CommonsenseQA Dataset: ID를 기준으로 Choices(선택지) 가져오기
        3. 두 데이터를 ID 기준으로 병합
        """
        logger.info(f"1. Loading Big Model results from: {json_filepath}")
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                big_model_data = json.load(f)
                # JSON 구조에 따라 접근 경로 수정 (예: detailed_results 키 사용)
                json_results = big_model_data.get("detailed_results", [])
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise

        logger.info("2. Loading original CommonsenseQA dataset (Validation split)...")
        try:
            # HuggingFace 데이터셋 로드
            csqa_dataset = load_dataset("commonsense_qa", split="validation")
            
            # 검색 속도 향상을 위해 ID를 Key로 하는 Dictionary 생성
            # 예: {'id_123': {row_data}, ...}
            dataset_map = {item['id']: item for item in csqa_dataset}
        except Exception as e:
            logger.error(f"Failed to load CommonsenseQA dataset: {e}")
            raise

        merged_data = []
        logger.info(f"3. Merging {len(json_results)} JSON samples with Dataset choices...")

        for json_item in json_results:
            # 1) Big Model의 생각(Thinking)이 있는 경우만 처리
            if not json_item.get('thinking_content'):
                continue

            item_id = json_item['id']
            
            # 2) 원본 데이터셋에서 해당 ID의 선택지(Choices) 조회
            original_item = dataset_map.get(item_id)
            
            if original_item:
                # 데이터셋에서 선택지 포맷팅: (A) text (B) text ...
                choices_raw = original_item['choices'] # {'label': ['A', 'B'..], 'text': ['txt1', 'txt2'..]}
                formatted_choices = "\n".join(
                    [f"({label}) {text}" for label, text in zip(choices_raw['label'], choices_raw['text'])]
                )
            else:
                # 원본 데이터셋에 없는 경우 (거의 없겠지만 예외처리)
                logger.warning(f"ID {item_id} not found in original dataset. Skipping choice mapping.")
                # JSON에 choices가 있다면 백업으로 사용, 없으면 빈 문자열
                formatted_choices = json_item.get('choices', "")

            correct_ans = json_item.get('correct_answer_key') 
            if not correct_ans:
                correct_ans = json_item.get('correct_answer')

            # 3) 데이터 병합하여 리스트에 추가
            merged_data.append({
                'id': item_id,
                'question': json_item['question'],           # From Big Model JSON
                'thinking_content': json_item['thinking_content'], # From Big Model JSON
                'correct_answer': correct_ans, # From Big Model JSON
                'choices': formatted_choices                 # From CommonsenseQA Dataset
            })

        # ID 순으로 정렬 (일관성을 위해)
        merged_data.sort(key=lambda x: x['id'])
        
        logger.info(f"Successfully merged {len(merged_data)} samples.")
        return merged_data

    def save_results(self, results: Dict, filename: str):
        output_dir = "evaluation_results_csqa_bigTosmall"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results successfully saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save results to {filepath}: {e}")

    def evaluate(self, max_samples: int = None, batch_size: int = 16):
        logger.info(f"Starting CommonsenseQA big To small Evaluation: {self.model_name}")

        # 1. 평가 데이터 준비
        target_samples = self.eval_data
        if max_samples:
            target_samples = target_samples[:max_samples]

        logger.info(f"Processing {len(target_samples)} samples...")

        # 2. 프롬프트 구성
        # Small Model에게는 "질문 + 선택지 + Big Model의 생각"을 주고 "답만 말해"라고 지시
        prompts = []
        
        system_prompt = (
            "You will be given a commonsense question, options, and a guide. "
            "Your task is to provide ONLY the final correct option letter . "
            "Do not generate any new reasoning. End your response strictly with 'Final answer: A' with a single letter (A, B, C, D, or E)"
        )

        for sample in target_samples:
            # CSQA는 선택지(Options)가 프롬프트에 포함되어야 함
            user_content = (
                f"Question: {sample['question']}\n\n"
                f"Options:\n{sample['choices']}\n\n"
                f"<think>\n{sample['thinking_content']}\n</think>\n\n"
                #"Based on the reasoning inside the <think> tags, select the best option."
            )
            
            full_prompt = self.hf_tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, 
                 {"role": "user", "content": user_content}],
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=True 
            )
            prompts.append(full_prompt)

        # 3. Small Model 추론 (Batch)
        all_outputs = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        logger.info(f"Starting inference in {total_batches} batches (Batch Size: {batch_size})...")
        start_time = time.time()

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            if i % (batch_size * 5) == 0:
                logger.info(f"Processing batch {i // batch_size + 1}/{total_batches}...")
            
            batch_outputs = self.llm.generate(batch_prompts, self.sampling_params)
            all_outputs.extend(batch_outputs)

        end_time = time.time()
        logger.info("Inference complete.")

        # 4. 채점 및 결과 정리
        detailed_results = []
        correct_count = 0
        
        for i, output in enumerate(all_outputs):
            generated_text = output.outputs[0].text.strip()
            sample = target_samples[i]
            
            # 정답 추출 (A-E) - 부모 클래스의 메서드 활용하거나 여기서 직접 구현
            # 여기서는 부모 클래스의 _extract_answer_from_text 재사용 (thinking content는 비워둠)
            pred = self._extract_answer_from_text(generated_text) 
            
            # CSQA 정답 비교 (문자열 일치)
            is_correct = (pred == sample['correct_answer'])
            if is_correct: 
                correct_count += 1

            small_tokens = self._count_tokens(generated_text)
            
            # 전체 입력 컨텍스트 (저장용)
            actual_input_context = (
                f"Question: {sample['question']}\n"
                f"Options: {sample['choices']}\n"
                f"<think>{sample['thinking_content']}</think>"
            )

            detailed_results.append({
                'id': sample['id'],
                'question': sample['question'],
                'full_input_context': actual_input_context,
                'correct_answer': sample['correct_answer'],
                'predicted_answer': pred,
                'is_correct': is_correct,
                'small_model_output': generated_text,
                'small_model_tokens': small_tokens, 
                'big_model_thought_tokens': self._count_tokens(sample['thinking_content'])
            })

        # 5. 통계 및 저장
        num_samples = len(detailed_results)
        accuracy = correct_count / num_samples if num_samples > 0 else 0
        total_small_tokens = sum(r['small_model_tokens'] for r in detailed_results)
        avg_small_tokens = total_small_tokens / num_samples if num_samples > 0 else 0
        total_time = end_time - start_time

        logger.info("CSQA Distillation Evaluation Complete.")
        logger.info(f"   Accuracy: {accuracy:.3f} ({correct_count}/{num_samples})")
        logger.info(f"   Avg Small Model Tokens: {avg_small_tokens:.2f}")
        logger.info(f"   Total Time: {total_time:.2f} s")

        results = {
            "model": self.model_name,
            "type": "Big-to-Small_CSQA",
            "accuracy": accuracy,
            "total_samples": num_samples,
            "avg_small_model_tokens": avg_small_tokens,
            "correct_count": correct_count,
            "time_taken": total_time,
            "detailed_results": detailed_results,
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self.model_name.replace('/', '_')
        filename = f"distill_csqa_{safe_name}_{timestamp}.json"
        
        self.save_results(results, filename)
        return results

if __name__ == "__main__":
    # --- 설정 ---
    
    # 1. Big Model이 생성한 결과 파일 경로 (앞서 실행한 CommonsenseQA 결과 JSON)
    # 반드시 'choices' 필드가 포함된 JSON이어야 합니다.
    #BIG_MODEL_RESULT_FILE = "evaluation_results_csqa/csqa_results_think_Qwen_Qwen3-32B_20251102_XXXXXX.json"
    # 32B
    BIG_MODEL_RESULT_FILE = "evaluation_results_commonsense_multipass/commonsenseqa_multipass_Qwen_Qwen3-32B_20251124_121151.json"
    # 14B
    #BIG_MODEL_RESULT_FILE = "evaluation_results_commonsense_multipass/commonsenseqa_multipass_Qwen_Qwen3-14B_20251123_122139.json"
    # 2. 평가할 Small Model (예: 8B, 7B 등)
    TARGET_SMALL_MODEL = "Qwen/Qwen3-8B"
    
    logger.info(f"=== Starting CSQA Distillation for: {TARGET_SMALL_MODEL} ===")
    
    # 평가기 초기화 및 실행
    # (주의: BigToSmallCSQAEvaluator는 CommonsenseQAEvaluator 클래스가 정의되어 있어야 합니다)
    evaluator = BigToSmallCSQAEvaluator(TARGET_SMALL_MODEL, BIG_MODEL_RESULT_FILE)
    evaluator.evaluate(max_samples=None, batch_size=16)
            
    logger.info("=== All Evaluations Completed ===")