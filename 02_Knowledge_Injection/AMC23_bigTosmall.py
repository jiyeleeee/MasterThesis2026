import os
import re
import json
import time
import gc
from typing import List, Dict, Tuple, Optional
from vllm import LLM, SamplingParams
from datasets import load_dataset
from datetime import datetime
from transformers import AutoTokenizer
import logging
import torch  # 
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel



# 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMC23Evaluator:
    
    def __init__(self, model_name: str = "Qwen/Qwen3-32B"):
        """Initializes the evaluator with a specified vLLM model."""
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
            trust_remote_code=True,
            dtype="auto"
        )
        
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens = 16384,
            stop=["</s>","<|im_end|>", "<|endoftext|>", "(END)", "\\[END\\]", "**END**", "[END]"]
        )
        logger.info("Model and tokenizer initialization complete!")

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given string."""
        if not text:
            return 0
        return len(self.hf_tokenizer.encode(text))

    def _is_completely_wrapped_by_text(self, input_string: str) -> Optional[str]:
        pattern = r'^\\text{(.*)}$'
        match = re.match(pattern, input_string)
        if match:
            extracted_content = match.group(1).replace("(", "").replace(")", "").replace(",", "")
            return extracted_content
        return None

    def _math_answer_cleaning(self, answer: str) -> str:
        extracted_content = self._is_completely_wrapped_by_text(answer)
        answer = extracted_content if extracted_content else answer
        answer = answer.replace(",\\!", "").replace("{,}", "").replace("\\$", "")
        answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
        answer = answer.replace("^\\circ", "").replace("^{\\circ}", "")
        answer = answer.replace("\\quad", "")
        answer = re.sub(r'\\,\\text\{.*?\}', '', answer)
        answer = re.sub(r'\\text\{.*?\}', '', answer)
        answer = re.sub(r'(\s\^\{-\d+\})', '', answer)
        answer = answer.replace(" ", "").replace("\n", "").replace("\\n", "")
        answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}', r'\1e\2', answer)
        answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)', r'\1e\2', answer)
        answer = re.sub(r'(\d+)\^{(\d+)}', r'\1^\2', answer)
        answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
        answer = answer.replace(",", "")
        answer = answer.lower()
        if answer.endswith("\\"):
            answer = answer[:-1]
        func_pattern = r'^[a-zA-Z_]\w*\([a-zA-Z_]\w*\)$'
        if "=" in answer and (re.match(func_pattern, answer.split("=")[0]) or len(answer.split("=")[0]) <= 3):
            answer = answer.split("=", 1)[1]
        return answer

    def _calculate_numbers(self, input_string: str) -> Optional[float]:
        if not isinstance(input_string, str): return None
        evaluable_string = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', input_string)
        try:
            if not re.match(r'^[\d\s\(\)\/\.\+\-\*eE]+$', evaluable_string):
                return None
            result = eval(evaluable_string)
            return float(result)
        except Exception:
            return None

    def _is_equal_after_calculation(self, extracted_answer: str, gold: str) -> bool:
        gold_result = self._calculate_numbers(gold)
        extracted_answer_result = self._calculate_numbers(extracted_answer)
        if gold_result is not None and extracted_answer_result is not None:
            return abs(gold_result - extracted_answer_result) < 1e-6
        return False

    def _run_inference_batch(self, prompts: List[str], batch_size: int) -> Tuple[List[str], List[str], List[int]]:
        """
        Runs inference and parses the output into thinking/final parts.
        """
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
        return all_thinking_contents, all_final_contents, all_thinking_token_counts
    
    def _extract_answer_from_text(self, final_content: str, thinking_content: str) -> str:
        full_text = thinking_content + "\n" + final_content
        match = re.search(r"Final Answer:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", full_text)
        if match:
            return match.group(1).strip()
        patterns_to_try = [
            r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
            r"answer is\s*([-+]?\d*\.?\d+)",
            r"answer should be\s*([-+]?\d*\.?\d+)", # check
            r"\*\*(.*?)\*\*",
            r"\\\[\n(.*?)\n\\\]",
            r'is \\\((.*?)\\\)',
            r"\\\[\\n(.*?)\\n\\\]"
        ]

        # modified
        for pattern in patterns_to_try:
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            if match:
                potential_answer = match.group(1)
                num_match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", potential_answer)
                if num_match:
                    return num_match.group(1).strip()

        numerical_fallback = re.findall(r'[-+]?\d*\.?\d+', full_text)
        if numerical_fallback:
            return numerical_fallback[-1].strip()
        return ""

    def _is_correct(self, pred: str, gold: str) -> bool:
        cleaned_pred = self._math_answer_cleaning(pred)
        cleaned_gold = self._math_answer_cleaning(gold)
        if cleaned_pred == cleaned_gold:
            return True
        if self._is_equal_after_calculation(cleaned_pred, cleaned_gold):
            return True
        return False
    
    def save_results(self, results: Dict, filename: str):
        output_dir = "evaluation_results_amc23_4"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results successfully saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save results to {filepath}: {e}")

    #  2-pass
    def evaluate(self, max_samples: int = None, batch_size: int = 16, max_passes: int = 3, exclude_ids: Optional[List[int]] = None):
        """
        Main method to run the evaluation with automatic retries for failed samples.
        """
        logger.info(f"Starting AMC23 Accuracy Evaluation with {max_passes}-Pass Retry Logic")
        logger.info("="*50)

        try:
            dataset = load_dataset("zwhe99/amc23", split="test")
        except Exception as e:
            logger.error(f"Failed to load dataset 'zwhe99/amc23': {e}")
            return {}
        
        if exclude_ids:
            logger.info(f"Excluding {len(exclude_ids)} specific IDs: {exclude_ids}")
            exclude_id_set = set(exclude_ids)
            dataset = dataset.filter(lambda example: example['id'] not in exclude_id_set)
            logger.info(f"Dataset size after exclusion: {len(dataset)} samples.")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        #  
        logger.info(f"Preparing initial data for {len(dataset)} samples...")
        all_prompts_map = {}
        all_metadata_map = {}
        
        system_prompt = (
            "You are a math expert. Solve the following math problem. "
            "First, provide your step-by-step reasoning within <think> and </think> tags using this 2-step structure:\n"
            "- Briefly describe your approach.\n"
            "- Provide the detailed, step-by-step calculation.\n\n"
            "After the </think> tag, Have to state the final numerical answer with 'Final Answer:'."
        )
        
        initial_ids_to_process = []
        
        for example in dataset:
            question = example['question']
            example_id = example.get('id') 
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem: {question}"}
            ]
            
            final_prompt = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # key parameter

            )

            all_prompts_map[example_id] = final_prompt
            all_metadata_map[example_id] = {
                'id': example_id,
                'question': question,
                'correct_answer': str(example['answer']).strip()
            }
            initial_ids_to_process.append(example_id)
        
        # 
        final_detailed_results = {} # Map[id, result_dict] 
        ids_to_process = initial_ids_to_process
        current_pass = 0
        start_time = time.time()

        #  
        while ids_to_process and current_pass < max_passes:
            current_pass += 1
            logger.info(f"--- Starting Pass {current_pass}/{max_passes} for {len(ids_to_process)} samples ---")

            # A) 
            prompts_for_this_pass = [all_prompts_map[id] for id in ids_to_process]
            metadata_for_this_pass = [all_metadata_map[id] for id in ids_to_process]

            # B) 
            all_thinking_contents, all_final_contents, all_thinking_token_counts = \
                self._run_inference_batch(prompts_for_this_pass, batch_size)
            
            ids_failing_this_pass = [] # 

            # C) 
            for i, final_content in enumerate(all_final_contents):
                metadata = metadata_for_this_pass[i]
                current_id = metadata['id']
                thinking_content = all_thinking_contents[i]
                
                # 
                is_failure = (thinking_content == "--- No <think> tag found ---")
                
                # D)
                # 
                if is_failure and current_pass < max_passes:
                    ids_failing_this_pass.append(current_id)
                else:
                    # 
                    # 
                    predicted_answer = self._extract_answer_from_text(final_content, thinking_content)
                    correct_answer = metadata['correct_answer']
                    is_correct = self._is_correct(predicted_answer, correct_answer)
                    
                    result_detail = {
                        'id': current_id,
                        'question': metadata['question'],
                        'correct_answer': correct_answer,
                        'predicted_answer': predicted_answer,
                        'thinking_content': thinking_content, 
                        'final_content': final_content, 
                        'thinking_tokens': all_thinking_token_counts[i],
                        'is_correct': is_correct,
                        'pass_processed': current_pass # 
                    }
                    final_detailed_results[current_id] = result_detail

            # E) 
            ids_to_process = ids_failing_this_pass
            if ids_to_process:
                logger.info(f"Pass {current_pass} complete. {len(ids_to_process)} samples failed <think> check and will be retried.")
            else:
                logger.info(f"Pass {current_pass} complete. All samples passed <think> check.")

        end_time = time.time()
        
        #  
        if ids_to_process:
            logger.warning(f"After {max_passes} passes, {len(ids_to_process)} samples still had <think> tag issues. They are included with their last failed state.")

        # 
        detailed_results = list(final_detailed_results.values())
        detailed_results.sort(key=lambda x: x['id']) 

        total_correct = sum(1 for r in detailed_results if r['is_correct'])
        total_thinking_tokens = sum(r['thinking_tokens'] for r in detailed_results)
        num_samples = len(detailed_results)
        
        # 
        if num_samples != len(dataset):
             logger.error(f"Result count mismatch! Expected {len(dataset)} results, but got {num_samples}.")

        accuracy = total_correct / num_samples if num_samples > 0 else 0
        avg_thinking_tokens = total_thinking_tokens / num_samples if num_samples > 0 else 0
        
        logger.info("AMC23 Accuracy Evaluation Complete.")
        logger.info(f"   Accuracy: {accuracy:.3f} ({total_correct}/{num_samples})")
        logger.info(f"   Average Thinking Tokens: {avg_thinking_tokens:.2f}")
        logger.info(f"   Total Evaluation Time: {end_time - start_time:.2f} seconds")
        
        results = {
            "model": self.model_name,
            "dataset": "AMC23",
            "accuracy": accuracy,
            "correct_answers_count": total_correct,
            "total_questions": num_samples,
            "avg_thinking_tokens": avg_thinking_tokens,
            "time_taken_seconds": end_time - start_time,
            "detailed_results": detailed_results
        }
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_model_safe = self.model_name.replace('/', '_') 
        self.save_results(results, filename=f"amc23_results_think_{filename_model_safe}_{timestamp_str}.json")
        
        return results

# ---------------------------------------------------------------------------
# --- Big to small and small to big ---
# ---------------------------------------------------------------------------

class BigToSmallEvaluator(AMC23Evaluator):
   
    def __init__(self, small_model_name: str, big_results_file: str):
        # load Small Model
        super().__init__(model_name=small_model_name)
        # load jsonfile generated by big  (질문, 생각, 정답)
        self.eval_data = self._load_all_data_from_json(big_results_file)

    def _load_all_data_from_json(self, filepath: str) -> List[Dict]:
        """
        JSON 파일에서 필요한 모든 데이터(질문, 생각, 정답)를 한 번에 리스트로 로드합니다.
        """
        logger.info(f"Loading all evaluation data from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            loaded_samples = []
            for item in data.get("detailed_results", []):
                # 
                if item.get('thinking_content') :
                    loaded_samples.append({
                        'id': item['id'],
                        'question': item['question'],
                        'thinking_content': item['thinking_content'], # 
                        'correct_answer': item['correct_answer']      # 
                    })
            
            # 
            loaded_samples.sort(key=lambda x: x['id'])
            
            logger.info(f"Successfully loaded {len(loaded_samples)} samples from JSON.")
            return loaded_samples
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise

    def save_results(self, results: Dict, filename: str):
        output_dir = "evaluation_results_amc23_bigTosmall_14B" # 원하시는 폴더명####
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results successfully saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save results to {filepath}: {e}")

    def evaluate(self, max_samples: int = None, batch_size: int = 16, exclude_ids: List[int] = None):
        logger.info(f"Starting Distillation Evaluation: {self.model_name}")

        # 1. 
        target_samples = self.eval_data
        
        if exclude_ids:
            target_samples = [s for s in target_samples if s['id'] not in exclude_ids]
            
        if max_samples:
            target_samples = target_samples[:max_samples]

        logger.info(f"Processing {len(target_samples)} samples...")

        # 2. 
        prompts = []
        system_prompt = (
            "You will be given a math problem and a step-by-step reasoning process with guide "
            "Your task provide ONLY the final numerical answer. "
            "Do not repeat the reasoning. "
            "End your response strictly with 'Final Answer: X'."
        )

        for sample in target_samples:
            #question = sample['question']
            #raw_thought = sample['thinking_content']
            
            user_content = (
                f"{sample['question']}\n\n"
                f"<think>\n{sample['thinking_content']}\n</think>\n\n"
                #"Based on the reasoning inside the <think> tags, calculate and provide the final answer."
            )
            
            #messages = [
            #    {"role": "system", "content": system_prompt},
            #    {"role": "user", "content": user_content}
            #]
            
            full_prompt = self.hf_tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_content}],
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=True # key parameter

            )
            prompts.append(full_prompt)


        # 3. Small Model 

        all_outputs = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        logger.info(f"Starting inference in {total_batches} batches (Batch Size: {batch_size})...")
        start_time = time.time()

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{total_batches}...")
            
            # 
            batch_outputs = self.llm.generate(batch_prompts, self.sampling_params)
            all_outputs.extend(batch_outputs)

        end_time = time.time()
        logger.info("Inference complete.")


        # 4. 
        detailed_results = []
        correct_count = 0
        
        for i, output in enumerate(all_outputs):
            generated_text = output.outputs[0].text.strip()
            sample = target_samples[i]
            
            # 
            pred = self._extract_answer_from_text(generated_text, "")
            is_correct = self._is_correct(pred, sample['correct_answer'])
            
            if is_correct: correct_count += 1


            small_tokens = self._count_tokens(generated_text)

            actual_input_context = (
                f"{sample['question']}\n\n"
                f"<think>\n{sample['thinking_content']}\n</think>"
            )

            detailed_results.append({
                'id': sample['id'],
                'question': sample['question'],
                'full_input_context': actual_input_context,
                'correct_answer': sample['correct_answer'],
                'predicted_answer': pred,
                'is_correct': is_correct,

                'small_model_output': generated_text,
                'small_model_tokens': small_tokens, # Small Model 토큰 수 저장
                
                'big_model_thought_tokens': self._count_tokens(sample['thinking_content'])
            })


        # 5. 
        num_samples = len(detailed_results)
        accuracy = correct_count / len(detailed_results) if detailed_results else 0
        
        # 

        total_small_tokens = sum(r['small_model_tokens'] for r in detailed_results)
        avg_small_tokens = total_small_tokens / num_samples if num_samples > 0 else 0

        
        total_time = end_time - start_time

        # --- [Log Output] ---
        logger.info("AMC23 Accuracy Evaluation Complete.")
        logger.info(f"   Accuracy: {accuracy:.3f} ({correct_count}/{num_samples})")
        logger.info(f"   Average Thinking Tokens: {avg_small_tokens:.2f}")
        logger.info(f"   Total Evaluation Time: {total_time:.2f} seconds")


        results = {
            "model": self.model_name,
            "type": "Big-to-Small_amc23",
            "accuracy": accuracy,
            "total_samples": len(detailed_results),
            "avg_small_model_tokens": avg_small_tokens,
            "correct_count": correct_count,
            "time_taken": total_time,
            "detailed_results": detailed_results,
        }
        
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_time = datetime.now().strftime("%H%M%S")
        safe_name = self.model_name.replace('/', '_')
        filename = f"distill_json_{safe_name}_{current_time}.json"
        
        self.save_results(results, filename)
        
        return results


if __name__ == "__main__":
    # 
    #32B
    #BIG_MODEL_FILE ="AMC23_remove answers from result/amc23_results_think_Qwen_Qwen3-32B_20251214_180434.json"

    #14B
    BIG_MODEL_FILE = 'AMC23_remove answers from result/amc23_results_think_Qwen_Qwen3-14B_20251212_234333.json'

    #BIG_MODEL_FILE = 'Transferability_of_Reasoning_Traces/AMC23_remove_answers_from_original_result/amc23_results_think_Qwen_Qwen3-14B_20251212_234333.json'
    #Small_model_file = ""
    #Small_model_file = ""
    
    
    # 2. 
    target_model = "Qwen/Qwen3-8B" 
    logger.info(f"=== Starting Evaluation for: {target_model} ===")
    
    evaluator = BigToSmallEvaluator(target_model, BIG_MODEL_FILE)
    evaluator.evaluate(max_samples=None, batch_size=5)
            
    logger.info("=== All Evaluations Completed ===")