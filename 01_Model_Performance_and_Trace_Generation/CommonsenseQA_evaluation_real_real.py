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

# Configure logging
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
            tensor_parallel_size=2, 
            gpu_memory_utilization=0.9,
            max_model_len=3072, 
            trust_remote_code=True,
            dtype="auto" 
        )
        
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_tokenizer.pad_token_id = 151643

        #self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    # 
    # 
    #  
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
        # 
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

        # 1. [Strong Pattern] 
        strong_pattern = r"(?:Final\s*Answer|Answer|Best\s*Answer|Correct\s*Answer)(?:\s*is)?[:.\s]*([A-E])"
        all_matches = re.findall(strong_pattern, cleaned_text, re.IGNORECASE)

        if all_matches:
            return all_matches[-1].strip().upper()

        # 2. [Weak Pattern] 
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

        # System Prompt  
        system_content = (
            "Please do your step-by-step reasoning and conclude the best answer following the format like 'Final answer: A' with a single letter (A, B, C, D, or E) "

        )

        for i, example in enumerate(dataset):
            # 
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

            # A) 
            prompts_for_this_pass = [all_prompts_map[eid] for eid in ids_to_process]
            metadata_for_this_pass = [all_metadata_map[eid] for eid in ids_to_process]

            # B) 
            # 
            batch_final_contents, batch_thinking_contents, batch_thinking_token_counts = \
                self._run_inference_batch(prompts_for_this_pass, batch_size)

            ids_failing_this_pass = []

            # C) 
            for i, final_content in enumerate(batch_final_contents):
                metadata = metadata_for_this_pass[i]
                current_id = metadata['id']
                thinking_content = batch_thinking_contents[i]

                # 
                is_failure = (thinking_content == "--- No <think> tag found ---")

                if is_failure and current_pass < max_passes:
                    # 
                    ids_failing_this_pass.append(current_id)
                else:
                    # 
                    # 
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

            # D) 
            ids_to_process = ids_failing_this_pass
            if ids_to_process:
                logger.info(f"Pass {current_pass} complete. {len(ids_to_process)} samples failed <think> check and will be retried.")
            else:
                logger.info(f"Pass {current_pass} complete. All samples processed successfully.")

        end_time = time.time()
        
        if ids_to_process:
            logger.warning(f"After {max_passes} passes, {len(ids_to_process)} samples still failed. Using last result.")

        # --- Step 4: Finalize & Sort Results ---
        # 
        sorted_results = [final_results_map[eid] for eid in initial_ids_to_process if eid in final_results_map]
        
        total_correct = sum(1 for r in sorted_results if r['is_correct'])
        num_samples = len(sorted_results)
        accuracy = total_correct / num_samples if num_samples > 0 else 0
        avg_thinking_tokens = sum(r['thinking_tokens'] for r in sorted_results) / num_samples if num_samples > 0 else 0
        total_time = end_time - start_time

        # 
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

if __name__ == "__main__":
    evaluator = CommonsenseQAEvaluator(model_name="Qwen/Qwen3-32B") 
    evaluator.evaluate(max_samples=None, batch_size=10, max_passes=5)

    