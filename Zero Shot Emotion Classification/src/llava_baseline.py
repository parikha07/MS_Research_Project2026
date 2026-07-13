import os
import random
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from transformers import AutoProcessor
import json
import argparse
from PIL import Image
from pathlib import Path
from vllm import LLM, SamplingParams
import torch
from huggingface_hub import login
from tqdm import tqdm
import logging
import pandas as pd
from collections import Counter

# Set up logging to suppress vLLM progress bars
logging.getLogger("vllm").setLevel(logging.WARNING)
tqdm_kwargs = dict(file=sys.stdout)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

login("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")  # Replace with your actual Hugging Face token

# Label mappings for MSED tasks
sentiment_map = {"a": "positive", "b": "negative", "c": "neutral"}
emotion_map = {
    "a": "happiness", "b": "sad", "c": "neutral",
    "d": "disgust", "e": "anger", "f": "fear"
}
desire_map = {
    "a": "vengeance", "b": "curiosity", "c": "social-contact",
    "d": "family", "e": "tranquility", "f": "romance", "g": "none"
}

@dataclass
class PredictionResult:
    task: str
    reasoning: str
    prediction: str
    label: str
    ground_truth: Optional[str] = None
    sample_id: Optional[int] = None
    input_text: Optional[str] = None
    image_path: Optional[str] = None
    raw_llm_output: Optional[str] = None

class MSEDAnalyzer:
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", 
                 seed: int = 42, debug_mode: bool = False):
        self.model_name = model_name
        self.debug_mode = debug_mode
        
        # Set random seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize processor for LLaVA-1.6
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Initialize vLLM engine with proper configuration for LLaVA-1.6
        self.llm = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=False,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        
        # Sampling parameters for chain-of-thought
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=["</s>", "USER:", "ASSISTANT:"]
        )
    
    def load_and_validate_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and validate an image"""
        try:
            image = Image.open(image_path).convert("RGB")
            # Validate image size
            if image.size[0] * image.size[1] > 4096 * 4096:  # Resize if too large
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            return None
    
    def create_text_only_prompt(self, text: str, task: str) -> Dict:
        """Create prompt for text-only analysis"""
        
        if task == "sentiment":
            task_desc = """Analyze the sentiment of this text.
    
**Classification Options:**
a = positive
b = negative  
c = neutral"""
        elif task == "emotion":
            task_desc = """Analyze the primary emotion expressed in this text.
    
**Classification Options:**
a = happiness
b = sad
c = neutral
d = disgust
e = anger
f = fear"""
        elif task == "desire":
            task_desc = """Analyze the underlying desire or need expressed in this text.
    
**Classification Options:**
a = vengeance
b = curiosity
c = social-contact
d = family
e = tranquility
f = romance
g = none"""
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""{task_desc}

Text: "{text}"

**Instructions:**
First, explain your reasoning step-by-step. Then provide your final answer as a single letter.

**Format your response exactly as:**
Reasoning: [your analysis]
Final Answer: [letter]"""}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        if self.debug_mode:
            print(f"\n🎯 Text-only {task} prompt created")
            print(f"   Text: {text[:100]}...")
        
        return {"prompt": prompt_text, "multi_modal_data": None}
    
    def create_image_only_prompt(self, image_path: str, task: str) -> Optional[Dict]:
        """Create prompt for image-only analysis"""
        
        image = self.load_and_validate_image(image_path)
        if image is None:
            return None
        
        if task == "sentiment":
            task_desc = """Analyze the sentiment conveyed by this image.
    
**Classification Options:**
a = positive
b = negative  
c = neutral"""
        elif task == "emotion":
            task_desc = """Analyze the primary emotion expressed in this image.
    
**Classification Options:**
a = happiness
b = sad
c = neutral
d = disgust
e = anger
f = fear"""
        elif task == "desire":
            task_desc = """Analyze what desire or need this image represents.
    
**Classification Options:**
a = vengeance
b = curiosity
c = social-contact
d = family
e = tranquility
f = romance
g = none"""
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""{task_desc}

**Instructions:**
First, explain your reasoning step-by-step. Then provide your final answer as a single letter.

**Format your response exactly as:**
Reasoning: [your analysis]
Final Answer: [letter]"""}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        if self.debug_mode:
            print(f"\n🎯 Image-only {task} prompt created")
            print(f"   Image: {image_path}")
        
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image}
        }
    
    def create_multimodal_prompt(self, text: str, image_path: str, task: str) -> Optional[Dict]:
        """Create prompt for multimodal analysis"""
        
        image = self.load_and_validate_image(image_path)
        if image is None:
            return None
        
        if task == "sentiment":
            task_desc = """Analyze the combined sentiment of this image and text together.
    
**Classification Options:**
a = positive
b = negative  
c = neutral"""
        elif task == "emotion":
            task_desc = """Analyze the primary emotion expressed by combining this image and text.
    
**Classification Options:**
a = happiness
b = sad
c = neutral
d = disgust
e = anger
f = fear"""
        elif task == "desire":
            task_desc = """Analyze what desire or need is expressed by combining this image and text.
    
**Classification Options:**
a = vengeance
b = curiosity
c = social-contact
d = family
e = tranquility
f = romance
g = none"""
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""{task_desc}

Text: "{text}"

**Instructions:**
First, explain your reasoning step-by-step. Then provide your final answer as a single letter.

**Format your response exactly as:**
Reasoning: [your analysis]
Final Answer: [letter]"""}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        if self.debug_mode:
            print(f"\n🎯 Multimodal {task} prompt created")
            print(f"   Text: {text[:100]}...")
            print(f"   Image: {image_path}")
        
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image}
        }
    
    def extract_label_from_text(self, text: str) -> str:
        """Extract label from generated text with multiple strategies"""
        
        if not text:
            return 'x'
        
        # Clean the text
        text = text.strip().lower()
        
        # Strategy 1: Check if the text is just a single letter
        if len(text) == 1 and text in 'abcdefg':
            return text
        
        # Strategy 2: Look for the last valid letter in the text (most likely the final answer)
        valid_letters = []
        for char in reversed(text):
            if char in 'abcdefg':
                valid_letters.append(char)
        
        if valid_letters:
            return valid_letters[0]  # Return the last valid letter found
        
        # Strategy 3: Look for patterns like "Response: a" or "Answer: b"
        import re
        
        # Pattern for "Response: X" or "Answer: X" or just ": X"
        response_pattern = r'(?:response|answer|result)?\s*:?\s*([abcdefg])'
        matches = re.findall(response_pattern, text)
        if matches:
            return matches[-1]  # Return the last match
        
        # Strategy 4: Look for the letter after common phrases
        common_phrases = [
            r'the answer is\s*([abcdefg])',
            r'i choose\s*([abcdefg])',
            r'my response is\s*([abcdefg])',
            r'therefore\s*([abcdefg])',
            r'so\s*([abcdefg])',
            r'thus\s*([abcdefg])',
            r'final answer\s*:?\s*([abcdefg])',
            r'conclusion\s*:?\s*([abcdefg])'
        ]
        
        for pattern in common_phrases:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1]  # Return the last match
        
        # Strategy 5: Look for standalone letters (word boundaries)
        standalone_pattern = r'\b([abcdefg])\b'
        matches = re.findall(standalone_pattern, text)
        if matches:
            return matches[-1]  # Return the last standalone letter
        
        # Strategy 6: If text contains explanations, look for the letter at the end
        # Split by common separators and check each part
        parts = re.split(r'[.,!?;:\n\t\s]+', text)
        for part in reversed(parts):
            if part and len(part) == 1 and part in 'abcdefg':
                return part
        
        # Strategy 7: Look for the first valid letter as fallback
        for char in text:
            if char in 'abcdefg':
                return char
        
        # Default fallback
        return 'x'
    
    def generate_prediction(self, prompt_data: Dict, task: str, 
                          sample_id: int = None, input_text: str = None, 
                          image_path: str = None) -> PredictionResult:
        """Generate prediction with reasoning"""
        
        try:
            # Check if this is text-only or multimodal
            is_multimodal = prompt_data.get("multi_modal_data") is not None
            
            if is_multimodal:
                outputs = self.llm.generate(
                    [prompt_data],
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
            else:
                outputs = self.llm.generate(
                    [prompt_data["prompt"]],
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
            
            generated_text = outputs[0].outputs[0].text.strip()
            raw_llm_output = generated_text
            
            # Parse reasoning and prediction
            reasoning = ""
            
            # Look for "Final Answer:" pattern
            if "Final Answer:" in generated_text:
                parts = generated_text.split("Final Answer:")
                if len(parts) >= 2:
                    reasoning = parts[0].strip()
                    if reasoning.startswith("Reasoning:"):
                        reasoning = reasoning[10:].strip()
                    
                    answer_part = parts[1].strip()
                    prediction = self.extract_label_from_text(answer_part)
            else:
                reasoning = generated_text
                if reasoning.startswith("Reasoning:"):
                    reasoning = reasoning[10:].strip()
                
                prediction = self.extract_label_from_text(generated_text)
            
            # Convert to label using the convert_prediction_to_label method
            label = self.convert_prediction_to_label(prediction, task)
            
            if self.debug_mode:
                print(f"\n📤 MODEL OUTPUT for {task}:")
                print(f"   Raw: {raw_llm_output[:200]}...")
                print(f"   Prediction: {prediction} → {label}")
            
            return PredictionResult(
                task=task,
                reasoning=reasoning,
                prediction=prediction,
                label=label,
                sample_id=sample_id,
                input_text=input_text,
                image_path=image_path,
                raw_llm_output=raw_llm_output
            )
                
        except Exception as e:
            print(f"❌ Error in prediction for {task}: {e}")
            default_pred = 'g' if task == "desire" else 'c'
            default_label = self.convert_prediction_to_label(default_pred, task)
            
            return PredictionResult(
                task=task,
                reasoning=f"Error occurred: {str(e)}",
                prediction=default_pred,
                label=default_label,
                sample_id=sample_id,
                input_text=input_text,
                image_path=image_path,
                raw_llm_output=f"Error occurred: {str(e)}"
            )
    
    def convert_prediction_to_label(self, prediction: str, task: str) -> str:
        """Convert letter prediction to human-readable label"""
        
        if task == "sentiment":
            return sentiment_map.get(prediction, "neutral")
        elif task == "emotion":
            return emotion_map.get(prediction, "neutral")
        elif task == "desire":
            return desire_map.get(prediction, "none")
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def analyze_text_only(self, texts: List[str], 
                         sample_ids: List[int] = None) -> Dict[str, List[PredictionResult]]:
        """Perform text-only analysis for all tasks"""
        
        if sample_ids is None:
            sample_ids = list(range(len(texts)))
        
        results = {"sentiment": [], "emotion": [], "desire": []}
        total_operations = len(texts) * 3
        
        with tqdm(total=total_operations, desc="Text-Only Analysis", unit="prediction", **tqdm_kwargs) as pbar:
            for i, (text, sample_id) in enumerate(zip(texts, sample_ids)):
                for task in ["sentiment", "emotion", "desire"]:
                    prompt_data = self.create_text_only_prompt(text, task)
                    
                    result = self.generate_prediction(
                        prompt_data, task, sample_id, text, None
                    )
                    
                    results[task].append(result)
                    
                    pbar.set_postfix({
                        'Sample': f"{i+1}/{len(texts)}",
                        'Task': task,
                        'Result': f"{result.prediction}→{result.label}"
                    })
                    pbar.update(1)
        
        return results
    
    def analyze_image_only(self, image_paths: List[str], 
                          sample_ids: List[int] = None) -> Dict[str, List[PredictionResult]]:
        """Perform image-only analysis for all tasks"""
        
        if sample_ids is None:
            sample_ids = list(range(len(image_paths)))
        
        results = {"sentiment": [], "emotion": [], "desire": []}
        total_operations = len(image_paths) * 3
        
        with tqdm(total=total_operations, desc="Image-Only Analysis", unit="prediction", **tqdm_kwargs) as pbar:
            for i, (image_path, sample_id) in enumerate(zip(image_paths, sample_ids)):
                for task in ["sentiment", "emotion", "desire"]:
                    prompt_data = self.create_image_only_prompt(image_path, task)
                    
                    if prompt_data is None:
                        default_pred = 'g' if task == "desire" else 'c'
                        default_label = desire_map.get(default_pred) if task == "desire" else "neutral"
                        
                        result = PredictionResult(
                            task=task,
                            reasoning="Error: Could not load image",
                            prediction=default_pred,
                            label=default_label,
                            sample_id=sample_id,
                            input_text=None,
                            image_path=image_path,
                            raw_llm_output="Error: Could not load image"
                        )
                    else:
                        result = self.generate_prediction(
                            prompt_data, task, sample_id, None, image_path
                        )
                    
                    results[task].append(result)
                    
                    pbar.set_postfix({
                        'Sample': f"{i+1}/{len(image_paths)}",
                        'Task': task,
                        'Result': f"{result.prediction}→{result.label}"
                    })
                    pbar.update(1)
        
        return results
    
    def analyze_multimodal(self, texts: List[str], image_paths: List[str],
                          sample_ids: List[int] = None) -> Dict[str, List[PredictionResult]]:
        """Perform multimodal analysis for all tasks"""
        
        if sample_ids is None:
            sample_ids = list(range(len(texts)))
        
        results = {"sentiment": [], "emotion": [], "desire": []}
        total_operations = len(texts) * 3
        
        with tqdm(total=total_operations, desc="Multimodal Analysis", unit="prediction", **tqdm_kwargs) as pbar:
            for i, (text, image_path, sample_id) in enumerate(zip(texts, image_paths, sample_ids)):
                for task in ["sentiment", "emotion", "desire"]:
                    prompt_data = self.create_multimodal_prompt(text, image_path, task)
                    
                    if prompt_data is None:
                        default_pred = 'g' if task == "desire" else 'c'
                        default_label = desire_map.get(default_pred) if task == "desire" else "neutral"
                        
                        result = PredictionResult(
                            task=task,
                            reasoning="Error: Could not load image",
                            prediction=default_pred,
                            label=default_label,
                            sample_id=sample_id,
                            input_text=text,
                            image_path=image_path,
                            raw_llm_output="Error: Could not load image"
                        )
                    else:
                        result = self.generate_prediction(
                            prompt_data, task, sample_id, text, image_path
                        )
                    
                    results[task].append(result)
                    
                    pbar.set_postfix({
                        'Sample': f"{i+1}/{len(texts)}",
                        'Task': task,
                        'Result': f"{result.prediction}→{result.label}"
                    })
                    pbar.update(1)
        
        return results


def load_msed_dataset(dataset_path: str, split: str = 'test', 
                     limit: int = None) -> Tuple[List[str], List[str], List[Dict], List[int]]:
    """
    Load MSED dataset
    
    Returns:
        texts, image_paths, ground_truth_labels, sample_ids
    """
    
    texts = []
    image_paths = []
    labels = []
    sample_ids = []
    
    dataset_dir = Path(dataset_path)
    split_dir = dataset_dir / split
    
    # Load CSV file
    csv_path = split_dir / f'{split}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"DEBUG: Loading MSED dataset from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    
    print(f"   - CSV columns: {df.columns.tolist()}", flush=True)
    print(f"   - Dataset shape: {df.shape}", flush=True)
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        print(f"   - Limited to first {limit} rows", flush=True)
    
    for idx, row in df.iterrows():
        # Extract text
        text_content = None
        possible_text_columns = ['Caption', 'Title', 'text', 'caption', 'content', 'sentence', 'description']
        
        for col in possible_text_columns:
            if col in df.columns and pd.notna(row[col]):
                text_content = str(row[col]).strip()
                if text_content and text_content.lower() != 'nan':
                    break
        
        if text_content is None or text_content == 'nan' or text_content == '':
            print(f"Warning: No text found for row {idx}, skipping...")
            continue
        
        # Images are 1-indexed (1.jpg, 2.jpg, ...) while CSV rows are 0-indexed
        # So row 0 maps to image 1.jpg, row 1 maps to image 2.jpg, etc.
        image_id = idx + 1
        
        # Construct image path
        image_name = f"{image_id}.jpg"
        image_path = split_dir / 'images' / image_name
        
        # Check if image exists
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}, skipping...")
            continue
        
        texts.append(text_content)
        image_paths.append(str(image_path))
        sample_ids.append(image_id)  # Use image_id (1-indexed) instead of idx
        
        # Extract ground truth labels
        label_dict = {}
        if 'Sentiment' in df.columns and pd.notna(row['Sentiment']):
            label_dict['sentiment'] = str(row['Sentiment']).lower()
        if 'Emotion' in df.columns and pd.notna(row['Emotion']):
            label_dict['emotion'] = str(row['Emotion']).lower()
        if 'Desire' in df.columns and pd.notna(row['Desire']):
            label_dict['desire'] = str(row['Desire']).lower()
        
        labels.append(label_dict)
        
        # Debug first few samples
        if len(texts) <= 3:
            print(f"   - Sample {len(texts)}: CSV Row {idx} → Image ID {image_id}", flush=True)
            print(f"     Text: {text_content[:100]}...", flush=True)
            print(f"     Image: {image_path}", flush=True)
            print(f"     Labels: {label_dict}", flush=True)
    
    print(f"Loaded {len(texts)} samples from MSED {split} split", flush=True)
    
    # Print label distributions
    if labels:
        for task in ['sentiment', 'emotion', 'desire']:
            task_labels = [l.get(task) for l in labels if task in l]
            if task_labels:
                dist = Counter(task_labels)
                print(f"   - {task.capitalize()} distribution: {dict(dist)}", flush=True)
    
    return texts, image_paths, labels, sample_ids


def save_results(results: Dict[str, Dict[str, List[PredictionResult]]], 
                ground_truth: List[Dict], output_path: str, seed: int = 42):
    """Save results with debugging information in both JSON and CSV formats"""
    
    # Prepare data for JSON
    json_results = {}
    summaries = {}
    
    # Prepare data for CSV (flat structure)
    csv_data = []
    
    for mode, mode_results in results.items():
        json_results[mode] = {}
        summaries[mode] = {}
        
        for task, pred_list in mode_results.items():
            # Prepare individual results for JSON
            task_results = []
            for i, pred in enumerate(pred_list):
                gt_label = ground_truth[i].get(task) if i < len(ground_truth) else None
                
                json_entry = {
                    'sample_id': pred.sample_id,
                    'input_text': pred.input_text,
                    'image_path': pred.image_path,
                    'raw_llm_output': pred.raw_llm_output,
                    'reasoning': pred.reasoning,
                    'prediction_letter': pred.prediction,
                    'predicted_label': pred.label,
                    'ground_truth': gt_label,
                    'correct': pred.label == gt_label if gt_label else None
                }
                task_results.append(json_entry)
                
                # Add to CSV data (one row per prediction)
                csv_entry = {
                    'mode': mode,
                    'task': task,
                    'sample_id': pred.sample_id,
                    'image_filename': f"{pred.sample_id}.jpg" if pred.sample_id else None,
                    'image_path': pred.image_path,
                    'input_text': pred.input_text,
                    'prediction_letter': pred.prediction,
                    'predicted_label': pred.label,
                    'ground_truth': gt_label,
                    'correct': pred.label == gt_label if gt_label else None,
                    'reasoning': pred.reasoning,
                    'raw_llm_output': pred.raw_llm_output
                }
                csv_data.append(csv_entry)
            
            json_results[mode][task] = task_results
            
            # Calculate accuracy
            predictions = [pr.label for pr in pred_list]
            gt_labels = [ground_truth[i].get(task) for i in range(len(pred_list)) 
                        if i < len(ground_truth) and task in ground_truth[i]]
            
            if gt_labels and len(predictions) == len(gt_labels):
                correct = sum(1 for p, g in zip(predictions, gt_labels) if p == g)
                accuracy = correct / len(gt_labels)
                
                summaries[mode][task] = {
                    'accuracy': accuracy,
                    'total_samples': len(gt_labels),
                    'correct_predictions': correct,
                    'label_distribution': dict(Counter(predictions))
                }
    
    # Save JSON
    json_output = {
        'results': json_results,
        'summary': summaries,
        'config': {
            'dataset': 'MSED',
            'tasks': ['sentiment', 'emotion', 'desire'],
            'seed': seed
        }
    }
    
    json_path = output_path.replace('.csv', '.json') if output_path.endswith('.csv') else f"{output_path}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    # Save CSV
    csv_path = output_path if output_path.endswith('.csv') else f"{output_path}.csv"
    csv_df = pd.DataFrame(csv_data)
    
    # Reorder columns for better readability
    column_order = [
        'mode', 'task', 'sample_id', 'image_filename', 
        'predicted_label', 'prediction_letter', 'ground_truth', 'correct',
        'input_text', 'reasoning', 'raw_llm_output', 'image_path'
    ]
    # Only include columns that exist
    column_order = [col for col in column_order if col in csv_df.columns]
    csv_df = csv_df[column_order]
    
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"\nResults saved to:", flush=True)
    print(f"   - CSV: {csv_path}", flush=True)
    print(f"   - JSON: {json_path}", flush=True)
    
    # Print summary
    print(f"\n{'='*80}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    
    for mode, mode_summary in summaries.items():
        print(f"\n{mode.upper()} Results:", flush=True)
        for task, stats in mode_summary.items():
            print(f"  {task.capitalize()}:", flush=True)
            print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['correct_predictions']}/{stats['total_samples']})", flush=True)
            print(f"    Distribution: {stats['label_distribution']}", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"Total CSV rows: {len(csv_df)}", flush=True)
    print(f"{'='*80}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='MSED Multi-task Analysis with LLaVA')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to MSED dataset directory')
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'], 
                       default='test', help='Dataset split to use')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples (None = all)')
    parser.add_argument('--output-path', type=str, default='msed_llava_results',
                       help='Output file path (without extension)')
    parser.add_argument('--model-name', type=str, 
                       default='llava-hf/llava-v1.6-mistral-7b-hf',
                       help='Model name (use HF-converted LLaVA versions)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug-mode', action='store_true', default=False,
                       help='Enable detailed debugging output')
    parser.add_argument('--mode', type=str, choices=['text', 'image', 'multimodal', 'all'],
                       default='all', help='Analysis mode')
    
    args = parser.parse_args()
    
    print("="*80, flush=True)
    print("MSED MULTI-TASK ANALYSIS WITH LLAVA-1.6", flush=True)
    print("="*80, flush=True)
    print(f"Dataset path: {args.dataset_path}", flush=True)
    print(f"Split: {args.split}", flush=True)
    print(f"Sample limit: {args.limit if args.limit else 'All'}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Debug mode: {args.debug_mode}", flush=True)
    print(f"Random seed: {args.seed}", flush=True)
    
    # Initialize analyzer
    print(f"\n{'='*80}", flush=True)
    print("INITIALIZING MODEL", flush=True)
    print("="*80, flush=True)
   
    analyzer = MSEDAnalyzer(
        args.model_name, 
        args.seed,
        args.debug_mode
    )
    
    # Load dataset
    print(f"\n{'='*80}", flush=True)
    print("LOADING DATASET", flush=True)
    print("="*80, flush=True)
    
    texts, image_paths, ground_truth, sample_ids = load_msed_dataset(
        args.dataset_path, args.split, args.limit
    )
    
    if len(texts) == 0:
        print("No samples loaded. Exiting.", flush=True)
        return
    
    # Perform analysis based on mode
    results = {}
    
    if args.mode in ['text', 'all']:
        print(f"\n{'='*80}", flush=True)
        print("TEXT-ONLY ANALYSIS", flush=True)
        print("="*80, flush=True)
        results['text_only'] = analyzer.analyze_text_only(texts, sample_ids)
    
    if args.mode in ['image', 'all']:
        print(f"\n{'='*80}", flush=True)
        print("IMAGE-ONLY ANALYSIS", flush=True)
        print("="*80, flush=True)
        results['image_only'] = analyzer.analyze_image_only(image_paths, sample_ids)
    
    if args.mode in ['multimodal', 'all']:
        print(f"\n{'='*80}", flush=True)
        print("MULTIMODAL ANALYSIS", flush=True)
        print("="*80, flush=True)
        results['multimodal'] = analyzer.analyze_multimodal(texts, image_paths, sample_ids)
    
    # Save results
    print(f"\n{'='*80}", flush=True)
    print("SAVING RESULTS", flush=True)
    print("="*80, flush=True)
    
    save_results(results, ground_truth, args.output_path, args.seed)
    
    # Print completion
    print(f"\n{'='*80}", flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Processed {len(texts)} samples", flush=True)
    print(f"Check output file: {args.output_path}.json", flush=True)


if __name__ == "__main__":
    main()