import os
import random
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from groq import Groq
import time
import pandas as pd
from collections import Counter

# Progress bar settings
tqdm_kwargs = dict(file=sys.stdout)

# Label mappings for MSED tasks
# IMPORTANT:
# Evaluation must use the ground-truth annotation schema available in the dataset.
# captions_test.csv contains only emotion labels with this schema:
# awe, contentment, excitement, anger, sadness, amusement, fear, disgust.
# Sentiment/desire are kept for compatibility with other datasets, but they are not
# evaluated unless the input dataset actually contains those ground-truth columns.
sentiment_map = {"a": "positive", "b": "negative", "c": "neutral"}
emotion_map = {
    "a": "awe",
    "b": "contentment",
    "c": "excitement",
    "d": "anger",
    "e": "sadness",
    "f": "amusement",
    "g": "fear",
    "h": "disgust",
}
desire_map = {
    "a": "vengeance", "b": "curiosity", "c": "social-contact",
    "d": "family", "e": "tranquility", "f": "romance", "g": "none"
}

TASK_LABEL_MAPS = {
    "sentiment": sentiment_map,
    "emotion": emotion_map,
    "desire": desire_map,
}

def normalize_label(label) -> Optional[str]:
    """Normalize labels so prediction and ground truth are compared in the same label space."""
    if label is None or pd.isna(label):
        return None
    label = str(label).strip().lower()
    label = label.replace("_", "-")
    aliases = {
        "happy": "amusement",
        "happiness": "amusement",
        "sad": "sadness",
    }
    return aliases.get(label, label)

def get_available_tasks(ground_truth: List[Dict]) -> List[str]:
    """Return only tasks that have actual ground-truth annotations in this dataset."""
    available_tasks = []
    for task in ["sentiment", "emotion", "desire"]:
        if any(task in label_dict and label_dict.get(task) is not None for label_dict in ground_truth):
            available_tasks.append(task)
    return available_tasks

def get_existing_completed_keys(output_path: str) -> set:
    """Read existing CSV output and return completed (mode, task, sample_id) keys for resume."""
    csv_path = output_path if str(output_path).endswith('.csv') else f"{output_path}.csv"
    if not os.path.exists(csv_path):
        return set()
    try:
        existing_df = pd.read_csv(csv_path)
        required_cols = {"mode", "task", "sample_id"}
        if not required_cols.issubset(set(existing_df.columns)):
            return set()
        existing_df = existing_df.dropna(subset=["mode", "task", "sample_id"])
        return set(
            (str(row["mode"]), str(row["task"]), int(row["sample_id"]))
            for _, row in existing_df.iterrows()
        )
    except Exception as e:
        print(f"Warning: Could not read existing output file for resume: {e}", flush=True)
        return set()

def append_progress_row(output_path: str, mode: str, pred: "PredictionResult", gt_label: Optional[str]):
    """Append one completed prediction immediately so progress is not lost if the run stops."""
    if not output_path:
        return

    csv_path = output_path if str(output_path).endswith('.csv') else f"{output_path}.csv"
    row = {
        'mode': mode,
        'task': pred.task,
        'sample_id': pred.sample_id,
        'image_filename': f"{pred.sample_id}.jpg" if pred.sample_id else None,
        'predicted_label': pred.label,
        'prediction_letter': pred.prediction,
        'ground_truth': gt_label,
        'correct': pred.label == gt_label if gt_label else None,
        'input_text': pred.input_text,
        'reasoning': pred.reasoning,
        'raw_llm_output': pred.raw_llm_output,
        'image_path': pred.image_path
    }

    file_exists = os.path.exists(csv_path)
    pd.DataFrame([row]).to_csv(
        csv_path,
        mode='a' if file_exists else 'w',
        header=not file_exists,
        index=False,
        encoding='utf-8'
    )

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

        # Groq API backend. This replaces ONLY the local LLaVA/vLLM backend.
        # All task prompts, label maps, extraction logic, evaluation, and saving logic are kept the same.
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it first, e.g. "
                'Windows PowerShell: setx GROQ_API_KEY "your_key_here"'
            )

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = 0.0
        self.max_tokens = 512
    
    def load_and_validate_image(self, image_path: str):
        """Image loading is intentionally disabled for Groq llama-3.3-70b-versatile.

        The original LLaVA baseline supported image and multimodal modes.
        Groq llama-3.3-70b-versatile is a text-only chat model, so for fair
        caption-based experiments use --mode text.
        """
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
            task_desc = """Analyze the primary emotion expressed in this text using the dataset's ground-truth emotion schema.

**Classification Options:**
a = awe
b = contentment
c = excitement
d = anger
e = sadness
f = amusement
g = fear
h = disgust"""
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
        
        prompt_text = f"""{task_desc}

Text: "{text}"

**Instructions:**
First, explain your reasoning step-by-step. Then provide your final answer as a single letter.

**Format your response exactly as:**
Reasoning: [your analysis]
Final Answer: [letter]"""
        
        if self.debug_mode:
            print(f"\n🎯 Text-only {task} prompt created")
            print(f"   Text: {text[:100]}...")
        
        return {"prompt": prompt_text, "multi_modal_data": None}
    
    def create_image_only_prompt(self, image_path: str, task: str) -> Optional[Dict]:
        """Image-only mode is not supported by Groq llama-3.3-70b-versatile.

        Kept as a function so the rest of the original code structure remains intact.
        Use --mode text for caption-based experiments.
        """
        if self.debug_mode:
            print(f"\n⚠️ Image-only {task} skipped: Groq llama-3.3-70b-versatile is text-only")
        return None

    def create_multimodal_prompt(self, text: str, image_path: str, task: str) -> Optional[Dict]:
        """Multimodal mode is not supported by Groq llama-3.3-70b-versatile.

        Kept as a function so the rest of the original code structure remains intact.
        Use --mode text for caption-based experiments.
        """
        if self.debug_mode:
            print(f"\n⚠️ Multimodal {task} skipped: Groq llama-3.3-70b-versatile is text-only")
        return None

    def extract_label_from_text(self, text: str) -> str:
        """Extract label from generated text with multiple strategies"""
        
        if not text:
            return 'x'
        
        # Clean the text
        text = text.strip().lower()
        
        # Strategy 1: Check if the text is just a single letter
        if len(text) == 1 and text in 'abcdefgh':
            return text
        
        # Strategy 2: Look for the last valid letter in the text (most likely the final answer)
        valid_letters = []
        for char in reversed(text):
            if char in 'abcdefgh':
                valid_letters.append(char)
        
        if valid_letters:
            return valid_letters[0]  # Return the last valid letter found
        
        # Strategy 3: Look for patterns like "Response: a" or "Answer: b"
        import re
        
        # Pattern for "Response: X" or "Answer: X" or just ": X"
        response_pattern = r'(?:response|answer|result)?\s*:?\s*([abcdefgh])'
        matches = re.findall(response_pattern, text)
        if matches:
            return matches[-1]  # Return the last match
        
        # Strategy 4: Look for the letter after common phrases
        common_phrases = [
            r'the answer is\s*([abcdefgh])',
            r'i choose\s*([abcdefgh])',
            r'my response is\s*([abcdefgh])',
            r'therefore\s*([abcdefgh])',
            r'so\s*([abcdefgh])',
            r'thus\s*([abcdefgh])',
            r'final answer\s*:?\s*([abcdefgh])',
            r'conclusion\s*:?\s*([abcdefgh])'
        ]
        
        for pattern in common_phrases:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1]  # Return the last match
        
        # Strategy 5: Look for standalone letters (word boundaries)
        standalone_pattern = r'\b([abcdefgh])\b'
        matches = re.findall(standalone_pattern, text)
        if matches:
            return matches[-1]  # Return the last standalone letter
        
        # Strategy 6: If text contains explanations, look for the letter at the end
        # Split by common separators and check each part
        parts = re.split(r'[.,!?;:\n\t\s]+', text)
        for part in reversed(parts):
            if part and len(part) == 1 and part in 'abcdefgh':
                return part
        
        # Strategy 7: Look for the first valid letter as fallback
        for char in text:
            if char in 'abcdefgh':
                return char
        
        # Default fallback
        return 'x'
    
    def generate_prediction(self, prompt_data: Dict, task: str, 
                          sample_id: int = None, input_text: str = None, 
                          image_path: str = None) -> PredictionResult:
        """Generate prediction with reasoning"""
        
        try:
            # Groq API call. This replaces self.llm.generate(...) only.
            # The prompt text and downstream parsing remain the same as the original code.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt_data["prompt"]}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            generated_text = response.choices[0].message.content.strip()
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
                         sample_ids: List[int] = None,
                         image_paths: List[str] = None,
                         tasks: List[str] = None,
                         ground_truth: List[Dict] = None,
                         output_path: str = None,
                         existing_keys: set = None) -> Dict[str, List[PredictionResult]]:
        """Perform text-only analysis only for tasks that have ground truth in the dataset.

        This prevents evaluating sentiment/desire on captions_test.csv because those
        annotations are not present. It also supports resumable progress by skipping
        predictions already present in the output CSV.
        """

        if sample_ids is None:
            sample_ids = list(range(len(texts)))

        if image_paths is None:
            image_paths = [None] * len(texts)

        if tasks is None:
            tasks = ["sentiment", "emotion", "desire"]

        if ground_truth is None:
            ground_truth = [{} for _ in texts]

        if existing_keys is None:
            existing_keys = set()

        results = {task: [] for task in tasks}
        pending_operations = sum(
            1
            for sample_id in sample_ids
            for task in tasks
            if ("text_only", task, int(sample_id)) not in existing_keys
        )

        if pending_operations == 0:
            print("All requested text-only predictions already exist in the output file. Nothing new to process.", flush=True)
            return results

        with tqdm(total=pending_operations, desc="Text-Only Analysis", unit="prediction", **tqdm_kwargs) as pbar:
            for i, (text, sample_id, image_path) in enumerate(zip(texts, sample_ids, image_paths)):
                for task in tasks:
                    resume_key = ("text_only", task, int(sample_id))
                    if resume_key in existing_keys:
                        continue

                    prompt_data = self.create_text_only_prompt(text, task)

                    result = self.generate_prediction(
                        prompt_data, task, sample_id, text, image_path
                    )

                    results[task].append(result)

                    gt_label = ground_truth[i].get(task) if i < len(ground_truth) else None
                    append_progress_row(output_path, "text_only", result, gt_label)
                    existing_keys.add(resume_key)

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

    # Original behavior: dataset_path points to an MSED directory containing split/split.csv.
    # Added compatibility: dataset_path can also point directly to a CSV such as captions_test.csv.
    if dataset_dir.is_file() and dataset_dir.suffix.lower() == ".csv":
        csv_path = dataset_dir
        split_dir = dataset_dir.parent
    else:
        split_dir = dataset_dir / split
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

        # Construct image path. If the CSV already has image_path, use it; otherwise use original MSED convention.
        # For Groq text-only experiments, do not skip rows just because images are unavailable.
        # Image and multimodal modes are unsupported by this text-only Groq model anyway.
        
        if 'image_path' in df.columns and pd.notna(row['image_path']):
            image_path_value = str(row['image_path'])
            
        else:
            image_name = f"{image_id}.jpg"
            image_path = split_dir / 'images' / image_name
            image_path_value = str(image_path) if image_path.exists() else None
        
        texts.append(text_content)
        image_paths.append(image_path_value)
        sample_ids.append(image_id)  # Use image_id (1-indexed) instead of idx
        
        # Extract ground truth labels. Supports both original MSED column names and lowercase CSV names.
        label_dict = {}
        for source_col, target_task in [
            ('Sentiment', 'sentiment'), ('sentiment', 'sentiment'),
            ('Emotion', 'emotion'), ('emotion', 'emotion'),
            ('Desire', 'desire'), ('desire', 'desire'),
        ]:
            if source_col in df.columns and pd.notna(row[source_col]):
                label_dict[target_task] = normalize_label(row[source_col])
        
        labels.append(label_dict)
        
        # Debug first few samples
        if len(texts) <= 3:
            print(f"   - Sample {len(texts)}: CSV Row {idx} → Image ID {image_id}", flush=True)
            print(f"     Text: {text_content[:100]}...", flush=True)
            print(f"     Image: {image_path_value}", flush=True)
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
                ground_truth: List[Dict], output_path: str, seed: int = 42,
                tasks: List[str] = None):
    """Save results with debugging information in both JSON and CSV formats.

    Resume behavior:
    - Existing CSV rows are preserved.
    - New rows are appended during processing.
    - Final save de-duplicates by (mode, task, sample_id), keeping the first row,
      so already completed work is not replaced.
    """

    if tasks is None:
        tasks = ["sentiment", "emotion", "desire"]

    csv_path = output_path if output_path.endswith('.csv') else f"{output_path}.csv"
    json_path = output_path.replace('.csv', '.json') if output_path.endswith('.csv') else f"{output_path}.json"

    # Prepare newly collected rows in case append_progress_row was disabled or interrupted.
    csv_data = []
    for mode, mode_results in results.items():
        for task, pred_list in mode_results.items():
            for pred in pred_list:
                sample_index = None
                if pred.sample_id is not None:
                    sample_index = int(pred.sample_id) - 1
                gt_label = None
                if sample_index is not None and 0 <= sample_index < len(ground_truth):
                    gt_label = ground_truth[sample_index].get(task)

                csv_data.append({
                    'mode': mode,
                    'task': task,
                    'sample_id': pred.sample_id,
                    'image_filename': f"{pred.sample_id}.jpg" if pred.sample_id else None,
                    'predicted_label': pred.label,
                    'prediction_letter': pred.prediction,
                    'ground_truth': gt_label,
                    'correct': pred.label == gt_label if gt_label else None,
                    'input_text': pred.input_text,
                    'reasoning': pred.reasoning,
                    'raw_llm_output': pred.raw_llm_output,
                    'image_path': pred.image_path
                })

    existing_df = pd.DataFrame()
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not read existing CSV during final save: {e}", flush=True)

    new_df = pd.DataFrame(csv_data)
    frames = [df for df in [existing_df, new_df] if not df.empty]
    if frames:
        csv_df = pd.concat(frames, ignore_index=True)
        if {"mode", "task", "sample_id"}.issubset(csv_df.columns):
            csv_df = csv_df.drop_duplicates(subset=["mode", "task", "sample_id"], keep="first")
    else:
        csv_df = pd.DataFrame(columns=[
            'mode', 'task', 'sample_id', 'image_filename', 'predicted_label',
            'prediction_letter', 'ground_truth', 'correct', 'input_text',
            'reasoning', 'raw_llm_output', 'image_path'
        ])

    column_order = [
        'mode', 'task', 'sample_id', 'image_filename', 
        'predicted_label', 'prediction_letter', 'ground_truth', 'correct',
        'input_text', 'reasoning', 'raw_llm_output', 'image_path'
    ]
    column_order = [col for col in column_order if col in csv_df.columns]
    csv_df = csv_df[column_order]
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')

    # Build JSON + summary from final merged CSV, not only from the current run.
    json_results = {}
    summaries = {}

    if not csv_df.empty:
        for mode in sorted(csv_df['mode'].dropna().unique()):
            mode_df = csv_df[csv_df['mode'] == mode]
            json_results[mode] = {}
            summaries[mode] = {}

            for task in tasks:
                task_df = mode_df[mode_df['task'] == task]
                if task_df.empty:
                    continue

                json_results[mode][task] = task_df.to_dict(orient='records')

                eval_df = task_df[
                    task_df['ground_truth'].notna() &
                    task_df['predicted_label'].notna()
                ].copy()

                if not eval_df.empty:
                    correct = (eval_df['predicted_label'] == eval_df['ground_truth']).sum()
                    total = len(eval_df)
                    summaries[mode][task] = {
                        'accuracy': float(correct / total) if total else None,
                        'total_samples': int(total),
                        'correct_predictions': int(correct),
                        'label_distribution': dict(Counter(task_df['predicted_label'].dropna()))
                    }

    json_output = {
        'results': json_results,
        'summary': summaries,
        'config': {
            'dataset': 'MSED/captions_test',
            'tasks_evaluated': tasks,
            'seed': seed,
            'resume_enabled': True,
            'note': 'Tasks are evaluated only when corresponding ground-truth labels exist in the dataset.'
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:", flush=True)
    print(f"   - CSV: {csv_path}", flush=True)
    print(f"   - JSON: {json_path}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    for mode, mode_summary in summaries.items():
        print(f"\n{mode.upper()} Results:", flush=True)
        for task, stats in mode_summary.items():
            print(f"  {task.capitalize()}:", flush=True)
            print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['correct_predictions']}/{stats['total_samples']})", flush=True)
            print(f"    Distribution: {stats['label_distribution']}", flush=True)

    skipped_tasks = [task for task in ["sentiment", "emotion", "desire"] if task not in tasks]
    if skipped_tasks:
        print(f"\nSkipped task evaluation because ground-truth labels are unavailable: {skipped_tasks}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='MSED Multi-task Analysis with Groq Llama 3.3 70B')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to MSED dataset directory')
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'], 
                       default='test', help='Dataset split to use')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples (None = all)')
    parser.add_argument('--output-path', type=str, default='msed_llava_results',
                       help='Output file path (without extension)')
    parser.add_argument('--model-name', type=str, 
                       default='llama-3.3-70b-versatile',
                       help='Groq model name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug-mode', action='store_true', default=False,
                       help='Enable detailed debugging output')
    parser.add_argument('--mode', type=str, choices=['text', 'image', 'multimodal', 'all'],
                       default='all', help='Analysis mode')
    
    args = parser.parse_args()
    
    print("="*80, flush=True)
    print("MSED MULTI-TASK ANALYSIS WITH GROQ LLAMA-3.3-70B", flush=True)
    print("="*80, flush=True)
    print(f"Dataset path: {args.dataset_path}", flush=True)
    print(f"Split: {args.split}", flush=True)
    print(f"Sample limit: {args.limit if args.limit else 'All'}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    if args.mode != "text":
        print("Note: Groq llama-3.3-70b-versatile is text-only. Use --mode text for caption experiments.", flush=True)
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
    
    # Decide which tasks to evaluate from the annotations actually available in this dataset.
    available_tasks = get_available_tasks(ground_truth)
    if not available_tasks:
        print("No ground-truth task annotations found in this dataset. Exiting without evaluation.", flush=True)
        return

    print(f"\nTasks with ground-truth annotations in this dataset: {available_tasks}", flush=True)
    excluded_tasks = [task for task in ["sentiment", "emotion", "desire"] if task not in available_tasks]
    if excluded_tasks:
        print(f"Excluding tasks without ground truth: {excluded_tasks}", flush=True)

    existing_keys = get_existing_completed_keys(args.output_path)
    if existing_keys:
        print(f"Resume enabled: found {len(existing_keys)} completed predictions in existing output file.", flush=True)

    # Perform analysis based on mode
    results = {}

    if args.mode in ['text', 'all']:
        print(f"\n{'='*80}", flush=True)
        print("TEXT-ONLY ANALYSIS", flush=True)
        print("="*80, flush=True)
        results['text_only'] = analyzer.analyze_text_only(
            texts, sample_ids, image_paths,
            tasks=available_tasks,
            ground_truth=ground_truth,
            output_path=args.output_path,
            existing_keys=existing_keys
        )

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
    
    save_results(results, ground_truth, args.output_path, args.seed, tasks=available_tasks)
    
    # Print completion
    print(f"\n{'='*80}", flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Processed {len(texts)} samples", flush=True)
    print(f"Check output file: {args.output_path}.json", flush=True)


if __name__ == "__main__":
    main()