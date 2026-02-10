# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import PIL

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
# from training.trainer import Qwen2VLGRPOTrainer
from training.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )    
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )



# Global variables for debug logging
_DEBUG_LOG_INITIALIZED = False
_FAILED_LOG_INITIALIZED = False  # Track if failed cases log has been initialized

def extract_bbox_from_solution(solution):
    """Extract bbox coordinates from solution string."""
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(bbox_pattern, solution)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    return None

def visualize_bbox(image_path, bbox, output_path):
    """Draw bbox on image and save to output_path."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Draw bbox rectangle
        draw.rectangle(bbox, outline='red', width=4)

        # Add text label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        text = "Ground Truth"
        text_bbox = draw.textbbox((bbox[0], bbox[1]-25), text, font=font)
        draw.rectangle(text_bbox, fill='red')
        draw.text((bbox[0], bbox[1]-25), text, fill='white', font=font)

        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error visualizing bbox: {e}")
        return False

def log_failed_case(global_step, sample_idx, total_samples, sample_info):
    """Log a failed case where all rollouts got 0.0 reward."""
    global _FAILED_LOG_INITIALIZED

    if os.getenv("DEBUG_MODE") != "true":
        return

    failed_log_path = os.getenv("FAILED_LOG_PATH")
    failed_img_dir = os.getenv("FAILED_IMG_DIR")

    if not failed_log_path or not failed_img_dir:
        return

    # Create image directory if needed
    os.makedirs(failed_img_dir, exist_ok=True)

    # Determine file mode: 'w' for first write (overwrite), 'a' for append
    if not _FAILED_LOG_INITIALIZED:
        file_mode = 'w'
        _FAILED_LOG_INITIALIZED = True
    else:
        file_mode = 'a'

    with open(failed_log_path, file_mode) as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"FAILED CASE | STEP {global_step} | Sample {sample_idx}/{total_samples}\n")
        f.write(f"{'='*100}\n")
        f.write(f"Instruction: {sample_info['instruction']}\n")
        f.write(f"Image Path: {sample_info['image_path']}\n")

        # Extract bbox and visualize if present
        bbox = extract_bbox_from_solution(sample_info['solution'])
        if bbox and sample_info['image_path']:
            vis_img_path = os.path.join(failed_img_dir, f"step_{global_step}_sample_{sample_idx}.png")
            if visualize_bbox(sample_info['image_path'], bbox, vis_img_path):
                f.write(f"Visualized Image: {vis_img_path}\n")
                f.write(f"Ground Truth BBox: {bbox}\n")

        f.write(f"\nGround Truth Solution:\n{sample_info['solution']}\n")
        f.write(f"\nModel Outputs (All {len(sample_info['rollouts'])} Rollouts Failed):\n")

        for rollout in sample_info['rollouts']:
            f.write(f"\n--- Rollout {rollout['rollout_idx']}/{len(sample_info['rollouts'])} | Reward: {rollout['reward']:.1f} ---\n")
            f.write(f"{rollout['content']}\n")

        f.write(f"\n{'='*100}\n")

def extract_action(response):
    action_tag_pattern = r'<action>(.*?)</action>'
    action_pattern = r"'action':\s*'(\w+)'"
    action_pattern_1 = r"'action':\s*(\w+)"
    content_action_match = re.search(action_tag_pattern, response, re.DOTALL)
    if content_action_match:
        content_action = content_action_match.group(1).strip()
        action_match = re.search(action_pattern, content_action)
        if action_match:
            return action_match.group(1)
        action_match = re.search(action_pattern_1, content_action)
        if action_match:
            return action_match.group(1)
    return None

def extract_coord(response):
    action_tag_pattern = r'<action>(.*?)</action>'
    bbox_pattern = r'\[(\d+),\s*(\d+)]'
    content_action_match = re.search(action_tag_pattern, response, re.DOTALL)
    if content_action_match:
        content_action = content_action_match.group(1).strip()
        coord_match = re.search(bbox_pattern, content_action)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2))]
            return coord , True
    return [0, 0], False
def extract_bbox(response):
    action_tag_pattern = r'<action>(.*?)</action>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    content_action_match = re.search(action_tag_pattern, response, re.DOTALL)
    if content_action_match:
        content_action = content_action_match.group(1).strip()
        coord_match = re.search(bbox_pattern, content_action)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2)), int(coord_match.group(3)), int(coord_match.group(4))]
            return coord, True
    return [0, 0, 0, 0] , False

def accuracy_reward_action(completions, solution,scales, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    global _DEBUG_LOG_INITIALIZED

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Get global_step from trainer_state
    trainer_state = kwargs.get('trainer_state', None)
    global_step = trainer_state.global_step if trainer_state else -1

    # Get num_generations from kwargs, or infer from data
    num_generations = kwargs.get('num_generations', None)
    if num_generations is None:
        # Try to infer: count how many times each solution appears
        from collections import Counter
        solution_counts = Counter(solution)
        if solution_counts:
            # The most common count is likely the num_generations
            num_generations = solution_counts.most_common(1)[0][1]
        else:
            num_generations = len(contents)  # Fallback: assume all are for one sample

    total_samples = len(contents) // num_generations if num_generations > 0 else 1

    show_flage = False
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        reward = 0.0

        # Calculate sample and rollout indices
        sample_idx = idx // num_generations + 1
        rollout_idx = idx % num_generations + 1

        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")

            # Determine file mode: 'w' for first write (overwrite), 'a' for append
            if not _DEBUG_LOG_INITIALIZED:
                file_mode = 'w'
                _DEBUG_LOG_INITIALIZED = True
            else:
                file_mode = 'a'

            with open(log_path, file_mode) as f:
                # Write step separator at the beginning of each step (sample 1, rollout 1)
                if sample_idx == 1 and rollout_idx == 1:
                    f.write(f"\n{'='*100}\n")
                    f.write(f"{'='*35} STEP {global_step} {'='*35}\n")
                    f.write(f"{'='*100}\n")

                # Write sample separator at the beginning of each sample (rollout 1)
                if rollout_idx == 1:
                    f.write(f"\n{'─'*100}\n")
                    f.write(f"Sample {sample_idx}/{total_samples}\n")
                    f.write(f"{'─'*100}\n")

                # Write rollout information
                f.write(f"\n--- Rollout {rollout_idx}/{num_generations} | Accuracy reward of Action: {reward} ---\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if student_answer_action and ground_truth_action:
                    f.write(f"student_answer_action: {student_answer_action}\n")
                    f.write(f"ground_truth_action: {ground_truth_action}\n")
    return rewards
def accuracy_reward_coord(completions, solution,scales, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    global _DEBUG_LOG_INITIALIZED

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Get instruction and image_path from kwargs
    instructions = kwargs.get('instruction', [None] * len(contents))
    image_paths = kwargs.get('image_path', [None] * len(contents))

    # Get global_step from trainer_state
    trainer_state = kwargs.get('trainer_state', None)
    global_step = trainer_state.global_step if trainer_state else -1

    # Get num_generations from kwargs, or infer from data
    # In GRPO, each batch contains multiple completions for the same prompts
    # The actual num_generations per GPU might be different from the global setting
    num_generations = kwargs.get('num_generations', None)
    if num_generations is None:
        # Try to infer: count how many times each solution appears
        from collections import Counter
        solution_counts = Counter(solution)
        if solution_counts:
            # The most common count is likely the num_generations
            num_generations = solution_counts.most_common(1)[0][1]
        else:
            num_generations = len(contents)  # Fallback: assume all are for one sample

    total_samples = len(contents) // num_generations if num_generations > 0 else 1

    # Collect rollout information for failed case analysis
    sample_rollouts = {}

    show_flage = False
    for idx, (content, sol, scale, instruction, image_path) in enumerate(zip(contents, solution, scales, instructions, image_paths)):
        reward = 0.0

        # Calculate sample and rollout indices
        sample_idx = idx // num_generations + 1
        rollout_idx = idx % num_generations + 1

        # Try symbolic verification first
        # print("content: ", content)
        # print("sol: ", sol)
        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                # Only verify coordinates for click and hover
                if student_answer_action in ["click", "hover"]:
                    student_answer_coord, flag1 = extract_coord(content)
                    student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1])]
                    ground_truth_bbox, flag2 = extract_bbox(sol)
                    show_flage = flag1 and flag2
                    if ground_truth_bbox[0] <= student_answer_coord[0] <= ground_truth_bbox[2] and ground_truth_bbox[1] <= student_answer_coord[1] <= ground_truth_bbox[3]:
                        reward = 1.0
                    else:
                        reward = 0.0
                # Other actions only need correct action type: drag, type_text, press_enter, scroll, answer
                else:
                    reward = 1.0
            else:
                reward = 0.0
        except Exception:
            pass  # Continue to next verification method if this fails

        rewards.append(reward)

        # Collect rollout information for failed case analysis
        if sample_idx not in sample_rollouts:
            sample_rollouts[sample_idx] = {
                'instruction': instruction,
                'image_path': image_path,
                'solution': sol,
                'rollouts': []
            }

        sample_rollouts[sample_idx]['rollouts'].append({
            'rollout_idx': rollout_idx,
            'content': content,
            'reward': reward
        })

        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")

            # Determine file mode: 'w' for first write (overwrite), 'a' for append
            if not _DEBUG_LOG_INITIALIZED:
                file_mode = 'w'
                _DEBUG_LOG_INITIALIZED = True
            else:
                file_mode = 'a'

            with open(log_path, file_mode) as f:
                # Write step separator at the beginning of each step (sample 1, rollout 1)
                if sample_idx == 1 and rollout_idx == 1:
                    f.write(f"\n{'='*100}\n")
                    f.write(f"{'='*35} STEP {global_step} {'='*35}\n")
                    f.write(f"{'='*100}\n")

                # Write sample separator at the beginning of each sample (rollout 1)
                if rollout_idx == 1:
                    f.write(f"\n{'─'*100}\n")
                    f.write(f"Sample {sample_idx}/{total_samples}")
                    if instruction:
                        f.write(f" | instruction: {instruction}")
                    f.write(f"\n")
                    if image_path:
                        f.write(f"image_path: {image_path}\n")
                    f.write(f"{'─'*100}\n")

                # Write rollout information
                f.write(f"\n--- Rollout {rollout_idx}/{num_generations} | Accuracy reward of Coord: {reward} ---\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage:
                    f.write(f"student_answer_coord: {student_answer_coord}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")

    # After processing all rollouts, check for failed cases
    if os.getenv("DEBUG_MODE") == "true":
        for sample_idx, sample_info in sample_rollouts.items():
            all_rewards = [r['reward'] for r in sample_info['rollouts']]

            # If all rollouts got 0.0 reward, log as failed case
            if all(r == 0.0 for r in all_rewards):
                log_failed_case(global_step, sample_idx, total_samples, sample_info)

    return rewards



def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<action>.*?</action>"
    # pattern = r"<action>.*?</action>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

###  reward registry three parts
reward_funcs_registry = {
    "accuracy_action": accuracy_reward_action,
    "accuracy_coord": accuracy_reward_coord,
    "format": format_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the action assistant would take. The reasoning "
    "process and action are enclosed within <think> </think> and <action> </action> tags, respectively, i.e., "
    "<think> reasoning process here </think><action> action here </action>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_action','accuracy_coord','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            # for line in f:
            data = json.load(f)
            for item in data:
                if 'img_filename' in item:
                    # Store image path instead of loading the image
                    item['image_path'] = os.path.join(image_folder, item['img_filename'])
                    del item['img_filename'] # remove the image column so that it can be loaded later
                # Remove immediate image loading
                task_prompt = item['instruction']

                # Add HISTORY if present
                if 'history' in item and item['history']:
                    task_prompt = f"{item['instruction']}\n\nHISTORY:\n{item['history']}"

                item['problem'] = (
                    f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
                    "If history information is provided, consider it when choosing the next action.\n\n"
                    "Available actions:\n"
                    "- click: {{'action': 'click', 'coordinate': [x, y]}}\n"
                    "- hover: {{'action': 'hover', 'coordinate': [x, y]}}\n"
                    "- drag: {{'action': 'drag', 'from': [x1, y1], 'to': [x2, y2]}}\n"
                    "- type_text: {{'action': 'type_text', 'text': 'content'}}\n"
                    "- press_enter: {{'action': 'press_enter'}}\n"
                    "- scroll: {{'action': 'scroll', 'value': down/up}}\n"
                    "- answer: {{'action': 'answer', 'text': content}} (MUST use when current screenshot contains sufficient information to answer the query)\n\n"
                    "Output format: <think>..</think><action>..</action>n>"
                )
                if 'bbox' in item and item['bbox']:
                    # Actions with bbox: click, hover, drag
                    item['solution'] = f"<action>[{{'action': '{item['action']}', 'coordinate': {item['bbox']}}}]</action>"
                else:
                    # Actions without bbox: type_text, press_enter, scroll, answer
                    item['solution'] = f"<action>[{{'action': '{item['action']}'}}]</action>"
                
                all_data.append(item)

    dataset = Dataset.from_list(all_data)
    def make_conversation_from_json(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                'image_path': example['image_path'],  # Store path instead of loaded image
                'solution': example['solution'],
                'instruction': example['instruction'],  # Keep instruction for logging
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                'instruction': example['instruction'],  # Keep instruction for logging
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }

    dataset = dataset.map(make_conversation_from_json, num_proc=8)
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
