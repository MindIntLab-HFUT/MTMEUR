import os
import cv2
import json
import base64
from openai import OpenAI
from typing import List, Dict
import time

openai_api_key = "your_key"
openai_api_base = "your_url"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def video_to_sampled_frames(video_path: str, output_dir: str, fps: float, max_frames: int) -> List[str]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        cap.release()
        return []
    sample_interval = max(int(frame_rate / fps), 1)
    frame_count = 0
    img_files = []
    start_time = time.time()
    timeout = 30
    while cap.isOpened() and len(img_files) < max_frames:
        if time.time() - start_time > timeout:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            height, width = frame.shape[:2]
            target_width, target_height = 640, 300
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame_file = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            img_files.append(frame_file)
        frame_count += 1
    cap.release()
    return img_files


def image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read())
            return f"data:image/jpeg;base64,{encoded_image.decode('utf-8')}"
    except Exception:
        return ""


def infer_with_qwen(image_paths: List[str], question: str, options: Dict[str, str]) -> str:
    question_with_options = (
        f"These images are frames extracted from a video, representing the content of the video. "
        f"Based on the content of the video, please answer the following question:\n"
        f"{question}\n"
        f"A: {options.get('A', 'Not Provided')}\n"
        f"B: {options.get('B', 'Not Provided')}\n"
        f"C: {options.get('C', 'Not Provided')}\n"
        f"D: {options.get('D', 'Not Provided')}\n\n"
        "Please provide the letter(s) of the correct option(s)."
    )
    image_contents = []
    for img_path in image_paths[:15]:
        base64_image = image_to_base64(img_path)
        if base64_image:
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            })

    if not image_contents:
        return "Unknown"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [*image_contents, {"type": "text", "text": question_with_options}]}
    ]
    try:
        chat_response = client.chat.completions.create(
            model="qwen2vl",
            messages=messages,
            timeout=60
        )
        response_text = chat_response.choices[0].message.content.strip()
        predicted_answer = "Unknown"
        for option in ['A', 'B', 'C', 'D']:
            if option in response_text.upper():
                predicted_answer = option
                break
        return predicted_answer
    except Exception:
        return "Unknown"
def process_videos_from_json(
    json_path: str,
    output_dir: str,
    output_file: str = 'results.json',
    fps: float = 0.8,
    max_frames: int = 20
) -> List[Dict]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    results = []
    total_questions = 0
    correct_count = 0
    for video_data in data:
        video_path = video_data.get("video_path")
        if not video_path or not os.path.exists(video_path):
            continue
        img_files = video_to_sampled_frames(video_path, output_dir, fps, max_frames)
        if not img_files:
            continue
        video_results = {
            "video_path": video_path,
            "questions": []
        }
        questions = video_data.get("questions", [])
        for question_data in questions:
            question = question_data.get("question")
            options = question_data.get("options", {})
            correct_answer = question_data.get("correct_answer")
            if not question or not options or not correct_answer:
                continue
            predicted_answer = infer_with_qwen(img_files, question, options)
            correct_answers = [ans.strip().upper() for ans in correct_answer.split(',')]
            is_correct = (len(correct_answers) == 1) and (predicted_answer.upper() == correct_answers[0])
            result = "Correct" if is_correct else "Incorrect"
            total_questions += 1
            if is_correct:
                correct_count += 1
            video_results["questions"].append({
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "model_output": predicted_answer,
                "result": result
            })
        if video_results["questions"]:
            results.append(video_results)
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    final_results = {
        "total_questions": total_questions,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "videos": results
    }
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(final_results, outfile, indent=4, ensure_ascii=False)
    except Exception:
        pass
    return final_results

if __name__ == "__main__":
    json_path = './data/example.json'
    output_dir = './tmp'
    output_file = '/data/qwen2vl_experiment.json'
    process_videos_from_json(json_path, output_dir, output_file, fps=1, max_frames=25)
