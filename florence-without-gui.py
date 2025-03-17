import os
from PIL import Image, ImageDraw, ImageFont

from typing import Optional
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import random
import numpy as np
import copy

class Florence2Analyzer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._initialize_model()
        
    def _initialize_model(self):
        # Keep the original model loading logic
        def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports.remove("flash_attn")
            return imports

        model_path = "microsoft/Florence-2-large"

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                attn_implementation="sdpa", 
                trust_remote_code=True
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        self.task_mapping = {
            "Object Detection": "od",
            "OCR": "ocr",
            "Expression Segmentation": "referring_expression_segmentation",
            "Caption": "caption",
            "Detailed Caption": "detailed_caption",
            "More Detailed Caption": "more_detailed_caption",
            "Dense Region Caption": "dense_region_caption",
            "Region Proposal": "region_proposal",
            "OCR with Region": "ocr_with_region",
            "Caption to Phrase Grounding": "caption_to_phrase_grounding",
            "Open Vocabulary Detection": "open_vocabulary_detection"
        }

    def analyze_image(
        self,
        image_path: str,
        task: str = "Object Detection",
        text_input: Optional[str] = None,
        output_dir: Optional[str] = "outputs",
        save_image: bool = True,
        only_caption: bool = False
    ):
        """
        Analyze an image with specified task
        
        Args:
            image_path: Path to input image
            task: One of the supported tasks (default: "Object Detection")
            text_input: Optional text input for relevant tasks
            output_dir: Directory to save results
            save_image: Whether to save annotated image
            only_caption: If the user wants only to be returned the caption results of the image
        Returns:
            Dictionary with results and output paths
        """
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        task_key = self.task_mapping.get(task)
        if not task_key:
            raise ValueError(f"Invalid task. Supported tasks: {list(self.task_mapping.keys())}")

        # Process image
        image = Image.open(image_path).convert("RGB")
        processed_image, text_output = self._process_image(image, task_key, text_input)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if not only_caption:
            print("CAI NO ERRADO")
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            results = {
                "input_image": image_path,
                "task": task,
                "text_output": text_output
            }
            # Save outputs
            if save_image and processed_image is not None:
                output_image_path = os.path.join(output_dir, f"{base_name}_{task.replace(' ', '_')}.jpg")
                processed_image.save(output_image_path)
                results["output_image"] = output_image_path

            if text_output is not None:
                output_text_path = os.path.join(output_dir, f"{base_name}_{task.replace(' ', '_')}.txt")
                with open(output_text_path, "w") as f:
                    f.write(str(text_output))
                results["text_output_file"] = output_text_path

        if only_caption:
            results = text_output
            print("cheguie aq   ?")

        return results

    def generate_caption(
        self,
        image_path: str,
        task: str = "Caption",
    ):
        """
        Analyze an image with specified task
        
        Args:
            image_path: Path to input image
            task: One of the supported tasks (default: "Object Detection")

        Returns:
            Caption of the image
        """
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        task_key = self.task_mapping.get(task)
        if not task_key:
            raise ValueError(f"Invalid task. Supported tasks: {list(self.task_mapping.keys())}")
        
        # Process image
        image = Image.open(image_path).convert("RGB")
        processed_image, text_output = self._process_image(image, task_key)
        
        # Prepare results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        result = text_output
            
        return result


    def _process_image(self, image, task_key: str, text_input: Optional[str] = None):
        # Core processing logic from original detect_objects function
        if task_key in ["referring_expression_segmentation", 
                       "caption_to_phrase_grounding",
                       "open_vocabulary_detection"] and not text_input:
            raise ValueError(f"Text input required for {task_key} task")

        task_prompt = f"<{task_key.upper()}>"
        results = self._run_example(task_prompt, image, text_input)
        
        processed_image = image.copy()
        draw = ImageDraw.Draw(processed_image)
        
        # Handle different task types
        if task_key in ["od", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding"]:
            print(results)
            bboxes = results.get("bboxes", [])
            labels = results.get("labels", [])
            for bbox, label in zip(bboxes, labels):
                # Draw boxes and labels...
                x0, y0, x1, y1 = bbox
                
                draw.rectangle(bbox)
                draw.text((x0, y0), str(label)) 
            return processed_image, None
        
        elif task_key == "referring_expression_segmentation":
            self.draw_polygons(processed_image, results)
            return processed_image, None
        
        elif task_key == "ocr":
            return image, results  # Return original image + text results
        
        # Add handling for other task types...
        
        return processed_image, results

    def _run_example(self, task_prompt: str, image, text_input: Optional[str] = None):
        # Similar to original run_example function but using instance variables
        prompt = task_prompt if text_input is None else f"{task_prompt} {text_input}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

    colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
                'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

    def draw_polygons(image, prediction, fill_mask=False):  
        """  
        Draws segmentation masks with polygons on an image.  
    
        Parameters:  
        - image_path: Path to the image file.  
        - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                    'polygons' is a list of lists, each containing vertices of a polygon.  
                    'labels' is a list of labels corresponding to each polygon.  
        - fill_mask: Boolean indicating whether to fill the polygons with color.  
        """  
        draw = ImageDraw.Draw(image)  
        scale = 1  # Set up scale factor if needed (use 1 if not scaling)  
        
        # Iterate over polygons and labels  
        for polygons, label in zip(prediction['polygons'], prediction['labels']):  
            color = random.choice(colormap)  
            fill_color = random.choice(colormap) if fill_mask else None  
            
            for _polygon in polygons:  
                _polygon = np.array(_polygon).reshape(-1, 2)  
                if len(_polygon) < 3:  
                    print('Invalid polygon:', _polygon)  
                    continue  
                
                _polygon = (_polygon * scale).reshape(-1).tolist()  
                
                # Draw the polygon  
                if fill_mask:  
                    draw.polygon(_polygon, outline=color, fill=fill_color)  
                else:  
                    draw.polygon(_polygon, outline=color)  
                
                # Draw the label text  
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  

    def draw_ocr_bboxes(image, prediction):
        scale = 1
        draw = ImageDraw.Draw(image)
        bboxes, labels = prediction['quad_boxes'], prediction['labels']
        for box, label in zip(bboxes, labels):
            color = random.choice(colormap)
            new_box = (np.array(box) * scale).tolist()
            draw.polygon(new_box, width=3, outline=color)
            draw.text((new_box[0]+8, new_box[1]+2),
                        "{}".format(label),
                        align="right",
                        fill=color)

    def detect_objects(image, task, text_input=None):
        if not task:
            return image, None
        
        task_prompt = f"<{task.upper()}>"
        results = run_example(task_prompt, image, text_input)
        
        draw = ImageDraw.Draw(image)
        # Load a font
        try:
            font = ImageFont.truetype(font="arial.ttf", size=100)
        except IOError:
            font = ImageFont.load_default(size=50)
        
        if task in ["od", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding"]:
            for bbox, label in zip(results[f"<{task.upper()}>"]["bboxes"], results[f"<{task.upper()}>"]["labels"]):
                x0, y0, x1, y1 = bbox
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                if task == "od":
                    color = get_color(label)
                else:
                    color = "lightgreen"
                    label = "" if task == "region_proposal" else label
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                
                # Draw background rectangle for text
                text_bbox = draw.textbbox((x0, y0), label, font=font)
                draw.rectangle(text_bbox, fill="white")
                
                draw.text((x0, y0), label, fill="black", font=font)
            
            return image, None
        elif task == "referring_expression_segmentation":
            output_image = copy.deepcopy(image)
            draw_polygons(output_image, results[f"<{task.upper()}>"], fill_mask=True)
            return output_image, None
        elif task == "ocr":
            return image, results[f"<{task.upper()}>"]
        elif task == "ocr_with_region":
            output_image = copy.deepcopy(image)
            draw_ocr_bboxes(output_image, results[f"<{task.upper()}>"])
            return output_image, None
        elif task == "open_vocabulary_detection":
            for bbox, label in zip(results[f"<{task.upper()}>"]["bboxes"], results[f"<{task.upper()}>"]["bboxes_labels"]):
                x0, y0, x1, y1 = bbox
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
                color = get_color(label)
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                
                # Draw background rectangle for text
                text_bbox = draw.textbbox((x0, y0), label, font=font)
                draw.rectangle(text_bbox, fill="white")
                
                draw.text((x0, y0), label, fill="black", font=font)
            
            return image, None
        else:
            return image, results[f"<{task.upper()}>"]

# Example usage
if __name__ == "__main__":
    analyzer = Florence2Analyzer()
    
    # Analyze single image
    results = analyzer.generate_caption(
        image_path="/home/cluster/Documentos/hl_projects/florence-2-large-playground/images/armaecelular.png",
    )
    
    print("Analysis results:", results)
    string = results["<CAPTION>"]
    if "gun" in string or "handgun" in string:
        print("tipo da resposta:", results["<CAPTION>"])
    
    # Batch processing example
"""     image_folder = "/home/cluster/Documentos/hl_projects/florence-2-large-playground/images/"
    for img_file in os.listdir(image_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, img_file)
            results = analyzer.analyze_image(
                image_path=img_path,
                task="Detailed Caption",
                output_dir="batch_analysis"
            )
            print(f"Processed {img_file}: {results['text_output']}") """