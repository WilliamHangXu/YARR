import numpy as np
import torch
import sys
sys.path.append('/home/hangxu/Grounded-SAM-2')
import json
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import base64
import openai
import os
import cv2
import supervision as sv
import pprint
import shutil
from yarr.utils.openai_api_keys import API_KEY_1, API_KEY_2

class GSProcessor:
    def __init__(self, port=20107):
        self.API_KEY = API_KEY_1
        self.client = openai.OpenAI(api_key=self.API_KEY)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_models()
        self._enable_mixed_precision()
        self.init = False
        self.messages = []

    def init_models(self):
        print("Initializing models...")
        self.predictor = build_sam2_camera_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "/home/hangxu/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        )
        model_id = "/home/hangxu/Grounded-SAM-2/groundingdino"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to("cuda")
        print("Models initialized")

    def _enable_mixed_precision(self):
        """Configure mixed precision settings"""
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Input: the first frame of observation and task description
    # Output: a list of objects, and a dictionary that categorizes these objects.
    def get_detection_prompt(self, task_prompt, frame, model_name="gpt-4o"):

        
        img_uri = image_to_data_uri(frame, convert=True)

        # Convert the image string back to an image to check if chatgpt is fed the correct image
        if img_uri.startswith("data:"):
            header, image_data = img_uri.split(",", 1)
        else:
            image_data = img_uri

        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Specify the target directory and file name
        target_dir = "/home/hangxu/RVT"
        

        file_path = os.path.join(target_dir, "output_image.jpg")

        # Save the image bytes to a file
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a robotic manipulation task planner. "
                        f"You are given the following task: \"{task_prompt}\" and the initial observation of the task as an image. "
                        "Please identify task-related objects (except the robot itself) in this image and state their quantity. "
                        "If there are objects that are irrelevant to the task, please also identify them. "
                        
                        # "Articulated objects (those composed of multiple rigid parts connected by joints) should be decomposed into separate objects based on their task-related components. Other objects should not be decomposed. "
                        "Articulated objects (those composed of multiple rigid parts connected by joints) should not be decomposed into separate objects based on their task-related components, even when these components are of different colors. "
                        "You should not decompose any objects based on their task-related components. "

                        "If there are identical objects, you do not need to distinguish them, like based on location. "
                        "Simply repeat the same name. Objects that are not identical should be named differently. "
                        "Then, explain how to complete the task step by step and list the objects used in each step."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img_uri}
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=model_name,
            messages=self.messages,
            temperature=0.0,
            max_tokens=300
        )
        self.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        print("Detection step response:", self.messages[-1]["content"])

        # Refine object names to produce bounding box labels as a raw JSON array
        self.messages.append({
            "role": "user",
            "content": ("Next, I wish to generate bounding boxes for the objects you identified using the Grounding DINO model. "
                        "Please adjust the object names you just used as needed, so that they become suitable text prompts for the Grounding DINO model. Do not add or remove any objects. "
                        # "Please consider the following format: color + shape/size + object name. Please avoid using prepositions. "
                        "Please consider using adjectives that accurately describe the color, shape, size of the objects. "
                        "Except for articulated objects, in which case you should not include any adjectives and use a name that can be used to detect all components. Avoid complicated structural descriptions. "
                        # "For articulated objects with multiple task related components, please just use a name that can be used to detect all components. Avoid complicated structural descriptions. "
                        # "If articulated objects have components with different colors, maybe consider omit the color part. "
                        "Again, you do not need to distinguish between objects that are identical. Simply repeat the same name. "
                        "Please output the bounding box labels as a raw JSON array without any markdown formatting.")
        })

        response = self.client.chat.completions.create(
            model=model_name,
            messages=self.messages,
            temperature=0.0,
            max_tokens=300
        )
        self.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        
        # Attempt to parse the bounding box labels as a JSON array.
        # dino_objects = self.get_valid_json("array")
        print("Bounding box labels response:", self.messages[-1]["content"])



        # Categorize the objects into manipulation objects, receiver objects, and other objects
        self.messages.append({
            "role": "user",
            "content": ("Please categorize the bounding box labels into three categories: manipulation objects (task-relevant objects that are directly manipulated or interacted with by the robot. "
                        "For instance, if the robot wants to hit a ball into a box with a stick, the stick is the manipulation object. Notice that all tasks must have at least one manipulation object), "
                        "receiver objects (task-relevant objects that are not directly interacted with by the robot. For instance, if the robot wants to hit a ball into a box with a stick, then ball and box are receiver objects. "
                        "Notice that not all tasks have receiver objects, such as tasks that only involve one object), and other objects (objects that are not manipulation or receiver objects and are irrelevant to the task. "
                        "Notice that not all scenes have other objects). "
                        "Please only output a raw JSON dictionary without any markdown formatting, with keys being \"mo\", \"ro\", and \"other\" and values being arrays of bounding box labels. "
                        "The bounding box labels should be the same as the ones you used in the previous step. Do not change, add or remove any bounding box labels from the previous step.")
        })

        response = self.client.chat.completions.create(
            model=model_name,
            messages=self.messages,
            temperature=0.0,
            max_tokens=300
        )
        self.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })

        # Attempt to parse the categorization as a JSON dictionary.
        categories = self.get_valid_json("dict", model_name=model_name)
        print("Categorization response:", self.messages[-1]["content"])
        # dino_objects.append("robot arm")
        categories["robot"] = ["robot arm"]
        dino_objects = categories.get("mo", []) + categories.get("ro", []) + categories.get("other", []) + ["robot arm"]

        return dino_objects, categories


    # Input: an image, and the the name of object
    # Output: bounding boxes, their confidence levels, and the names assigned by DINO
    def _detect_single_object(self, image, dino_prompt, b_t, t_t):
        # print("--------------------single object detection--------------------")
        # print("dino_prompt: ", dino_prompt)
        inputs = self.processor(images=image, text=f"{dino_prompt}.", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=b_t,
            text_threshold=t_t,
            target_sizes=[image.size[::-1]]
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        dino_names = results[0]["labels"]
        # sort the confidences by descending order, and sort input_boxes correspondingly
        sorted_indices = np.argsort(confidences)[::-1]
        confidences = [confidences[i] for i in sorted_indices]
        input_boxes = [input_boxes[i] for i in sorted_indices]
        dino_names = [dino_names[i] for i in sorted_indices]
        # print("input_boxes: ", input_boxes)
        # print("confidences: ", confidences)
        
        # These are ordered by descending confidence
        return input_boxes, confidences, dino_names

    def _detect_objects(self, frame, object_counts, categories, detect_dir, b_t=0.1, t_t=0.1, threshold=0.5, model_name="gpt-4o"):

        
        img_np = np.array(frame)
        orig_img = img_np.copy()
        
        # cv2.imwrite(f"{detect_dir}/orig.png", cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        orig_img_uri = image_to_data_uri(orig_img)
        
        class_names = []
        total_input_boxes = []
        total_confidences = []
        total_dino_names = []
        decisiveness = []

        for obj, count in object_counts.items():

            class_names.append(obj)
            input_box, confidence, dino_name = self._detect_single_object(frame, obj, b_t, t_t)
            total_input_boxes.append(input_box)
            total_confidences.append(confidence)
            total_dino_names.append(dino_name)
            decisiveness.append(self.get_confidence(confidence, count))
        # sort the total_input_boxes, total_confidences, total_class_names by decisiveness
        print("decisiveness: ", decisiveness)

        # These are ordered by descending decisiveness.
        sorted_indices = np.argsort(decisiveness)[::-1]
        class_names = [class_names[i] for i in sorted_indices]
        total_input_boxes = [total_input_boxes[i] for i in sorted_indices]
        total_confidences = [total_confidences[i] for i in sorted_indices]
        total_dino_names = [total_dino_names[i] for i in sorted_indices]
        sorted_decisiveness = [decisiveness[i] for i in sorted_indices]
        object_counts = {class_names[i]: object_counts[class_names[i]] for i in range(len(class_names))}

        final_input_boxes = []
        final_input_boxes_aliases = []
        final_confidences = []
        final_dino_names = []
        final_box_idx = []
        final_class_names = []
        
        similar_count = 0
        similar_boxes = {}
        orig_box_idx = []

        category_lookup = {}
        for cat_key, cat_values in categories.items():
            for val in cat_values:
                category_lookup[val] = cat_key
        print("category_lookup: ", category_lookup)

        self.messages.append({
            "role": "user",
            "content": [
                        {
                            "type": "text",
                            "text": ("Now I have generated bounding boxes for individual objects that you have identified and the robot arm using Grounding DINO. "
                                    "For each object, I am going to give you an ordered list of bounding boxes visualized in the original image. "
                                    "You are going to inspect these images and select the most likely bounding boxes for the object. "
                                    "The number of bounding boxes you select should be the same as the number of objects, which I will give you. "
                                    "I am also going to give you the object names and confidence levels that Grounding DINO has predicted for each bounding box. "
                                    "Due to Grounding DINO's limited performance, these are just for your reference, and you should not strictly follow them when making your selection. "
                                    "You will output the indices (using 0-based index) of the selected bounding boxes as a raw JSON array without any markdown formatting."
                                    "Before we start, please give me a color for the bounding boxes. This color should be distinct from the color of key objects and background in the image. "
                                    "Please output the rgb values in a JSON array, in the RGB order. Do not include any other text or formatting."
                                    "Below is the original image for your reference.")
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": orig_img_uri}
                        }
                    ]

        })

        response = self.client.chat.completions.create(
            model=model_name,
            messages=self.messages,
            temperature=0.0,
            max_tokens=300
        )
        self.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })

        box_color = self.get_valid_json("array", model_name=model_name)
        print("box_color: ", box_color)

        for i, (obj, count) in enumerate(object_counts.items()):
            box_images = []
            orig_idx = []
            # This loop extracts the bounding boxes that are not similar to any of the final bounding boxes
            # and draws them on the image. 
            for j, box in enumerate(total_input_boxes[i]):
                img_copy = img_np.copy()
                print("total_confidences[i][j]: ", total_confidences[i][j])
                
                box_image = self.draw_box(img_copy, [box], box_color=box_color, padding=True)
                cv2.imwrite(f"{detect_dir}/{''.join(ch for ch in obj if not ch.isspace())[:5]}_{i}_{j}.png", cv2.cvtColor(box_image, cv2.COLOR_RGB2BGR))
                similar, input_box, idx = self.get_iou(box, final_input_boxes)
                if similar:
                    print(f"{obj}_{j} is similar to {final_input_boxes_aliases[idx]}")
                    print(f"{obj}_{j}: {box}")
                    print(f"{final_input_boxes_aliases[idx]}: {input_box}")
                    similar_boxes[similar_count] = {
                        "box1": f"{obj}_{j}",
                        "box2": final_input_boxes_aliases[idx]
                    }
                    similar_count += 1
                    continue
                else:
                    orig_idx.append(j)
                    
                    box_images.append(box_image)
            orig_box_idx.append(orig_idx)
            
            selected_boxes = []
            # for j in range(count):
                

            # If the decisiveness is greater than threshold, then we select bounding boxes based on confidence levels.
            if sorted_decisiveness[i] > threshold:
                selected_boxes = orig_idx[:count]
            else:
                
            
                img_uri_list = [image_to_data_uri(img, convert=True) for img in box_images]
                
                img_uri_list.append(orig_img_uri)
                print("img_uri_list: ", len(img_uri_list) - 1)

                # for idx, uri in enumerate(img_uri_list):
                #     # Assuming uri is your data URI string
                #     header, encoded = uri.split(",", 1)
                #     image_bytes = base64.b64decode(encoded)
                #     image = Image.open(io.BytesIO(image_bytes))

                #     # Save the image directly (e.g., as a JPEG file)
                #     image.save(f"{detect_dir}/test_{obj}_{idx}.jpg")
                

                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (f"Now you are going to select bounding boxes for the object \"{obj}\". There are {len(img_uri_list)-1} candidate bounding boxes in total. "
                                        f"The number of objects is {count}. The number of bounding boxes you select must match the number of objects. "
                                        f"You are provided with an ordered list of images, each with one bounding box whose RGB color is {box_color}. The last image is the original image for your reference. "
                                        f"You are also provided with an ordered list of object names predicted by Grounding DINO. {total_dino_names[i]}. "
                                        "These are the names of the objects that are predicted to be in the bounding box. The order of the object names is the same as the order of the images. "
                                        f"You are also provided with an ordered list of confidence levels predicted by Grounding DINO. {total_confidences[i]}. "
                                        "These are the confidence levels of the objects that are predicted to be in the bounding box. The order of the confidence levels is the same as the order of the images. "
                                        "These detected object names and confidence levels are only for your reference. They are not meant to be followed strictly. "
                                        f"The bounding box you select must enclose the object as tightly as possible without including any other objects. "
                                        "For articulated objects, make sure that the bounding box encloses all components of the object. "
                                        f"Please review the bounding boxes and select the one(s) that best represents \"{obj}\". "
                                        "Output a JSON object with a key named selected_index with value being a list of indices of the selected bounding boxes, using 0-based index, "
                                        "and another key named explanation with value being a sentence (a string, not a list or a dictionary) that explains your decision for each bounding box, including those not selected. "
                                        "Do not include any additional text or formatting.")
                            }
                        ] + [
                            {"type": "image_url", "image_url": {"url": img_uri}}
                            for img_uri in img_uri_list
                        ]
                    }
                )
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    temperature=0.0,
                    max_tokens=300
                )
                self.messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content
                })

                

                # Indices of the selected boxes (in a list with repetition removed)
                selected_boxes_raw = self.get_valid_json("dict", model_name=model_name)["selected_index"]
                print("obj: ", obj)
                # explanation = self.get_valid_json("dict")["explanation"]
                print("Selection response:", self.messages[-1]["content"])
                print("original indices: ", orig_idx)
                print("selected indices: ", selected_boxes_raw)
                selected_boxes = [orig_idx[j] for j in selected_boxes_raw]


            print("selected indices (original): ", selected_boxes)
            for j in selected_boxes:
                print("j: ", j)
                final_input_boxes.append(total_input_boxes[i][j])
                final_input_boxes_aliases.append(f"{obj}_{j}")
                # final_confidences.append(total_confidences[i][j])
                final_dino_names.append(total_dino_names[i][j])
                final_class_names.append(obj)
            final_box_idx.append(selected_boxes)
            # for i, j in explanation.items():
            #     print("i: ", i)
            #     box_note[orig_idx[int(i)]] = j
            # box_notes.append(box_note)

        
        # flat_input_boxes = [item for sublist in final_input_boxes for item in sublist]
        # flat_confidences = [item for sublist in final_confidences for item in sublist]

        
        # Convert each word in the list to the corresponding key, if it exists.
        cat_type_names = [category_lookup[obj] for obj in class_names if obj in category_lookup]
        type_names = [category_lookup[obj] for obj in final_class_names if obj in category_lookup]


        
        
        obj_log = []
        for obj, type_name, decisive, box_conf, box_aliases, dino_name, orig_idx in zip(class_names, cat_type_names, sorted_decisiveness, total_confidences, final_box_idx, total_dino_names, orig_box_idx):
            count = object_counts[obj]
            obj_log.append({
                "1. object": obj,
                "2. type": type_name,
                "3. count": count,
                "4. decisive": decisive,
                "5. box_conf": box_conf,
                "6. dino_name": dino_name,
                "7. box_aliases": box_aliases,
                "8. orig_idx": orig_idx
            })

        similar_boxes["final_box_aliases"] = final_input_boxes_aliases
        obj_log.append(similar_boxes)
        


        final_vis = img_np.copy()
        final_vis = self.draw_box(final_vis, final_input_boxes, labels=type_names, box_color=box_color)
        cv2.imwrite(f"{detect_dir}/#final.png", cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))        
            
        pprint.pprint(obj_log)
        return final_input_boxes, type_names, obj_log

    def get_valid_json(self, expected_type, max_retries=3, model_name="gpt-4o"):
        """
        Try to parse the most recent assistant message as JSON.
        If parsing fails or the type is incorrect, prompt ChatGPT to re-send valid raw JSON.
        expected_type: "array" or "dict"
        """
        for attempt in range(max_retries):
            try:
                content = self.messages[-1]["content"]
                print("content: ", content)
                data = json.loads(content)
                if expected_type == "array" and isinstance(data, list):
                    return data
                elif expected_type == "dict" and isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

            retry_prompt = ("Your previous response was not a valid raw JSON {}. "
                            "Please output only a raw JSON {} with no extra text or markdown formatting."
                        ).format("array" if expected_type=="array" else "dictionary",
                                    "array" if expected_type=="array" else "dictionary")
            self.messages.append({
                "role": "user",
                "content": retry_prompt
            })
            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages,
                temperature=0.0,
                max_tokens=300
            )
            self.messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
        raise ValueError("Unable to obtain a valid JSON {} after {} attempts.".format(
            "array" if expected_type=="array" else "dictionary", max_retries))

    def _init_tracking(self, frame, class_names, input_boxes):
        """Initialize SAM tracking with first frame"""
        self.predictor.load_first_frame(np.array(frame))
        
        # The names are sorted by priority.
        for object_id, (label, box) in enumerate(zip(class_names, input_boxes)):
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, 
                obj_id=object_id, 
                bbox=box
            )
        return out_obj_ids, out_mask_logits
        
    def process_frame(self, frame_idx, frames, task_description, log_dir):
        frames = frames.transpose(0,2,3,1)
        
        if frame_idx == 0:
            if self.init:
                self.predictor.reset_state()

            
            # First frame: Initialize tracking with Grounding DINO
            image = Image.fromarray(frames[0])
            # image = np.array(image)
            dino_prompt, categories = self.get_detection_prompt(task_description, image)
            object_counts = self.count_obj(dino_prompt)
            detect_dir = os.path.join(log_dir, "detection", f"frame_{frame_idx}")
            if os.path.exists(detect_dir):
                shutil.rmtree(detect_dir)
            os.makedirs(detect_dir, exist_ok=True)
            input_boxes, class_names, detect_log = self._detect_objects(image, object_counts, categories, detect_dir, threshold=1, model_name="chatgpt-4o-latest")

            print(class_names)
            # Initialize SAM tracking
            out_obj_ids, out_mask_logits = self._init_tracking(image, class_names, input_boxes)
        
            
        else:
            # Subsequent frames: Track objects
            for frame in frames:
                out_obj_ids, out_mask_logits = self.predictor.track(frame)
        print(1111111111111111111)
        print(out_obj_ids)
        object_ids = out_obj_ids
        print("segment ids origin", object_ids)
        class_ids=np.array(object_ids, dtype=np.int32)
        for i in range(len(class_ids)):
            if class_ids[i] == 0:
                class_ids[i] = 1
            else:
                class_ids[i] = 2
        print("vis ids", class_ids)
        masks = [ (out_mask_logits[i] > 0.0).cpu().numpy() for i in range(len(out_mask_logits))]
        masks = np.concatenate(masks, axis=0)
        
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=class_ids,
        )
        mask_annotator = sv.MaskAnnotator(opacity=1.0)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
        annotated_frame = mask_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
        # cv2.imwrite(os.path.join("./tracking_results", f"annotated_frame_{frame_idx:05d}_origin.jpg"), frame_bgr)
        # cv2.imwrite(os.path.join("./tracking_results", f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

        return annotated_frame

    def _reassign_labels(self, class_names):
        """Create consistent integer labels for object types"""
        labels = []
        for i in class_names:
            if i == "robot":
                labels.append(1)
            elif i == "mo":
                labels.append(2)
            elif i == "ro":
                labels.append(3)
            else:
                labels.append(0)
        return labels
        
    # def close(self):
    #     self.socket.send_json({"command": "exit"})
    #     _ = self.socket.recv_json()  # Wait for acknowledgment
    #     self.socket.close()
    #     self.context.term()
    
    # def __del__(self):
    #     self.close()
    def draw_box(self, image, boxes, labels=None, confs=None, box_color=[255, 0, 0], padding=False, thickness=4):
        print("boxes: ", len(boxes))
        
        if labels is not None:
            print("labels: ", labels)
        if confs is not None:
            print("confs: ", len(confs))
        
        # Ensure consistency between boxes, labels, and confs if provided
        if labels is not None and confs is not None:
            assert len(boxes) == len(labels) == len(confs), (
                f"Length mismatch. boxes: {len(boxes)}, labels: {len(labels)}, confs: {len(confs)}"
            )
        
        # Get image dimensions (assuming image is a numpy array with shape [height, width, channels])
        h, w = image.shape[:2]
        
        for i, box in enumerate(boxes):
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            box_width = x2 - x1
            box_height = y2 - y1
            if padding:
                if (box_width > 85 or box_height > 85):
                    pad = 8
                else:
                    pad = 4
                if (box_width > w - 10 or box_height > h - 10):
                    pad = 0
            else:
                pad = 0
            
            # Expand the box by the calculated padding while clamping to image boundaries
            p_x1 = max(0, x1 - pad)
            p_y1 = max(0, y1 - pad)
            p_x2 = min(w, x2 + pad)
            p_y2 = min(h, y2 + pad)

            # Adjust coordinates to ensure the entire boundary line is visible.
            # Since the line has a thickness of 4, we set a margin of thickness//2 = 2.
            margin = thickness // 2
            p_x1 = max(p_x1, margin)
            p_y1 = max(p_y1, margin)
            p_x2 = min(p_x2, w - margin)
            p_y2 = min(p_y2, h - margin)
            
            # Draw the padded bounding box
            cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), box_color, thickness)
            
            if labels is not None:
                # Prepare the label text
                label_text = f"{labels[i]}"
                if confs is not None:
                    label_text = f"{labels[i]}: {confs[i]:.2f}"
                print("label_text: ", label_text)
                
                # Get text size for the label
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Draw a filled rectangle as the background for the text label
                cv2.rectangle(image, (p_x1, p_y1 - text_height - 4), (p_x1 + text_width, p_y1), (255, 255, 255), -1)
                
                # Draw the label text above the bounding box
                cv2.putText(image, label_text, (p_x1, p_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return image

    def count_obj(self, objects) -> dict:
        print("objects: ", objects)
        counts = {}
        for obj in objects:
            counts[obj] = counts.get(obj, 0) + 1  # Increment the count for each word.
        return counts
        
    def get_confidence(self, confidence, count):
        # This is a measure of how confident the model is about the detection of a certain object in general.
        # It is calculated by the difference between the confidence of the last object and the confidence of the current object.
        # The higher the difference, the more confident the model is about the detection of the current object.
        if len(confidence) <= count:
            return 1
        else:
            return (confidence[count-1] - confidence[count]) / confidence[count-1]

    def get_iou(self, box, input_boxes, threshold=0.9, pixel_tolerance=5):
        # Determines whether box is similar to any of the input_boxes.
        # If the iou is greater than the threshold, it is considered similar.
        for idx, input_box in enumerate(input_boxes):
            iou = self.get_iou_single(box, input_box, pixel_tolerance)
            if iou > threshold:
                return True, input_box, idx
        return False, None, None

    def get_iou_single(self, box1, box2, pixel_tolerance=5):
        """
        Determines whether two bounding boxes are essentially the same.

        Args:
            box1 (list): Bounding box in the format [x_min, y_min, x_max, y_max].
            box2 (list): Bounding box in the format [x_min, y_min, x_max, y_max].
            iou_threshold (float): Minimum IoU threshold to consider the boxes identical.
            pixel_tolerance (int): Allowable difference in pixel values for each coordinate.

        Returns:
            bool: True if the bounding boxes are considered the same, False otherwise.
        """
        # Ensure box format is valid
        assert len(box1) == 4 and len(box2) == 4, f"Each bounding box must contain 4 values. box1: {len(box1)}, box2: {len(box2)}"

        # Check if all coordinates are within the pixel tolerance
        if all(abs(c1 - c2) <= pixel_tolerance for c1, c2 in zip(box1, box2)):
            return True

        # Compute intersection box
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        # Compute intersection area
        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        intersection_area = inter_width * inter_height

        # Compute areas of both boxes
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute union area
        union_area = area_box1 + area_box2 - intersection_area

        # Compute IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        # Compare with threshold
        return iou

def image_to_data_uri(image, convert=False):
    """
    Convert a NumPy image (e.g., from cv2) to a data URI.
    """
    # Convert from BGR (OpenCV default) to RGB
    image = np.array(image)
    if convert:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Encode the image as JPEG in memory
    retval, buffer = cv2.imencode('.jpg', image)
    if not retval:
        raise ValueError("Image encoding failed.")
    
    # Base64 encode the image bytes
    img_base64 = base64.b64encode(buffer).decode()
    
    # Create a data URI
    return f"data:image/jpeg;base64,{img_base64}"