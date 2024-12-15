import anthropic
from typing import List, Dict, Optional, Any
import re

class ModelHandler:
    """Handles all interactions with AI models"""
    
    def __init__(self, model_name="claude-3-sonnet-20240229"):
        self.client = anthropic.Anthropic()
        self.model_name = model_name

    def plan_task(self, task_description: str, available_skills: str) -> str:
        """
        Get task planning from the model
        Args:
            task_description: Description of the task to plan
            available_skills: Description of available skills
        Returns:
            str: Model's response with planned steps
        """
        prompt = f"""You are a helpful robot assistant that can execute the following skills:

{available_skills}

For the task of moving to another spot: "{task_description}"

Break this down into a sequence of skills from the above list. You MUST format your response exactly as shown:
1. [skill_name]: Brief justification
2. [skill_name]: Brief justification
...

IMPORTANT: Each skill name MUST be enclosed in square brackets []. 
Only use skills from the provided list. Be concise but clear in your justifications.
If there is nothing to do, respond with: 1. [wave]: Default greeting action
"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{
                "role": "user", 
                "content": prompt
            }]
        )
        
        return message.content[0].text if isinstance(message.content, list) else message.content

    def analyze_scene(self, 
                     image_data: str, 
                     available_skills: str,
                     voice_command: Optional[str] = None) -> str:
        """
        Analyze scene from image and optional voice command
        Args:
            image_data: Base64 encoded image data
            available_skills: Description of available skills
            voice_command: Optional voice command to consider
        Returns:
            str: Model's analysis and planned steps
        """
        base_prompt = """Given an external view of a robot"""
        if voice_command:
            base_prompt += f" and the voice command: '{voice_command}'"
        base_prompt += """. What task needs to be done?
        After describing the scene, break down the needed task into specific steps using available skills.
        
        Available skills:
        {skills}
        
        You MUST format your response EXACTLY as shown:
        Scene description: <brief description>
        Required task: <task description>
        Planned steps:
        1. [skill_name]: Brief justification
        2. [skill_name]: Brief justification
        ...

        IMPORTANT RULES:
        - Each skill name MUST be enclosed in square brackets []
        - Only use skills from the provided list
        - Each step MUST start with a number followed by a period
        - Each step MUST have exactly one pair of square brackets
        - Each step MUST have a colon and justification after the brackets"""

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text", 
                        "text": base_prompt.format(skills=available_skills)
                    }
                ],
            }]
        )
        
        return message.content[0].text if isinstance(message.content, list) else message.content

    @staticmethod
    def extract_skills(response: str, available_skills: Dict[str, Any]) -> List[str]:
        """
        Extract skill names from model response
        Args:
            response: Model's response text
            available_skills: Dictionary of available skills
        Returns:
            List[str]: List of valid skill names
        """
        skills = []
        for line in response.split('\n'):
            matches = re.findall(r'^\d+\.\s*\[([^\]]+)\]:', line)
            if matches:
                skill_name = matches[0].strip()
                if skill_name in available_skills:
                    skills.append(skill_name)
                else:
                    print(f"Warning: Unknown skill '{skill_name}' found in response")
        return skills

    @staticmethod
    def validate_response(response: str) -> List[str]:
        """
        Validate model response format and extract valid steps
        Args:
            response: Model's response text
        Returns:
            List[str]: List of valid step strings
        """
        if "Planned steps:" not in response:
            return []
            
        steps = response.split("Planned steps:")[1].strip().split("\n")
        valid_steps = []
        
        for step in steps:
            if re.match(r'^\d+\.\s*\[[^\]]+\]:', step):
                valid_steps.append(step)
            else:
                print(f"Warning: Invalid step format: {step}")
                
        return valid_steps 