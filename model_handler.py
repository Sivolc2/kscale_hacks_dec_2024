import anthropic
from typing import List, Dict, Optional, Any, Tuple
import re
import base64
import cv2
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class ModelHandler:
    """Handles all interactions with AI models"""
    
    def __init__(self, model_name="claude-3-sonnet-20240229", max_retries=3):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_api_call(self, messages):
        """Make API call with retry logic"""
        try:
            return self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=messages
            )
        except Exception as e:
            print(f"API call failed: {str(e)}")
            raise

    def analyze_scene(self, 
                     image_data: str, 
                     available_skills: str,
                     voice_command: Optional[str] = None) -> Tuple[str, str]:
        """
        Analyze scene and determine next action
        
        Args:
            image_data: Base64 encoded image data
            available_skills: Description of available skills
            voice_command: Optional voice command to consider
            
        Returns:
            Tuple[str, str]: (selected_skill, objective)
        """
        base_prompt = """Given an external view of a robot"""
        if voice_command:
            base_prompt += f" and the voice command: '{voice_command}'"
            
        base_prompt += f""", analyze the scene and determine:
        1. What is the most appropriate next action?
        2. What specific objective needs to be achieved?

        Available skills:
        {available_skills}
        
        You MUST format your response EXACTLY as shown:
        Scene description: <brief description>
        Selected skill: [skill_name]
        Objective: <specific, measurable objective for this skill>

        IMPORTANT:
        - Selected skill MUST be from the available skills list
        - Objective MUST be specific and measurable
        - Objective should define a clear end state
        """

        message = self._make_api_call([{
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
                    "text": base_prompt
                }
            ],
        }])
        
        response = message.content[0].text if isinstance(message.content, list) else message.content
        
        # Extract skill and objective
        skill_match = re.search(r'Selected skill: \[([^\]]+)\]', response)
        objective_match = re.search(r'Objective: (.+)$', response, re.MULTILINE)
        
        if not skill_match or not objective_match:
            raise ValueError("Could not parse skill or objective from response")
            
        return skill_match.group(1), objective_match.group(1)

    def action_agent(self, 
                    skill_name: str,
                    objective: str,
                    image_data: str,
                    previous_attempts: int = 0,
                    context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate current state and decide if more skill invocations are needed
        """
        prompt = f"""You are evaluating the progress of a robot skill execution.

Skill: [{skill_name}]
Objective: {objective}
Attempts so far: {previous_attempts}

Based on the image, determine if:
1. The objective has been achieved
2. If not, should we try the skill again?

YOU MUST RESPOND EXACTLY IN THIS FORMAT:
Analysis: <brief description of what you see>
Continue execution: <YES or NO>
Additional invocations: <number between 1-3 if continuing, or 0 if complete>
Reason: <brief explanation of your decision>

Example response:
Analysis: Robot is partially standing but leaning left
Continue execution: YES
Additional invocations: 1
Reason: Need one more attempt to achieve stable standing position
"""

        try:
            # Make API call
            response = self._make_api_call([{
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
                        "text": prompt
                    }
                ],
            }])
            
            # Extract text from response
            response_text = response.content[0].text if isinstance(response.content, list) else response.content
            
            # Print full response for debugging
            print("\nFull response from model:")
            print(response_text)
            print("-" * 50)
            
            # Extract decision parameters with more flexible parsing
            continue_match = re.search(r'Continue execution:\s*(YES|NO)', response_text, re.IGNORECASE)
            invocations_match = re.search(r'Additional invocations:\s*(\d+)', response_text)
            reason_match = re.search(r'Reason:\s*(.+?)(?:\n|$)', response_text, re.DOTALL)
            analysis_match = re.search(r'Analysis:\s*(.+?)(?:\n|$)', response_text, re.DOTALL)
            
            # Default values if parsing fails
            continue_execution = True  # Default to continue
            num_invocations = 1      # Default to 1 more try
            reason = "Format error - defaulting to one more attempt"
            analysis = "Could not parse analysis"
            
            # Override defaults with parsed values if available
            if continue_match:
                continue_execution = continue_match.group(1).upper() == "YES"
            
            if continue_execution and invocations_match:
                num_invocations = min(int(invocations_match.group(1)), 3)
            elif not continue_execution:
                num_invocations = 0
                
            if reason_match:
                reason = reason_match.group(1).strip()
                
            if analysis_match:
                analysis = analysis_match.group(1).strip()
                
            # If this is not the first attempt, be more conservative
            if previous_attempts >= 2:
                continue_execution = False
                num_invocations = 0
                reason = f"Maximum attempts ({previous_attempts}) reached"
            
            return {
                "continue_execution": continue_execution,
                "num_invocations": num_invocations,
                "reason": reason,
                "analysis": analysis,
                "full_response": response_text
            }
                
        except Exception as e:
            print(f"\nError in action_agent: {str(e)}")
            if 'response_text' in locals():
                print(f"Raw response: {response_text}")
                
            # Default fallback response
            return {
                "continue_execution": previous_attempts < 2,  # Only continue if less than 2 attempts
                "num_invocations": 1 if previous_attempts < 2 else 0,
                "reason": f"Error parsing response: {str(e)} - defaulting to {'continue' if previous_attempts < 2 else 'stop'}",
                "analysis": "Error parsing model response",
                "full_response": response_text if 'response_text' in locals() else "No response"
            }

    def action_agent_compare(self, 
                            skill_name: str,
                            objective: str,
                            initial_image_data: str,
                            current_image_data: str,
                            previous_attempts: int = 0,
                            context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate progress by comparing initial and current states
        """
        prompt = f"""You are evaluating the progress of a robot skill execution by comparing before and after images.

Skill: [{skill_name}]
Objective: {objective}
Attempts so far: {previous_attempts}

Compare the initial state (Image 1) with the current state (Image 2) to determine if:
1. The objective has been achieved
2. What changes have occurred
3. If more attempts are needed

RESPOND EXACTLY IN THIS FORMAT:
Initial state: <brief description of first image>
Current state: <brief description of second image>
Changes observed: <list key differences>
Continue execution: <YES or NO>
Additional invocations: <number between 1-3 if continuing, or 0 if complete>
Reason: <brief explanation of your decision>
"""

        try:
            # Make API call with both images
            response = self._make_api_call([{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Initial state:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": initial_image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Current state:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": current_image_data,
                        },
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    }
                ],
            }])
            
            # Extract text from response
            response_text = response.content[0].text if isinstance(response.content, list) else response.content
            
            # Print full response for debugging
            print("\nFull response from model:")
            print(response_text)
            print("-" * 50)
            
            # Extract decision parameters
            continue_match = re.search(r'Continue execution:\s*(YES|NO)', response_text, re.IGNORECASE)
            invocations_match = re.search(r'Additional invocations:\s*(\d+)', response_text)
            reason_match = re.search(r'Reason:\s*(.+?)(?:\n|$)', response_text, re.DOTALL)
            changes_match = re.search(r'Changes observed:\s*(.+?)(?:\n|$)', response_text, re.DOTALL)
            
            if not continue_match:
                raise ValueError("Could not find 'Continue execution' in response")
            if not reason_match:
                raise ValueError("Could not find 'Reason' in response")
                
            continue_execution = continue_match.group(1).upper() == "YES"
            
            # Default to 1 invocation if continuing but no number specified
            num_invocations = int(invocations_match.group(1)) if continue_execution and invocations_match else (1 if continue_execution else 0)
            
            # Limit number of additional invocations
            num_invocations = min(num_invocations, 3)
            
            return {
                "continue_execution": continue_execution,
                "num_invocations": num_invocations,
                "reason": reason_match.group(1).strip(),
                "changes": changes_match.group(1).strip() if changes_match else "No changes described",
                "full_response": response_text
            }
                
        except Exception as e:
            print(f"\nError parsing action agent response: {str(e)}")
            if 'response_text' in locals():
                print(f"Raw response: {response_text}")
            raise ValueError(f"Could not parse action agent response: {str(e)}")

    def plan_tasks(self,
                  available_skills: str,
                  image_data: Optional[str] = None, 
                  voice_command: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Plan sequence of tasks based on image and/or voice input
        
        Args:
            available_skills: Description of available skills
            image_data: Optional base64 encoded image data
            voice_command: Optional voice command to consider
            
        Returns:
            List[Tuple[str, str]]: List of (skill_name, objective) pairs
        """
        try:
            base_prompt = """Plan the sequence of robot actions needed."""
            
            if voice_command:
                base_prompt += f"\nVoice command: '{voice_command}'"
                
            base_prompt += f"""\n
    Available skills:
    {available_skills}

    You MUST format your response as a sequence of skill-objective pairs:
    - [skill_name]: specific objective
    - [skill_name]: specific objective
    ...

    IMPORTANT:
    - Each skill MUST be from the available skills list
    - Each objective MUST be specific and measurable
    - Order the skills in the sequence they should be executed
    """

            # Construct message content
            content = []
            if image_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                })
            content.append({
                "type": "text", 
                "text": base_prompt
            })

            # Make API call with retries
            message = self._make_api_call([{
                "role": "user",
                "content": content,
            }])
            
            response = message.content[0].text if isinstance(message.content, list) else message.content
            
            # Extract skill-objective pairs using regex
            pairs = re.findall(r'-\s*\[([^\]]+)\]:\s*(.+?)(?=\n|$)', response)
            
            if not pairs:
                raise ValueError("Could not parse skill-objective pairs from response")
            
            return pairs
                
        except Exception as e:
            print(f"Error in plan_tasks: {str(e)}")
            if 'response' in locals():
                print(f"Raw response: {response}")
            raise