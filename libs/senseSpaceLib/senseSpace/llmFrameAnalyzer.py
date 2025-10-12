"""
Generic LLM Frame Analyzer for SenseSpace
Connects frame data to LLM expert system based on configuration
"""

from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.interpretation import interpret_pose_from_angles
from senseSpaceLib.senseSpace.llmClient import LLMClient


class LLMFrameAnalyzer:
    """Generic analyzer that processes SenseSpace frames using LLM expert system"""
    
    def __init__(self, llm_client: LLMClient, confidence_threshold=70.0):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.cur_people = []
        self.latest_frame = None
        self.response_callback = None  # Optional callback for responses
        
    def on_init(self):
        """Initialize LLM client"""
        self.llm_client.on_init()
        return self.llm_client.ollama_ready
    
    def on_frame(self, frame: Frame):
        """Process incoming frame and update current people"""
        self.latest_frame = frame
        people = getattr(frame, "people", None)
        
        if people:
            self.cur_people = [p for p in people 
                             if getattr(p, 'confidence', 0) > self.confidence_threshold]
        else:
            self.cur_people = []
    
    def analyze_current_pose(self, custom_prompt: str = None):
        """
        Analyze current pose using LLM expert system
        
        Args:
            custom_prompt: Optional custom prompt to append to pose data
        """
        if not self.cur_people:
            return None
        
        person = self.cur_people[0]
        angles = person.get_skeletal_angles()
        pose_description = interpret_pose_from_angles(angles)
        
        # Build expert input (format controlled by expert config)
        expert_input = self._format_expert_input(person, pose_description, custom_prompt)
        
        # Call LLM expert asynchronously
        self.llm_client.call_expert_async(
            expert_input,
            callback=self._handle_response
        )
        
        return pose_description
    
    def _format_expert_input(self, person, pose_description: str, custom_prompt: str = None) -> str:
        """Format input according to expert system expectations"""
        base_input = f"""Person ID: {person.id}
Confidence: {person.confidence:.1f}%

Skeletal Analysis:
{pose_description}"""
        
        if custom_prompt:
            return f"{base_input}\n\n{custom_prompt}"
        else:
            return base_input
    
    def _handle_response(self, response: str):
        """Handle LLM response"""
        if self.response_callback:
            self.response_callback(response)
    
    def set_response_callback(self, callback):
        """Set callback for handling responses"""
        self.response_callback = callback
    
    def reset_context(self):
        """Reset LLM conversation context"""
        self.llm_client.reset_context()