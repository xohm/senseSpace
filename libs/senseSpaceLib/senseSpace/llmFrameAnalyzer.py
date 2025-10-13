"""
Generic LLM Frame Analyzer for SenseSpace
Connects frame data to LLM expert system based on configuration
"""

from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.interpretation import interpret_pose_from_angles
from senseSpaceLib.senseSpace.llmClient import LLMClient


class LLMFrameAnalyzer:
    """Generic analyzer that processes SenseSpace frames using LLM expert system"""
    
    def __init__(self, llm_client: LLMClient, confidence_threshold=70.0, verbose=False):
        """
        Initialize the LLM Frame Analyzer
        
        Args:
            llm_client: LLMClient instance for processing
            confidence_threshold: Minimum confidence for person detection
            verbose: Enable detailed debug output
        """
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.cur_people = []
        self.response_callback = None
        
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
            print("[WARN] No person detected in frame")
            return None
        
        if self.verbose:
            print(f"[REQUEST] Analyzing {len(self.cur_people)} person(s)...")
        
        # Handle multiple people
        if len(self.cur_people) > 1:
            # Build combined input for all people
            all_descriptions = []
            for i, person in enumerate(self.cur_people):
                angles = person.get_skeletal_angles()
                pose_desc = interpret_pose_from_angles(angles)
                all_descriptions.append(f"Person {i+1}:\n{pose_desc}")
            
            combined_input = "\n\n".join(all_descriptions)
            expert_input = self._format_expert_input(None, combined_input, custom_prompt)
            
            # Call LLM expert asynchronously
            self.llm_client.call_expert_async(
                expert_input,
                callback=self._handle_response
            )
            
            return combined_input
        else:
            # Single person (existing code)
            person = self.cur_people[0]
            angles = person.get_skeletal_angles()
            pose_description = interpret_pose_from_angles(angles)
            
            # Build expert input
            expert_input = self._format_expert_input(person, pose_description, custom_prompt)
            
            # Call LLM expert asynchronously
            self.llm_client.call_expert_async(
                expert_input,
                callback=self._handle_response
            )
            
            return pose_description
    
    def _format_expert_input(self, person, pose_description: str, custom_prompt: str = None) -> str:
        """Format input - send full pose description"""
        if custom_prompt:
            return f"{pose_description}\n\n{custom_prompt}"
        
        return pose_description
    
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
    
    def get_leg_azimuth_elevation(self, leg: str = 'left') -> tuple:
        """
        Get azimuth and elevation for a specific leg
        
        Args:
            leg: 'left' or 'right'
            
        Returns:
            Tuple of (azimuth, elevation) in degrees, or (None, None) if not available
        """
        if not self.cur_people:
            return (None, None)
        
        person = self.cur_people[0]
        angles = person.get_skeletal_angles()
        
        # Get leg azimuth and elevation from angles dict
        leg_prefix = 'left' if leg.lower() == 'left' else 'right'
        
        azimuth = angles.get(f'{leg_prefix}_leg_azimuth')
        elevation = angles.get(f'{leg_prefix}_leg_elevation')
        
        return (azimuth, elevation)
    
    def get_arm_azimuth_elevation(self, arm: str = 'left') -> tuple:
        """
        Get azimuth and elevation for a specific arm
        
        Args:
            arm: 'left' or 'right'
            
        Returns:
            Tuple of (azimuth, elevation) in degrees, or (None, None) if not available
        """
        if not self.cur_people:
            return (None, None)
        
        person = self.cur_people[0]
        angles = person.get_skeletal_angles()
        
        # Get arm azimuth and elevation from angles dict
        arm_prefix = 'left' if arm.lower() == 'left' else 'right'
        
        azimuth = angles.get(f'{arm_prefix}_arm_azimuth')
        elevation = angles.get(f'{arm_prefix}_arm_elevation')
        
        return (azimuth, elevation)
    
    def get_all_limb_orientations(self) -> dict:
        """
        Get azimuth and elevation for all limbs
        
        Returns:
            Dictionary with azimuth/elevation for left/right arms and legs
        """
        if not self.cur_people:
            return {}
        
        person = self.cur_people[0]
        angles = person.get_skeletal_angles()
        
        return {
            'left_arm': {
                'azimuth': angles.get('left_arm_azimuth'),
                'elevation': angles.get('left_arm_elevation')
            },
            'right_arm': {
                'azimuth': angles.get('right_arm_azimuth'),
                'elevation': angles.get('right_arm_elevation')
            },
            'left_leg': {
                'azimuth': angles.get('left_leg_azimuth'),
                'elevation': angles.get('left_leg_elevation')
            },
            'right_leg': {
                'azimuth': angles.get('right_leg_azimuth'),
                'elevation': angles.get('right_leg_elevation')
            }
        }
    
    def analyze_pose_ultra_brief(self, custom_prompt: str = None):
        """
        Analyze current pose using ultra-brief LLM expert system
        
        Args:
            custom_prompt: Optional custom prompt to append to pose data
        """
        if not self.cur_people:
            return None
        
        person = self.cur_people[0]
        angles = person.get_skeletal_angles()
        pose_description = interpret_pose_from_angles(angles)
        
        # Build ultra-brief expert input
        expert_input = self._format_ultra_brief_expert_input(person, pose_description, custom_prompt)
        
        # Call LLM expert asynchronously
        self.llm_client.call_expert_async(
            expert_input,
            callback=self._handle_response
        )
        
        return pose_description
    
    def _format_ultra_brief_expert_input(self, person, pose_description: str, custom_prompt: str = None) -> str:
        """Format input for ultra-brief expert system"""
        body_state = pose_description.split('\n')[0]  # Take the first line as body state
        
        # Construct the input for the LLM
        base_input = f"Subject detected (confidence: {person.confidence:.1f}%): {body_state}"
        
        if custom_prompt:
            return f"{base_input} | {custom_prompt}"
        else:
            return base_input