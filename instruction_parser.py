"""
Text Instruction Parser for Scene Manipulation

This module parses natural language instructions and converts them into
structured actions that can be executed by the scene manipulation pipeline.
"""

import re
import spacy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions that can be performed on objects."""
    MOVE = "move"
    RELOCATE = "relocate"
    RELIGHT = "relight"
    ADD_LIGHTING = "add_lighting"
    REMOVE = "remove"
    ADD = "add"


class LightingType(Enum):
    """Types of lighting conditions."""
    SUNSET = "sunset"
    GOLDEN_HOUR = "golden_hour"
    DRAMATIC = "dramatic"
    SOFT = "soft"
    HARSH = "harsh"
    NATURAL = "natural"
    ARTIFICIAL = "artificial"
    WARM = "warm"
    COOL = "cool"


class SpatialReference(Enum):
    """Spatial reference directions."""
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    FRONT = "front"
    BACK = "back"


@dataclass
class ObjectReference:
    """Reference to an object in the scene."""
    name: str
    attributes: List[str] = None  # e.g., ["red", "large", "old"]
    position: Optional[SpatialReference] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = []


@dataclass
class Action:
    """Structured action extracted from text instruction."""
    action_type: ActionType
    target_object: ObjectReference
    destination: Optional[SpatialReference] = None
    lighting_type: Optional[LightingType] = None
    intensity: float = 1.0  # 0.0 to 1.0
    confidence: float = 0.8


@dataclass
class ParsedInstruction:
    """Complete parsed instruction with all extracted information."""
    original_text: str
    actions: List[Action]
    global_lighting: Optional[LightingType] = None
    scene_context: Dict[str, any] = None
    
    def __post_init__(self):
        if self.scene_context is None:
            self.scene_context = {}


class InstructionParser:
    """
    Main parser for converting natural language instructions into structured actions.
    
    This parser uses spaCy for NLP tasks and rule-based extraction for
    identifying objects, actions, and spatial references.
    """
    
    def __init__(self):
        """Initialize the parser with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to basic English model
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        
        # Define patterns for different action types
        self.move_patterns = [
            r"move\s+(\w+)\s+to\s+(\w+)",
            r"relocate\s+(\w+)\s+to\s+(\w+)",
            r"shift\s+(\w+)\s+to\s+(\w+)",
            r"put\s+(\w+)\s+(\w+)",
        ]
        
        self.lighting_patterns = [
            r"add\s+(\w+)\s+lighting",
            r"apply\s+(\w+)\s+lighting",
            r"change\s+lighting\s+to\s+(\w+)",
            r"make\s+it\s+(\w+)\s+lighting",
        ]
        
        # Object attribute patterns
        self.attribute_patterns = [
            r"(\w+)\s+(\w+)\s+(\w+)",  # "red large car"
            r"(\w+)\s+(\w+)",          # "red car"
        ]
    
    def parse(self, instruction: str) -> ParsedInstruction:
        """
        Parse a natural language instruction into structured actions.
        
        Args:
            instruction: Natural language instruction string
            
        Returns:
            ParsedInstruction object with extracted actions and context
        """
        instruction = instruction.lower().strip()
        doc = self.nlp(instruction)
        
        actions = []
        global_lighting = None
        
        # Extract move/relocate actions
        move_actions = self._extract_move_actions(instruction, doc)
        actions.extend(move_actions)
        
        # Extract lighting actions
        lighting_actions = self._extract_lighting_actions(instruction, doc)
        actions.extend(lighting_actions)
        
        # Extract global lighting changes
        global_lighting = self._extract_global_lighting(instruction, doc)
        
        # Extract scene context
        scene_context = self._extract_scene_context(instruction, doc)
        
        return ParsedInstruction(
            original_text=instruction,
            actions=actions,
            global_lighting=global_lighting,
            scene_context=scene_context
        )
    
    def _extract_move_actions(self, instruction: str, doc) -> List[Action]:
        """Extract move/relocate actions from the instruction."""
        actions = []
        
        for pattern in self.move_patterns:
            matches = re.finditer(pattern, instruction)
            for match in matches:
                if len(match.groups()) >= 2:
                    object_name = match.group(1)
                    destination = match.group(2)
                    
                    # Parse object reference
                    obj_ref = self._parse_object_reference(object_name, doc)
                    
                    # Parse destination
                    spatial_ref = self._parse_spatial_reference(destination)
                    
                    action = Action(
                        action_type=ActionType.MOVE,
                        target_object=obj_ref,
                        destination=spatial_ref
                    )
                    actions.append(action)
        
        return actions
    
    def _extract_lighting_actions(self, instruction: str, doc) -> List[Action]:
        """Extract lighting-related actions from the instruction."""
        actions = []
        
        for pattern in self.lighting_patterns:
            matches = re.finditer(pattern, instruction)
            for match in matches:
                if len(match.groups()) >= 1:
                    lighting_desc = match.group(1)
                    lighting_type = self._parse_lighting_type(lighting_desc)
                    
                    # Try to find target object for lighting
                    target_obj = self._extract_lighting_target(instruction, doc)
                    
                    action = Action(
                        action_type=ActionType.ADD_LIGHTING,
                        target_object=target_obj,
                        lighting_type=lighting_type
                    )
                    actions.append(action)
        
        return actions
    
    def _extract_global_lighting(self, instruction: str, doc) -> Optional[LightingType]:
        """Extract global lighting changes that affect the entire scene."""
        global_lighting_keywords = [
            "entire scene", "whole scene", "scene", "everything"
        ]
        
        for keyword in global_lighting_keywords:
            if keyword in instruction:
                # Look for lighting type near the keyword
                for lighting_type in LightingType:
                    if lighting_type.value in instruction:
                        return lighting_type
        
        return None
    
    def _parse_object_reference(self, object_name: str, doc) -> ObjectReference:
        """Parse object reference and extract attributes."""
        attributes = []
        
        # Look for adjectives describing the object
        for token in doc:
            if token.pos_ == "ADJ" and token.text in object_name:
                attributes.append(token.text)
        
        # Extract color, size, and other attributes
        color_patterns = ["red", "blue", "green", "yellow", "black", "white", "gray"]
        size_patterns = ["large", "small", "big", "tiny", "huge"]
        
        for pattern in color_patterns + size_patterns:
            if pattern in object_name:
                attributes.append(pattern)
        
        return ObjectReference(
            name=object_name,
            attributes=attributes
        )
    
    def _parse_spatial_reference(self, spatial_text: str) -> Optional[SpatialReference]:
        """Parse spatial reference from text."""
        spatial_mapping = {
            "left": SpatialReference.LEFT,
            "right": SpatialReference.RIGHT,
            "center": SpatialReference.CENTER,
            "middle": SpatialReference.CENTER,
            "top": SpatialReference.TOP,
            "bottom": SpatialReference.BOTTOM,
            "foreground": SpatialReference.FOREGROUND,
            "background": SpatialReference.BACKGROUND,
            "front": SpatialReference.FRONT,
            "back": SpatialReference.BACK,
        }
        
        return spatial_mapping.get(spatial_text.lower())
    
    def _parse_lighting_type(self, lighting_desc: str) -> Optional[LightingType]:
        """Parse lighting type from description."""
        lighting_mapping = {
            "sunset": LightingType.SUNSET,
            "golden": LightingType.GOLDEN_HOUR,
            "dramatic": LightingType.DRAMATIC,
            "soft": LightingType.SOFT,
            "harsh": LightingType.HARSH,
            "natural": LightingType.NATURAL,
            "artificial": LightingType.ARTIFICIAL,
            "warm": LightingType.WARM,
            "cool": LightingType.COOL,
        }
        
        return lighting_mapping.get(lighting_desc.lower())
    
    def _extract_lighting_target(self, instruction: str, doc) -> ObjectReference:
        """Extract target object for lighting changes."""
        # Look for objects mentioned in the instruction
        objects = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                objects.append(token.text)
        
        if objects:
            return ObjectReference(name=objects[0])
        else:
            return ObjectReference(name="scene")  # Default to entire scene
    
    def _extract_scene_context(self, instruction: str, doc) -> Dict[str, any]:
        """Extract additional scene context information."""
        context = {}
        
        # Extract time of day
        time_keywords = ["morning", "afternoon", "evening", "night", "day"]
        for keyword in time_keywords:
            if keyword in instruction:
                context["time_of_day"] = keyword
        
        # Extract weather conditions
        weather_keywords = ["sunny", "cloudy", "rainy", "foggy", "stormy"]
        for keyword in weather_keywords:
            if keyword in instruction:
                context["weather"] = keyword
        
        return context


# Example usage and testing
if __name__ == "__main__":
    parser = InstructionParser()
    
    # Test cases
    test_instructions = [
        "Move the red car to the left and add sunset lighting",
        "Relocate the person to the center",
        "Add golden hour lighting to the entire scene",
        "Move the tree to the background and apply dramatic shadows"
    ]
    
    for instruction in test_instructions:
        print(f"\nInstruction: {instruction}")
        result = parser.parse(instruction)
        print(f"Actions: {len(result.actions)}")
        for action in result.actions:
            print(f"  - {action.action_type.value}: {action.target_object.name}")
            if action.destination:
                print(f"    Destination: {action.destination.value}")
            if action.lighting_type:
                print(f"    Lighting: {action.lighting_type.value}") 