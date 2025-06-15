"""
Action Extractor for Scene Manipulation

This module provides utilities for extracting specific types of actions
and validating parsed instructions.
"""

import re
from typing import List, Dict, Optional, Tuple
from .instruction_parser import Action, ActionType, ObjectReference, SpatialReference, LightingType


class ActionExtractor:
    """
    Utility class for extracting and validating specific types of actions
    from parsed instructions.
    """
    
    def __init__(self):
        """Initialize the action extractor."""
        self.valid_objects = {
            "car", "person", "tree", "house", "building", "chair", "table",
            "dog", "cat", "bird", "flower", "grass", "road", "sky", "cloud",
            "mountain", "river", "lake", "boat", "bike", "bus", "truck"
        }
        
        self.valid_attributes = {
            "red", "blue", "green", "yellow", "black", "white", "gray",
            "large", "small", "big", "tiny", "huge", "old", "new", "tall",
            "short", "wide", "narrow", "bright", "dark", "shiny", "dull"
        }
    
    def extract_move_actions(self, actions: List[Action]) -> List[Action]:
        """Extract only move/relocate actions from a list of actions."""
        return [action for action in actions if action.action_type in [ActionType.MOVE, ActionType.RELOCATE]]
    
    def extract_lighting_actions(self, actions: List[Action]) -> List[Action]:
        """Extract only lighting-related actions from a list of actions."""
        return [action for action in actions if action.action_type in [ActionType.RELIGHT, ActionType.ADD_LIGHTING]]
    
    def validate_action(self, action: Action) -> Tuple[bool, List[str]]:
        """
        Validate an action for correctness and completeness.
        
        Args:
            action: Action to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate target object
        if not action.target_object.name:
            errors.append("Target object name is missing")
        
        # Validate object name is reasonable
        if action.target_object.name not in self.valid_objects:
            # Check if it's a compound object (e.g., "red car")
            compound_valid = False
            for valid_obj in self.valid_objects:
                if valid_obj in action.target_object.name:
                    compound_valid = True
                    break
            
            if not compound_valid:
                errors.append(f"Unknown object type: {action.target_object.name}")
        
        # Validate move actions have destination
        if action.action_type in [ActionType.MOVE, ActionType.RELOCATE]:
            if not action.destination:
                errors.append("Move action requires a destination")
        
        # Validate lighting actions have lighting type
        if action.action_type in [ActionType.RELIGHT, ActionType.ADD_LIGHTING]:
            if not action.lighting_type:
                errors.append("Lighting action requires a lighting type")
        
        # Validate confidence score
        if not 0.0 <= action.confidence <= 1.0:
            errors.append("Confidence score must be between 0.0 and 1.0")
        
        return len(errors) == 0, errors
    
    def merge_similar_actions(self, actions: List[Action]) -> List[Action]:
        """
        Merge similar actions that target the same object.
        
        Args:
            actions: List of actions to merge
            
        Returns:
            List of merged actions
        """
        if not actions:
            return actions
        
        # Group actions by target object
        object_groups = {}
        for action in actions:
            obj_key = action.target_object.name
            if obj_key not in object_groups:
                object_groups[obj_key] = []
            object_groups[obj_key].append(action)
        
        merged_actions = []
        
        for obj_key, obj_actions in object_groups.items():
            if len(obj_actions) == 1:
                merged_actions.append(obj_actions[0])
            else:
                # Merge multiple actions for the same object
                merged_action = self._merge_object_actions(obj_actions)
                merged_actions.append(merged_action)
        
        return merged_actions
    
    def _merge_object_actions(self, actions: List[Action]) -> Action:
        """
        Merge multiple actions for the same object into a single action.
        
        Args:
            actions: List of actions for the same object
            
        Returns:
            Merged action
        """
        if not actions:
            raise ValueError("Cannot merge empty action list")
        
        # Use the first action as base
        base_action = actions[0]
        
        # Merge destinations (take the last one)
        for action in actions:
            if action.destination:
                base_action.destination = action.destination
        
        # Merge lighting types (take the last one)
        for action in actions:
            if action.lighting_type:
                base_action.lighting_type = action.lighting_type
        
        # Average confidence scores
        total_confidence = sum(action.confidence for action in actions)
        base_action.confidence = total_confidence / len(actions)
        
        return base_action
    
    def prioritize_actions(self, actions: List[Action]) -> List[Action]:
        """
        Prioritize actions based on their importance and dependencies.
        
        Args:
            actions: List of actions to prioritize
            
        Returns:
            Prioritized list of actions
        """
        if not actions:
            return actions
        
        # Define action priorities (lower number = higher priority)
        priorities = {
            ActionType.MOVE: 1,
            ActionType.RELOCATE: 1,
            ActionType.REMOVE: 2,
            ActionType.ADD: 3,
            ActionType.RELIGHT: 4,
            ActionType.ADD_LIGHTING: 4,
        }
        
        # Sort by priority
        prioritized = sorted(actions, key=lambda a: priorities.get(a.action_type, 5))
        
        return prioritized
    
    def extract_spatial_constraints(self, actions: List[Action]) -> Dict[str, SpatialReference]:
        """
        Extract spatial constraints from actions.
        
        Args:
            actions: List of actions
            
        Returns:
            Dictionary mapping object names to their spatial constraints
        """
        constraints = {}
        
        for action in actions:
            if action.destination:
                constraints[action.target_object.name] = action.destination
        
        return constraints
    
    def extract_lighting_constraints(self, actions: List[Action]) -> Dict[str, LightingType]:
        """
        Extract lighting constraints from actions.
        
        Args:
            actions: List of actions
            
        Returns:
            Dictionary mapping object names to their lighting constraints
        """
        constraints = {}
        
        for action in actions:
            if action.lighting_type:
                constraints[action.target_object.name] = action.lighting_type
        
        return constraints
    
    def generate_execution_plan(self, actions: List[Action]) -> List[Dict[str, any]]:
        """
        Generate an execution plan from a list of actions.
        
        Args:
            actions: List of actions
            
        Returns:
            List of execution steps
        """
        plan = []
        
        # Validate all actions first
        for action in actions:
            is_valid, errors = self.validate_action(action)
            if not is_valid:
                raise ValueError(f"Invalid action: {errors}")
        
        # Prioritize actions
        prioritized_actions = self.prioritize_actions(actions)
        
        # Generate execution steps
        for i, action in enumerate(prioritized_actions):
            step = {
                "step_id": i + 1,
                "action": action,
                "description": self._generate_step_description(action),
                "dependencies": self._find_dependencies(action, prioritized_actions[:i])
            }
            plan.append(step)
        
        return plan
    
    def _generate_step_description(self, action: Action) -> str:
        """Generate a human-readable description of an action step."""
        if action.action_type in [ActionType.MOVE, ActionType.RELOCATE]:
            return f"Move {action.target_object.name} to {action.destination.value}"
        elif action.action_type in [ActionType.RELIGHT, ActionType.ADD_LIGHTING]:
            return f"Apply {action.lighting_type.value} lighting to {action.target_object.name}"
        else:
            return f"Perform {action.action_type.value} on {action.target_object.name}"
    
    def _find_dependencies(self, action: Action, previous_actions: List[Action]) -> List[int]:
        """Find dependencies for an action based on previous actions."""
        dependencies = []
        
        for i, prev_action in enumerate(previous_actions):
            # Check if this action depends on the previous one
            if (action.target_object.name == prev_action.target_object.name and
                action.action_type != prev_action.action_type):
                dependencies.append(i + 1)
        
        return dependencies


# Example usage
if __name__ == "__main__":
    from .instruction_parser import InstructionParser
    
    parser = InstructionParser()
    extractor = ActionExtractor()
    
    # Test instruction
    instruction = "Move the red car to the left and add sunset lighting to the entire scene"
    parsed = parser.parse(instruction)
    
    print("Original actions:")
    for action in parsed.actions:
        print(f"  - {action.action_type.value}: {action.target_object.name}")
    
    print("\nValidated actions:")
    for action in parsed.actions:
        is_valid, errors = extractor.validate_action(action)
        print(f"  - {action.action_type.value}: {action.target_object.name} - Valid: {is_valid}")
        if errors:
            print(f"    Errors: {errors}")
    
    print("\nExecution plan:")
    plan = extractor.generate_execution_plan(parsed.actions)
    for step in plan:
        print(f"  Step {step['step_id']}: {step['description']}")
        if step['dependencies']:
            print(f"    Dependencies: {step['dependencies']}") 